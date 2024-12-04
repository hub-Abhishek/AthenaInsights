import misc
from misc import utils
from misc.utils import BaseClass, get_alpaca_secrets, load_config, log, read_and_duplicate, read_df, save_df_as_parquet, save_df_as_csv, get_all_paths_from_loc, get_name_and_type
from misc.features_calc import update_config, load_features_config, get_paths_and_cols_from_config, check_index_is_monotonic_increasing
# from misc.features_calc import 

import re
import boto3
import warnings
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
warnings.filterwarnings("ignore")


def calculate_trend_slope(df, window=20, field='close'):
    """ Calculate the slope of the linear regression line for the last 'window' minutes based on a specified field """
    reg = LinearRegression()
    # Indices for X, specified field values for Y
    x = np.array(range(window)).reshape(-1, 1)
    y = df[field].values.reshape(-1, 1)
    reg.fit(x, y)
    # Slope of the regression line
    return reg.coef_[0][0]

def categorize_points(df, field='close', prev_data_points=20, positive_slope_threshold=0.0, negative_slope_threshold=0.0, positive_rise_threshold=0.0003, negative_drop_threshold=0.0003, positive_future_window=30, negative_future_window=30):
    """ Categorize each minute data point into A, B, or C with dynamic thresholds and fields """
    categories = []
    future_highs = []
    future_lows = []
    slopes = []

    for i in range(len(df)):
        if i < prev_data_points or i > len(df) - max(positive_future_window, negative_future_window):  # Not enough data to categorize
            categories.append('C')  # Consider as undecided for now
            future_highs.append(np.nan)
            future_lows.append(np.nan)
            slopes.append(np.nan)
            continue
        
        # Calculate the trend over the past 20 minutes using the specified field
        past_trend_slope = calculate_trend_slope(df.iloc[i-prev_data_points:i], window=prev_data_points,field=field)
        slopes.append(past_trend_slope)
        
        # Get the current price and future high/low based on the specified field
        current_price = df.iloc[i][field]
        future_high = df.iloc[i+1:i+1+positive_future_window][field].max()
        future_low = df.iloc[i+1:i+1+negative_future_window][field].min()
        future_highs.append(future_high)
        future_lows.append(future_low)
        
        # Calculate thresholds based on current price
        high_threshold = current_price * (1 + positive_rise_threshold)
        low_threshold = current_price * (1 - negative_drop_threshold)
        
        # Determine the category based on the criteria and trend
        if past_trend_slope < negative_slope_threshold and future_high > high_threshold:
            categories.append('A')
        elif past_trend_slope > positive_slope_threshold and future_low < low_threshold:
            categories.append('B')
        else:
            categories.append('C')
    
    return categories, future_highs, future_lows, slopes


class ModelDependentFeaturePrep(BaseClass):
    def __init__(self, config=None):
        super().__init__()
        self.name = 'ModelDependentFeaturePrep'
        log(f'running {self.name}')
        if config is None:
            self.config = load_config()
        else:
            self.config = config

        self.paths_config = self.config['paths_config']

        self.bucket_loc = f's3://{self.paths_config["s3_bucket"]}'
        self.base_folder = self.paths_config["base_folder"]
        self.data_folder = self.paths_config["data_folder"]
        self.data_prep_folder = self.paths_config["data_prep_folder"]
        self.reduced_autocorelation_folder = self.paths_config["reduced_autocorelation_folder"]
        self.feature_prep_folder = self.paths_config["feature_prep_folder"]
        self.client = boto3.client('s3')

        self.common_config = self.config['technical_yaml']['common']
        self.feature_prep = self.config['technical_yaml']['feature_prep']

        self.features_to_be_calculated = self.feature_prep['features_to_be_calculated']
        self.symbols = self.feature_prep['symbols']
        self.calc_metrics = self.feature_prep['calc_metrics']
        
        self.features_listing_config = load_features_config()
        
    def extract(self):
        cols, self.path = get_paths_and_cols_from_config(self.features_listing_config, dur='1min', file='stock_bars_1min_base_avg', store_loc='feature_store', )
        base_df = read_df(self.path)
        base_df = base_df[base_df.symbol=='SPY']
        base_df = check_index_is_monotonic_increasing(base_df)
        fields = list(set(['open', 'high', 'low', 'close'] + [x for x in base_df.columns if 'close' in x]))
        base_df = base_df[fields]
        return {"base_df": base_df}
        
    def transform(self, dfs):
        base_df = dfs["base_df"]
        field = self.feature_prep['dependent_var']['based_on']
        
        base_df['category'] = ''
        base_df['future_highs'] = np.nan
        base_df['future_lows'] = np.nan
        base_df['slopes'] = np.nan
        base_df = base_df[['open', 'high', 'low', 'close', 'close_sma_5m', 'category', 'future_highs', 'future_lows', 'slopes']]
        base_df[field] = np.where(base_df[field].isna(), base_df.close, base_df[field])
        
        
        dates = list(set(base_df.index.date))

        for date in tqdm(dates):
            df_day = base_df.loc[date.strftime('%Y-%m-%d')]
            categories, future_highs, future_lows, slopes = categorize_points(
                df_day, 
                field=field, 
                prev_data_points=self.feature_prep['dependent_var']['prev_data_points'], 
                positive_slope_threshold=self.feature_prep['dependent_var']['positive_slope_threshold'], 
                negative_slope_threshold=self.feature_prep['dependent_var']['negative_slope_threshold'], 
                positive_rise_threshold=self.feature_prep['dependent_var']['positive_rise_threshold'], 
                negative_drop_threshold=self.feature_prep['dependent_var']['negative_drop_threshold'], 
                positive_future_window=self.feature_prep['dependent_var']['positive_future_window'], 
                negative_future_window=self.feature_prep['dependent_var']['negative_future_window'])

            base_df.loc[date.strftime('%Y-%m-%d'), 'category'] = categories
            base_df.loc[date.strftime('%Y-%m-%d'), 'future_highs'] = future_highs
            base_df.loc[date.strftime('%Y-%m-%d'), 'future_lows'] = future_lows
            base_df.loc[date.strftime('%Y-%m-%d'), 'slopes'] = slopes
            del categories, future_highs, future_lows, df_day
        return {"base_df": base_df}
    
    def save(self, dfs):
        base_df = dfs["base_df"]
        # 's3://sisyphus-general-bucket/AthenaInsights/latest_data/dependent_variable/stock_bars_1min.parquet'
        save_df_as_parquet(base_df, self.path.replace('feature_prep', 'dependent_variable'))
        update_config(base_df, self.path.replace('feature_prep', 'dependent_variable'), store_loc='dependent_variable')

        
    def run(self):
        dfs = self.extract()
        dfs = self.transform(dfs)
        self.save(dfs)
        
if __name__ == '__main__':
    config = load_config()
    model_dependent_feature_prep = ModelDependentFeaturePrep(config)
    model_dependent_feature_prep.run()