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
warnings.filterwarnings("ignore")


class ModelCustomFeaturePrep(BaseClass):
    def __init__(self, config=None):
        super().__init__()
        self.name = 'ModelCustomFeaturePrep'
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
        
    @staticmethod
    def calculate_direction_changes(df, price_column='close'):
        df['price_change'] = df[price_column].diff()
        df['direction'] = df['price_change'].apply(lambda x: 'up' if x > 0 else 'down' if x < 0 else 'no change')
        df['direction_change'] = df['direction'].diff().ne(0) & df['direction'].ne('no change')
        direction_changes = df.groupby(['symbol', pd.Grouper(freq='D')])['direction_change'].sum().reset_index()
        direction_changes.rename(columns={'direction_change': 'daily_direction_changes'}, inplace=True)
        return direction_changes
        
    @staticmethod
    def maximas_and_minimas(base_df, window_sizes=[5, 10, 15, 30]):
        for window_size in window_sizes:
            base_df[f'local_max_{window_size}'] = (base_df['close'] >= base_df['close'].rolling(window=window_size, closed='left').max())
            base_df[f'local_min_{window_size}'] = (base_df['close'] <= base_df['close'].rolling(window=window_size, closed='left').min())

            # Cumulative count of rolling local maxima and minima
            base_df[f'cumulative_local_max_{window_size}'] = base_df[f'local_max_{window_size}'].cumsum()
            base_df[f'cumulative_local_min_{window_size}'] = base_df[f'local_min_{window_size}'].cumsum()

            distance_to_last_one = (base_df[f'local_max_{window_size}'].groupby((base_df[f'local_max_{window_size}'] == 1).cumsum()).cumcount()) * (base_df[f'local_max_{window_size}'] == 0) 
            base_df[f'time_since_prev_max_{window_size}'] = distance_to_last_one

            distance_to_last_one = (base_df[f'local_min_{window_size}'].groupby((base_df[f'local_min_{window_size}'] == 1).cumsum()).cumcount()) * (base_df[f'local_min_{window_size}'] == 0) 
            base_df[f'time_since_prev_min_{window_size}'] = distance_to_last_one
        
        base_df['max_today'] = base_df.groupby('day_of_year').high.cummax()
        base_df['min_today'] = base_df.groupby('day_of_year').low.cummin()
        base_df['max_today_session'] = base_df.groupby(['day_of_year', 'market_open']).high.cummax()
        base_df['min_today_session'] = base_df.groupby(['day_of_year', 'market_open']).low.cummin()
            
        return base_df
    
    @staticmethod
    def create_lags(df, lags = range(1, 16), fields = []):
        for lag in lags:
                for price in fields:
                    df[f'{price}_lag_{lag}'] = df[f'{price}'].shift(lag)
        return df
    
    def calculate_all_features(self, base_df, fields_for_lags, path):
            base_df = check_index_is_monotonic_increasing(base_df)
            base_df = self.create_lags(base_df, range(1, 16), fields_for_lags)
            base_df['us_eastern_timestamp'] = base_df.index
            base_df['date'] = base_df.us_eastern_timestamp.dt.date
            base_df['price_change'] = base_df['close'].diff()
            base_df['direction'] = base_df['price_change'].apply(lambda x: 'up' if x > 0 else 'down' if x < 0 else 'no change')
            base_df['direction_prev'] = base_df['direction'].shift()
            base_df['cumulative_ups'] = base_df['direction']=='up'
            base_df['cumulative_ups'] = base_df.groupby('date').cumulative_ups.cumsum()
            base_df['cumulative_downs'] = base_df['direction']=='down'
            base_df['cumulative_downs'] = base_df.groupby('date').cumulative_downs.cumsum()
            base_df['direction_change_up_to_down'] = (base_df.direction=='down')&(base_df.direction_prev=='up')
            base_df['direction_change_down_to_up'] = (base_df.direction=='up')&(base_df.direction_prev=='down')
            base_df['prev_date'] = base_df['date'].shift()
            base_df['prev_cumulative_ups'] = base_df['cumulative_ups'].shift()
            base_df['prev_cumulative_downs'] = base_df['cumulative_downs'].shift()
            base_df['prev_cumulative_ups'] = np.where(base_df.prev_date==base_df.date, np.nan, base_df.prev_cumulative_ups)
            base_df['prev_cumulative_downs'] = np.where(base_df.prev_date==base_df.date, np.nan, base_df.prev_cumulative_downs)
            base_df['prev_cumulative_ups'] = base_df['prev_cumulative_ups'].ffill()
            base_df['prev_cumulative_downs'] = base_df['prev_cumulative_downs'].ffill()
            base_df['hour'] = base_df.us_eastern_timestamp.dt.hour
            base_df['minute'] = base_df.us_eastern_timestamp.dt.minute
            base_df['day_of_year'] = base_df.us_eastern_timestamp.dt.day_of_year
            base_df = self.maximas_and_minimas(base_df, window_sizes=[5, 10, 15, 30])
            base_df = base_df.drop(columns=['us_eastern_timestamp', 'direction_prev', 'date', 'prev_date', 'symbol', ])
            path = path.replace('_avg.parquet', '_otherfeatures.parquet')
            save_df_as_parquet(base_df, path)
            update_config(base_df, path)
            return base_df

    def extract_transform_save(self, base_or_diff='diff'):
        if base_or_diff=='diff':
            
            cols, path = get_paths_and_cols_from_config(self.features_listing_config, dur='1min', file='stock_bars_1min_diff_avg', store_loc='feature_store', )

            base_df = read_df(path)
            base_df = base_df[base_df.symbol=='SPY']
            fields_for_lags = ['open_diff', 'high_diff', 'low_diff', 'close_diff']
        
        elif base_or_diff=='base':
            cols, path = get_paths_and_cols_from_config(self.features_listing_config, dur='1min', file='stock_bars_1min_base_avg', store_loc='feature_store', )
        
            base_df = read_df(path)
            base_df = base_df[base_df.symbol=='SPY']
            fields_for_lags = ['open', 'high', 'low', 'close']
            
        self.calculate_all_features(base_df, fields_for_lags, path)
        
    def run(self):
        for feature_type in ['base', 'diff']:
            self.extract_transform_save(feature_type)
        
if __name__ == '__main__':
    config = load_config()
    model_custom_feature_prep = ModelCustomFeaturePrep(config)
    model_custom_feature_prep.run()