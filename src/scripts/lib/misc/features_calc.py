import os
import yaml
import pandas as pd
import numpy as np
from misc.utils import log, load_yaml
from sklearn.linear_model import LinearRegression

def calc():
    pass

def update_config(df, loc, model_name='spy_30min_v1', store_loc='feature_store', time_period=None):
    file_name = loc.split('/')[-1]
    
    # base_class = 'feature_store' if feature_store else 'dependent_variable'
    name = file_name.split('.')[0]
    path = loc
    if time_period is None:
        time_period = name.split('_')[-3]
    columns = list(df.columns)
    
    config_loc = f'config/{model_name}/features.yaml'
    
    if os.path.exists(config_loc):
        log(f'reading from {config_loc}')
        features_config = load_yaml(config_loc)
        if store_loc in features_config.keys():
            if time_period in features_config[store_loc].keys():
                features_config[store_loc][time_period][name] = {'cols': columns, 'path': path}
                if name in features_config[store_loc][time_period].keys():
                    log(f'updating {store_loc} - {time_period} - {name} - already existed, replaced with new values')
                else:
                    log(f'updating {store_loc} - {time_period} - name - {name} - didnt exist')
            else:
                features_config[store_loc][time_period] = {name: {'cols': columns, 'path': path}}
                log(f'updating {store_loc} - time_period - {time_period}- didnt exist')
        else:
            features_config[store_loc] = {time_period:{name: {'cols': columns, 'path': path}}}
            log(f'updating - store_loc {store_loc} didnt exist')
            
    else:
        features_config = {store_loc: {time_period:{name: {'cols': columns, 'path': path}}}}
        log(f'config didnt exist')
    
    with open(config_loc, 'w+') as ff:
        yaml.dump(features_config, ff)
    log(f'updated {config_loc}')
    
def load_features_config(model_name='spy_30min_v1'):
    config_loc = f'config/{model_name}/features.yaml'
    if os.path.exists(config_loc):
        log(f'reading from {config_loc}')
        features_config = load_yaml(config_loc)
        return features_config
    else:
        return {'dependent_variable': None, 'feature_store': None}
    
def get_paths_and_cols_from_config(features_listing_config, dur='1min', file='stock_bars_1min_diff_avg', store_loc='feature_store', ):
    cols = features_listing_config[store_loc][dur][file]['cols']
    path = features_listing_config[store_loc][dur][file]['path']
    return cols, path

def check_index_is_monotonic_increasing(df):
    # Check if the Datetime index is sorted
    if df.index.is_monotonic_increasing:
        log("The index is sorted.")
        return df
    else:
        log("The index is not sorted. Sorting now.")
        df.sort_index(inplace=True)
        return df
    
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