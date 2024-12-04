import os
import yaml
import pandas as pd
import numpy as np
from misc.utils import log, load_yaml

def calc():
    pass

def update_config(df, loc, model_name='spy_30min_v1', store_loc='feature_store'):
    file_name = loc.split('/')[-1]
    
    # base_class = 'feature_store' if feature_store else 'dependent_variable'
    name = file_name.split('.')[0]
    path = loc
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