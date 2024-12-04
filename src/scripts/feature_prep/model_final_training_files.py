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


class ModelTrainingFilePrep(BaseClass):
    def __init__(self, config=None):
        super().__init__()
        self.name = 'ModelTrainingFilePrep'
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
        
        base_cols, base_path = get_paths_and_cols_from_config(self.features_listing_config, dur='1min', file='stock_bars_1min_base_avg', store_loc='feature_store', )
        base_df = read_df(base_path,)
        num_rows = base_df.shape[0]
        log(num_rows)
        name, df_type, features_type, base_or_diff, duration, duration_unit = get_name_and_type(base_path)
        
        save_df_as_parquet(base_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/model/data/{name}.parquet')
        update_config(base_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/model/data/{name}.parquet', store_loc='training_files')
        
        cols, path = get_paths_and_cols_from_config(self.features_listing_config, dur='1min', file='stock_bars_1min_base_rsi', store_loc='feature_store', )
        rsi_df = read_df(path, [x for x in cols if x not in base_df.columns])
        log(f'rsi_df.shape - {rsi_df.shape}')
        base_df = pd.concat([base_df, rsi_df], axis=1)
        log(f'base_df.shape - {base_df.shape}')
        del rsi_df
        new_name, new_df_type, new_features_type, new_base_or_diff, new_duration, new_duration_unit = get_name_and_type(path)
        name = f'{name}_{new_base_or_diff}_{new_features_type}' 
        assert base_df.shape[0]==num_rows, "rows not equal after rsi"
        
        save_df_as_parquet(base_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/model/data/{name}.parquet')
        update_config(base_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/model/data/{name}.parquet', store_loc='training_files')
                
        cols, path = get_paths_and_cols_from_config(self.features_listing_config, dur='1min', file='stock_bars_1min_base_macd', store_loc='feature_store', )
        macd_df = read_df(path, [x for x in cols if x not in base_df.columns])
        log(f'macd_df.shape - {macd_df.shape}')
        base_df = pd.concat([base_df, macd_df], axis=1)
        log(f'base_df.shape - {base_df.shape}')
        del macd_df
        new_name, new_df_type, new_features_type, new_base_or_diff, new_duration, new_duration_unit = get_name_and_type(path)
        name = f'{name}_{new_base_or_diff}_{new_features_type}' 
        assert base_df.shape[0]==num_rows, "rows not equal after macd"
        
        save_df_as_parquet(base_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/model/data/{name}.parquet')
        update_config(base_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/model/data/{name}.parquet', store_loc='training_files')
        
        cols, path = get_paths_and_cols_from_config(self.features_listing_config, dur='1min', file='stock_bars_1min_base_otherfeatures', store_loc='feature_store', )
        otherfeatures_df = read_df(path, [x for x in cols if x not in base_df.columns])
        log(f'otherfeatures_df.shape - {otherfeatures_df.shape}')
        log(f'otherfeatures_df.columns - {otherfeatures_df.columns}')
        base_df = pd.concat([base_df, otherfeatures_df], axis=1)
        log(f'base_df.shape - {base_df.shape}')
        del otherfeatures_df
        new_name, new_df_type, new_features_type, new_base_or_diff, new_duration, new_duration_unit = get_name_and_type(path)
        name = f'{name}_{new_base_or_diff}_{new_features_type}' 
        assert base_df.shape[0]==num_rows, "rows not equal after other features"
        
        save_df_as_parquet(base_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/model/data/{name}.parquet')
        update_config(base_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/model/data/{name}.parquet', store_loc='training_files')
        
        cols, path = get_paths_and_cols(self.features_listing_config, dur='1min', file='stock_bars_1min', store_loc='dependent_variable', )
        dependent_df = read_df(path, [x for x in cols if x not in base_df.columns])
        log(f'dependent_df.shape - {dependent_df.shape}')
        log(f'dependent_df.columns - {dependent_df.columns}')
        base_df = pd.concat([base_df, dependent_df], axis=1)
        log(f'base_df.shape - {base_df.shape}')
        del dependent_df
        new_name, new_df_type, new_features_type, new_base_or_diff, new_duration, new_duration_unit = get_name_and_type(path)
        name = f'{name}_{new_base_or_diff}_{new_features_type}' 
        assert base_df.shape[0]==num_rows, "rows not equal after other features"
        
        save_df_as_parquet(base_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/model/data/{name}.parquet')
        update_config(base_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/model/data/{name}.parquet', store_loc='training_files')
    
    def run(self, ):
        self.extract()
        
if __name__ == '__main__':
    config = load_config()
    model_training_file_prep = ModelTrainingFilePrep(config)
    model_training_file_prep.run()