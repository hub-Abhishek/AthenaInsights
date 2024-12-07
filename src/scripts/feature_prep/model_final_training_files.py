import misc
from misc import utils
from misc.utils import BaseClass, get_alpaca_secrets, load_config, log, read_and_duplicate, read_df, save_df_as_parquet, save_df_as_csv, get_all_paths_from_loc, get_name_and_type, extract_properties_from_name
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
        self.model_name = self.common_config['model_name']
        self.feature_prep = self.config['technical_yaml']['feature_prep']
        self.final_training_files_config = self.config['technical_yaml']['feature_prep']['final_training_files']

        self.features_to_be_calculated = self.feature_prep['features_to_be_calculated']
        self.symbols = self.feature_prep['symbols']
        self.calc_metrics = self.feature_prep['calc_metrics']
        
        self.features_listing_config = load_features_config()
        
    def extract(self):
        
        name, features_type, base_or_diff, base_duration, base_duration_unit = extract_properties_from_name(self.final_training_files_config['base_features_from_file'])
        base_cols, base_path = get_paths_and_cols_from_config(self.features_listing_config, dur=f'{base_duration}{base_duration_unit}',
                                                              file=name, store_loc='feature_store', )
        base_df = read_df(base_path,)
        num_rows = base_df.shape[0]
        log(num_rows)
        training_file_name, df_type, features_type, base_or_diff, base_duration, base_duration_unit = get_name_and_type(base_path)
        
        # save_df_as_parquet(base_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.model_name}/model/data/{name}.parquet')
        # update_config(base_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.model_name}/model/data/{name}.parquet', store_loc='training_files')
        
        for file_name in self.final_training_files_config['features_from_files']:
            log(f'file name: {file_name}')
            name, features_type, base_or_diff, duration, duration_unit = extract_properties_from_name(file_name)
            log(f'name: {name}, features_type: {features_type}, base_or_diff: {base_or_diff}, duration: {duration}, duration_unit: {duration_unit}')
            cols, path = get_paths_and_cols_from_config(self.features_listing_config, dur=f'{duration}{duration_unit}', file=name, store_loc='feature_store', )
            df = read_df(path, [x for x in cols if x not in base_df.columns])
            log(f'df.shape - {df.shape}')
            if df.shape[0]!= num_rows:
                log(f'skipping for {name}')
                continue
            base_df = pd.concat([base_df, df], axis=1)
            log(f'base_df.shape - {base_df.shape}')
            del df
            new_name, new_df_type, new_features_type, new_base_or_diff, new_duration, new_duration_unit = get_name_and_type(path)
            training_file_name = f'{training_file_name}_{new_base_or_diff}_{new_features_type}' 
            assert base_df.shape[0]==num_rows, "rows not equal after merging"
        
            # save_df_as_parquet(base_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.model_name}/model/data/{training_file_name}.parquet')
            # update_config(base_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.model_name}/model/data/{training_file_name}.parquet', store_loc='training_files', time_period='1min')
        
        name, features_type, base_or_diff, duration, duration_unit = extract_properties_from_name(self.final_training_files_config['dependent_variable_from_file'])
        cols, path = get_paths_and_cols_from_config(self.features_listing_config, dur=f'{duration}{duration_unit}', file=name, store_loc='dependent_variable', )
        dependent_df = read_df(path, [x for x in cols if x not in base_df.columns])
        log(f'dependent_df.shape - {dependent_df.shape}')
        # log(f'dependent_df.columns - {dependent_df.columns}')
        if dependent_df.shape[0]!= num_rows:
            log(f'skipping for dependent_df')
            raise Exception('dependent variable shape not the same')
        base_df = pd.concat([base_df, dependent_df], axis=1)
        log(f'base_df.shape - {base_df.shape}')
        del dependent_df
        new_name, new_df_type, new_features_type, new_base_or_diff, new_duration, new_duration_unit = get_name_and_type(path)
        training_file_name = f'{training_file_name}_{new_base_or_diff}_{new_features_type}' 
        assert base_df.shape[0]==num_rows, "rows not equal after other features"

        columns_to_be_dropped = base_df.select_dtypes('object').columns
        if len(columns_to_be_dropped) > 0:
            log(f'dropping columns - {columns_to_be_dropped}')
            base_df = base_df.drop(columns=columns_to_be_dropped)
        
        
        save_df_as_parquet(base_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.model_name}/model/data/{training_file_name}.parquet')
        update_config(base_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.model_name}/model/data/{training_file_name}.parquet', store_loc='training_files', time_period=f'{base_duration}{base_duration_unit}')
    
    def run(self, ):
        self.extract()
        
if __name__ == '__main__':
    config = load_config()
    model_training_file_prep = ModelTrainingFilePrep(config)
    model_training_file_prep.run()