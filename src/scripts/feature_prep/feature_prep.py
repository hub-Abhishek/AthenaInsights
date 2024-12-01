import misc
from misc import utils
from misc.utils import BaseClass, get_alpaca_secrets, load_config, log, read_and_duplicate, read_df, save_df_as_parquet, save_df_as_csv, get_all_paths_from_loc, get_name_and_type
# from misc.features_calc import 

import re
import boto3
import warnings
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")


class FeaturePrep(BaseClass):
    def __init__(self, config=None):
        super().__init__()
        self.name = 'FeaturePrep'
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
        self.stage_order = self.feature_prep['stage_order']
        self.calc_metrics = self.feature_prep['calc_metrics']


    def remove_stationarity(self, prev_step, source_folder_for_next_step):
        log(f'removing stationarity')
        paths = get_all_paths_from_loc(self.client, self.paths_config["s3_bucket"], f'{self.base_folder}/{self.data_folder}/{source_folder_for_next_step}')
        log(paths)
        
        for path in paths:
            log(f"removing stationarity for - {path}")
            df = read_df(path)
            df = df[df.symbol.isin(self.symbols)]
            df = df.reset_index().sort_values(by=['symbol', 'us_eastern_timestamp']).set_index('us_eastern_timestamp')
            symbols = df.symbol.unique()
            for sym in symbols:
                if df[df.symbol==sym].index.is_monotonic_increasing:
                    log(f"The index is sorted for {sym} in the file {path}.")
                else:
                    log(f"The index is not sorted for {sym} in the file {path}.")
                    raise IndexError
            fields = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']
            fields = [f for f in fields if f in df.columns]
            for field in fields:
                df[f'{field}_diff'] = df.groupby('symbol')[field].diff()
            loc = path.replace(source_folder_for_next_step, self.reduced_autocorelation_folder)
            save_df_as_parquet(df, loc)
            log(f'Saved to {loc}. Fields - {df.columns}. Shape - {df.shape}')

    @staticmethod
    def calculate_ma(df, ema=True, sma=True, calc_windows=[]):
        # Function to apply moving averages
        def apply_moving_averages(group):
            for window in tqdm(calc_windows):
                for price in fields:
                    if price=='symbol' or price=='market_open':
                        continue
                    if ema:
                        group[f'{price}_ema_{window}m'] = group[price].ewm(span=window, adjust=False).mean()
                    if sma:
                        group[f'{price}_sma_{window}m'] = group[price].rolling(window=window).mean()
            return group

        # Apply function by group
        fields = list(df.columns)
        return df.groupby('symbol').apply(apply_moving_averages)        
            
    def calculate_moving_averages(self, prev_step, source_folder_for_next_step):
        log(f'calculating moving averages')
        log(f'{prev_step}, source_folder_for_next_step - {source_folder_for_next_step}')
        paths = get_all_paths_from_loc(self.client, self.paths_config["s3_bucket"], f'{self.base_folder}/{self.data_folder}/{source_folder_for_next_step}')
        log(paths)
        for path in paths:            
            name, df_type = get_name_and_type(path)
            log(f"path - {path}, name - {name}, df_type - {df_type}")
            duration = name.split('_')[-1]            
            if 'min' in duration:
                if int(duration.replace('min', '')) <= 30:
                    calc_windows = self.calc_metrics['moving_averages']['minutes']['window_lt_30']
                else:
                    calc_windows = self.calc_metrics['moving_averages']['minutes']['window_lt_30']
            elif 'D' in duration:
                if int(duration.replace('D', '')) < 100:
                    calc_windows = self.calc_metrics['moving_averages']['days']['window_lt_100']
                else: 
                    calc_windows = self.calc_metrics['moving_averages']['days']['window_gt_100']
            
            df = read_df(path)
            df = df[df.symbol.isin(self.symbols)]
            # import pdb;pdb.set_trace();
            if 'reduced_autocorelation' in path:
                df_ma = self.calculate_ma(df[['symbol'] + [x for x in df.columns if '_diff' in x]], 
                                          calc_windows=calc_windows).rename(columns={'symbol': 'symbol1'}).reset_index().set_index('us_eastern_timestamp')
                df = pd.concat([df, df_ma[[x for x in df_ma.columns if x not in df.columns]]], axis=1)
                loc = path.replace('reduced_autocorelation', 'feature_prep').replace('.parquet', '_diff_avg.parquet')
            else:
                df_ma = self.calculate_ma(df, 
                                          calc_windows=calc_windows).rename(columns={'symbol': 'symbol1'}).reset_index().set_index('us_eastern_timestamp')
                df = pd.concat([df, df_ma[[x for x in df_ma.columns if x not in df.columns]]], axis=1)
                loc = path.replace('data_prep', 'feature_prep').replace('.parquet', '_base_avg.parquet')
            # import pdb;pdb.set_trace();
            del df_ma
            log(f'new df.shape - {df.shape}')
            save_df_as_parquet(df, loc)
            
    def calculate_rsi(self, prev_step, source_folder_for_next_step):
        log(f'calculating rsi')
        log(f'{prev_step}, source_folder_for_next_step - {source_folder_for_next_step}')
        paths = get_all_paths_from_loc(self.client, self.paths_config["s3_bucket"], f'{self.base_folder}/{self.data_folder}/{source_folder_for_next_step}', filters='_avg')
        log(paths)
        for path in paths:            
            name, df_type = get_name_and_type(path)
            log(f"path - {path}, name - {name}, df_type - {df_type}")
            duration = name.split('_')[-1]            
        
    
    def extract(self):
        pass
    
    def transform(self):
        # extract and transform
        prev_step = None
        source_folder_for_next_step = self.data_prep_folder
        for feature_name in self.features_to_be_calculated:
            if feature_name=='autocorrelation_removal_v1':
                self.remove_stationarity(prev_step, source_folder_for_next_step)
                prev_step = 'autocorrelation_removal_v1'
                source_folder_for_next_step = self.reduced_autocorelation_folder
                
            elif feature_name=='moving_averages':
                self.calculate_moving_averages(prev_step, source_folder_for_next_step=self.data_prep_folder)
                self.calculate_moving_averages(prev_step, source_folder_for_next_step=self.reduced_autocorelation_folder)
                prev_step = 'moving_averages'
                source_folder_for_next_step = self.feature_prep_folder
                
            elif feature_name=='rsi':
                self.calculate_rsi(prev_step, source_folder_for_next_step=self.feature_prep_folder)
    
    def save(self):
        pass
    
    def run(self):
        pass


if __name__=='__main__':
    config = load_config()
    feature_prep = FeaturePrep(config)
    # feature_prep.run()
    feature_prep.transform()