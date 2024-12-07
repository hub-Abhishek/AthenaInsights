import misc
from misc import utils
from misc.utils import BaseClass, get_alpaca_secrets, load_config, log, read_and_duplicate, read_df, save_df_as_parquet, save_df_as_csv, get_all_paths_from_loc, get_name_and_type
from misc.features_calc import update_config, load_features_config
# from misc.features_calc import 

import re
import boto3
import warnings
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")


class TechnicalFeaturePrep(BaseClass):
    def __init__(self, config=None):
        super().__init__()
        self.name = 'TechnicalFeaturePrep'
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

        self.features_to_be_calculated = self.feature_prep['features_to_be_calculated']
        self.symbols = self.feature_prep['symbols']
        self.calc_metrics = self.feature_prep['calc_metrics']


    def remove_stationarity(self, prev_step, source_folder_for_next_step):
        log('removing stationarity')
        paths = get_all_paths_from_loc(self.client, self.paths_config["s3_bucket"], f'{self.base_folder}/{self.data_folder}/{self.model_name}/{source_folder_for_next_step}')
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
            update_config(df, loc)
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
        log('calculating moving averages')
        log(f'{prev_step}, source_folder_for_next_step - {source_folder_for_next_step}')
        paths = get_all_paths_from_loc(self.client, self.paths_config["s3_bucket"], f'{self.base_folder}/{self.data_folder}/{self.model_name}/{source_folder_for_next_step}')
        log(paths)
        for path in paths:
            name, df_type, features_type, base_or_diff, duration, duration_unit = get_name_and_type(path)
            log(f"path - {path}, name - {name}, df_type - {df_type}, features_type - {features_type}, base_or_diff - {base_or_diff}, duration - {duration}, duration_unit - {duration_unit}")
            if 'min' == duration_unit:
                if duration <= 30:
                    calc_windows = self.calc_metrics['avg']['minutes']['window_lt_30']
                else:
                    calc_windows = self.calc_metrics['avg']['minutes']['window_lt_30']
            elif 'D' == duration_unit:
                if duration < 100:
                    calc_windows = self.calc_metrics['avg']['days']['window_lt_100']
                else: 
                    calc_windows = self.calc_metrics['avg']['days']['window_gt_100']

            df = read_df(path)
            df = df[df.symbol.isin(self.symbols)]
            # import pdb;pdb.set_trace();
            if 'reduced_autocorelation' in path:
                df_ma = self.calculate_ma(df[['symbol'] + [x for x in df.columns if '_diff' in x]], 
                                          calc_windows=calc_windows).rename(columns={'symbol': 'symbol1'}).reset_index().set_index('us_eastern_timestamp')
                df = pd.concat([df, df_ma[[x for x in df_ma.columns if x not in df.columns]]], axis=1)
                loc = path.replace('reduced_autocorelation', f'feature_prep').replace('.parquet', '_diff_avg.parquet')
            else:
                df_ma = self.calculate_ma(df, 
                                          calc_windows=calc_windows).rename(columns={'symbol': 'symbol1'}).reset_index().set_index('us_eastern_timestamp')
                df = pd.concat([df, df_ma[[x for x in df_ma.columns if x not in df.columns]]], axis=1)
                loc = path.replace('data_prep', f'feature_prep').replace('.parquet', '_base_avg.parquet')
            # import pdb;pdb.set_trace();
            del df_ma
            log(f'new df.shape - {df.shape}')
            save_df_as_parquet(df, loc)
            update_config(df, loc) # [name, time_period, path, columns]
        
    @staticmethod
    def calculate_rsi_for_field(data, window=14):
        # Calculate price differences
        delta = data.diff()
        # Make two series: one for gains and one for losses
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        # Calculate the Exponential Moving Average of gains and losses
        avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

        # Calculate the RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def read_and_calculate_rsi(self, path, windows, fields, features_type):

        print(f'Reading from {path} for fields - {fields}')
        df = read_df(path)
        log(f'columns found - {df.columns}')
        df = df[fields + ['symbol']]
        grouped = df.groupby('symbol')
        results = []
        for symbol, group in grouped:
            calculated_fields = []
            for col in fields:
                for window in windows:
                    rsi_values = self.calculate_rsi_for_field(group[col], window)
                    group[f'{col}_{window}window_rsi'] = rsi_values
                    calculated_fields.append(f'{col}_{window}window_rsi')
            results.append(group[fields + calculated_fields + ['symbol']])
            # Concatenate all the grouped results back into a single DataFrame
        df_concat = pd.concat(results)
        loc = path.replace(f'_{features_type}.parquet', '_rsi.parquet')
        print(f'Saving to {loc}')
        # df_concat.to_parquet(loc)
        del df, grouped, group, rsi_values, results
        save_df_as_parquet(df_concat, loc)
        update_config(df_concat, loc)
        del df_concat

    def calculate_rsi(self, prev_step, source_folder_for_next_step):
        log('calculating rsi')
        log(f'{prev_step}, source_folder_for_next_step - {source_folder_for_next_step}')
        
        paths = get_all_paths_from_loc(self.client, self.paths_config["s3_bucket"], f'{self.base_folder}/{self.data_folder}/{self.model_name}/{source_folder_for_next_step}', filters=f'_{prev_step}.parquet')
        log(paths)
        
        for path in paths:
            name, df_type, features_type, base_or_diff, duration, duration_unit = get_name_and_type(path)
            log(f"path - {path}, name - {name}, df_type - {df_type}, features_type - {features_type}, base_or_diff - {base_or_diff}, duration - {duration}, duration_unit - {duration_unit}")

            if duration_unit=='min':
                if duration <= 10:
                    windows, fields = self.calc_metrics['rsi']['minutes']['window_lt_10']
                elif duration <= 30:
                    windows, fields = self.calc_metrics['rsi']['minutes']['window_lt_30']
                elif duration < 60:
                    windows, fields = self.calc_metrics['rsi']['minutes']['window_lt_60']
                else:
                    windows, fields = self.calc_metrics['rsi']['minutes']['window_gt_60']
            elif duration_unit=='D':
                if duration < 100:
                    windows, fields = self.calc_metrics['rsi']['days']['window_lt_100']
                else: 
                    windows, fields = self.calc_metrics['rsi']['days']['window_gt_100']
            
            if base_or_diff=='diff':
                fields = [f.replace('close', 'close_diff') for f in fields]
            self.read_and_calculate_rsi(path, windows, fields, features_type)
            
    @staticmethod
    def calculate_macd_for_field(df, signal=14, ema_columns=[]):
        for i in range(len(ema_columns)):
            for j in range(i + 1, len(ema_columns)):
                fast_ema = ema_columns[i]
                slow_ema = ema_columns[j]

                # Calculate MACD
                # macd_col_name = f'MACD_{fast_ema}_{slow_ema}'
                # df[macd_col_name] = df[fast_ema] - df[slow_ema]

                # Calculate Signal line
                signal_col_name = f'Signal_{fast_ema}_{slow_ema}_signal{signal}'
                # df[signal_col_name] = df[macd_col_name].ewm(span=signal, adjust=False).mean()
                df[signal_col_name] = (df[fast_ema] - df[slow_ema]).ewm(span=signal, adjust=False).mean()


                # # Calculate Histogram
                # histogram_col_name = f'Histogram_{fast_ema}_{slow_ema}_signal{signal}'
                # df[histogram_col_name] = df[macd_col_name] - df[signal_col_name]
        return df

    def read_and_calculate_macd(self, path, signals, features_type):
        print(f'Reading from {path}')
        df = read_df(path)
        # log(f'columns found - {df.columns}')
        # log(df.head())
        fields = [z for z in df.columns if 'close_ema' in z or 'close_diff_ema' in z]
        df = df[['symbol'] + fields]
        grouped = df.groupby('symbol')
        results = []

        for symbol, group in grouped:
            ema_columns = [z for z in group.columns if z!='symbol']
            ema_columns = sorted(ema_columns, key=lambda x: int(re.search(r'\d+', x).group()))
            print(f'symbol - {symbol} for signals = {signals}')
            for signal in signals:
                # print(f'for signal = {signal}')
                group = self.calculate_macd_for_field(group, signal, ema_columns)
            results.append(group)

        # Concatenate all the grouped results back into a single DataFrame
        df = pd.concat(results)
        loc = path.replace(f'_{features_type}.parquet', '_macd.parquet')
        print(f'Saving to {loc}')
        save_df_as_parquet(df, loc)
        update_config(df, loc)
        del df, group, results
            
    def calculate_macd(self, prev_step, source_folder_for_next_step):
        log('calculating rsi')
        log(f'{prev_step}, source_folder_for_next_step - {source_folder_for_next_step}')
        
        paths = get_all_paths_from_loc(self.client, self.paths_config["s3_bucket"], f'{self.base_folder}/{self.data_folder}/{self.model_name}/{source_folder_for_next_step}', filters=f'_{prev_step}.parquet')
        log(paths)
        
        for path in paths:
            name, df_type, features_type, base_or_diff, duration, duration_unit = get_name_and_type(path)
            log(f"path - {path}, name - {name}, df_type - {df_type}, features_type - {features_type}, base_or_diff - {base_or_diff}, duration - {duration}, duration_unit - {duration_unit}")

            if duration_unit=='min':
                if duration<=10:
                    signals = self.calc_metrics['macd']['minutes']['window_lt_10']
                elif duration<=30:
                    signals = self.calc_metrics['macd']['minutes']['window_lt_30']
                else:
                    signals = self.calc_metrics['macd']['minutes']['window_gt_30']
            elif duration_unit=='D':
                if duration< 100:
                    signals = self.calc_metrics['macd']['days']['window_lt_100']
                else: 
                    signals = self.calc_metrics['macd']['days']['window_gt_100']

            self.read_and_calculate_macd(path, signals, features_type)

    def extract_transform_save(self):
        # extract and transform
        prev_step = None
        source_folder_for_next_step = self.data_prep_folder
        for feature_name in self.features_to_be_calculated:
            if feature_name=='autocorrelation_removal_v1':
                self.remove_stationarity(prev_step, source_folder_for_next_step)
                prev_step = feature_name
                source_folder_for_next_step = self.reduced_autocorelation_folder
                # import pdb; pdb.set_trace();

            elif feature_name=='avg':
                self.calculate_moving_averages(prev_step, source_folder_for_next_step=self.data_prep_folder)
                self.calculate_moving_averages(prev_step, source_folder_for_next_step=self.reduced_autocorelation_folder)
                prev_step = feature_name
                source_folder_for_next_step = self.feature_prep_folder

            elif feature_name=='rsi':
                self.calculate_rsi(prev_step, source_folder_for_next_step=self.feature_prep_folder)
                prev_step = feature_name
                source_folder_for_next_step = self.feature_prep_folder

            elif feature_name=='macd':
                self.calculate_macd(prev_step, source_folder_for_next_step=self.feature_prep_folder)
                prev_step = feature_name
                source_folder_for_next_step = self.feature_prep_folder
                
            # elif feature_name=='other_features':
            #     self.calculate_other_features(prev_step, source_folder_for_next_step=self.feature_prep_folder)
            #     prev_step = feature_name
            #     source_folder_for_next_step = self.feature_prep_folder
            
            else:
                log(feature_name)
                

    def run(self):
        self.extract_transform_save()


if __name__ == '__main__':
    config = load_config()
    technical_feature_prep = TechnicalFeaturePrep(config)
    technical_feature_prep.run()