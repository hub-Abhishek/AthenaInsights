import misc
from misc import utils
from misc.utils import BaseClass, get_alpaca_secrets, load_config, log, read_and_duplicate, save_df_as_parquet, read_df

import re
import pandas as pd
import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import datetime


class DataPrep(BaseClass):
    def __init__(self, config=None):
        super().__init__()
        self.name = 'DataPrep'
        log(f'running {self.name}')
        if config is None:
            self.config = load_config()
        else:
            self.config = config
        self.common_config = self.config['technical_yaml']['common']
        self.data_prep_technical_config = self.config['technical_yaml']['data_prep']
        self.paths_config = self.config['paths_config']
        self.alpaca_key, self.alpaca_secret = get_alpaca_secrets()

        self.bucket_loc = f's3://{self.paths_config["s3_bucket"]}'
        self.base_folder = self.paths_config["base_folder"]
        self.data_folder = self.paths_config["data_folder"]


    @staticmethod
    def process_dataset_min(df, datasets=None, durations=[]):
        df_backup = df.copy()
        df_backup = df_backup.reset_index()
        df_backup['date'] = df_backup['us_eastern_timestamp'].dt.date
        df_backup['hr'] = df_backup['us_eastern_timestamp'].dt.hour
        df_backup['min'] = df_backup['us_eastern_timestamp'].dt.minute

        for dur in durations:
            new_dataset = df.resample(dur).agg({'symbol': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'trade_count': 'sum', 'market_open': 'max'})
            new_dataset = new_dataset.reset_index()

            new_dataset['date'] = new_dataset['us_eastern_timestamp'].dt.date
            new_dataset['hr'] = new_dataset['us_eastern_timestamp'].dt.hour
            new_dataset['min'] = new_dataset['us_eastern_timestamp'].dt.minute

            new_dataset = new_dataset.merge(df_backup[['date', 'hr', 'min']], on=['date', 'hr', 'min'])

            datasets[dur] = new_dataset.set_index('us_eastern_timestamp').drop(columns=['date', 'hr', 'min']).copy()
            log(f'{dur} - {df.columns}')
        return datasets


    @staticmethod
    def process_dataset_hr(df, datasets=None, durations=[]):
        df_backup = df.copy()
        df_backup = df_backup.reset_index()
        df_backup['date'] = df_backup['us_eastern_timestamp'].dt.date
        df_backup['hr'] = df_backup['us_eastern_timestamp'].dt.hour
        df_backup['min'] = df_backup['us_eastern_timestamp'].dt.minute

        for dur in durations:
            new_dataset = df.resample(dur).agg({'symbol': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'trade_count': 'sum', 'market_open': 'max'})
            new_dataset = new_dataset.reset_index()

            new_dataset['date'] = new_dataset['us_eastern_timestamp'].dt.date
            new_dataset['hr'] = new_dataset['us_eastern_timestamp'].dt.hour
            new_dataset['min'] = new_dataset['us_eastern_timestamp'].dt.minute

            new_dataset = new_dataset.merge(df_backup[['date', 'hr', 'min']], on=['date', 'hr', 'min'])

            datasets[dur] = new_dataset.set_index('us_eastern_timestamp').drop(columns=['date', 'hr', 'min']).copy()
            log(f'{dur} - {df.columns}')
        return datasets

    @staticmethod
    def process_dataset_day(df, datasets=None, durations=[]):
        for dur in durations:
            new_dataset = df.resample(dur).agg({'symbol': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            datasets[dur] = new_dataset.copy()
            log(f'{dur} - {df.columns}')
        return datasets

    @staticmethod
    def fill_missing_minutes(df, freq='1T'):
        # Set datetime as the index temporarily
        df.set_index('us_eastern_timestamp', inplace=True)

        # Function to resample each group while respecting daily bounds
        def resample_group(group):
            # Group by each day to respect daily boundaries
            daily_groups = []
            for name, day_group in group.groupby(group.index.date):
                min_time = day_group.index.min()
                max_time = day_group.index.max()

                # Resample within the day's min and max times
                resampled = day_group.resample(freq).ffill()
                resampled = resampled[(resampled.index >= min_time) & (resampled.index <= max_time)]

                # Fill missing data within the day
                resampled['open'].fillna(resampled['close'], inplace=True)
                resampled['high'].fillna(resampled['close'], inplace=True)
                resampled['low'].fillna(resampled['close'], inplace=True)
                resampled['volume'].fillna(0, inplace=True)
                resampled['trade_count'].fillna(0, inplace=True)
                resampled['vwap'].fillna(resampled['close'], inplace=True)

                daily_groups.append(resampled)

            # Combine all daily resampled groups
            return pd.concat(daily_groups)

        # Apply the resampling function to each symbol group
        filled_df = df.groupby('symbol').apply(resample_group)

        # Clean up the index
        filled_df.reset_index(level=0, drop=True, inplace=True)

        return filled_df.reset_index()

    def extract(self):
        # cal = read_df(cal, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.paths_config["csv_folder"]}/calender.csv')

        alpaca_current_min_level_data = read_df(f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.paths_config["parquet_folder"]}/stock_bars_minute.parquet')
        log(f'alpaca_current_min_level_data shape: {alpaca_current_min_level_data.shape}')

        alpaca_current_hour_level_data = read_df(f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.paths_config["parquet_folder"]}/stock_bars_hour.parquet')
        log(f'alpaca_current_hour_level_data shape: {alpaca_current_hour_level_data.shape}')

        alpaca_current_day_level_data = read_df(f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.paths_config["parquet_folder"]}/stock_bars_day.parquet')
        log(f'alpaca_current_day_level_data shape: {alpaca_current_day_level_data.shape}')

        return {
            # "cal": cal,
            "alpaca_current_min_level_data": alpaca_current_min_level_data,
            "alpaca_current_hour_level_data": alpaca_current_hour_level_data,
            "alpaca_current_day_level_data": alpaca_current_day_level_data
        }

    @staticmethod
    def process_timestamps(df):
        assert df.open.isna().sum() == 0, 'Open values missing!'
        assert df.high.isna().sum() == 0, 'High values missing!'
        assert df.low.isna().sum() == 0, 'Low values missing!'
        assert df.close.isna().sum() == 0, 'Close values missing!'

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['us_eastern_timestamp'] = df['timestamp'].dt.tz_convert('US/Eastern')
        df['us_eastern_timestamp'] = df['us_eastern_timestamp'].dt.tz_localize(None)
        df = df.drop(columns='timestamp')
        return df

    def transform(self, dfs):

        symbols = self.common_config['symbols']
        datasets = {}

        for df_name in dfs.keys():
            dfs[df_name] = self.process_timestamps(dfs[df_name])

        for sym in symbols:
            datasets[sym] = {}
            for df_name in dfs.keys():
                df = dfs[df_name][dfs[df_name].symbol==sym].copy()
                if 'min_level_data' in df_name:
                    
                    df = self.fill_missing_minutes(df)
                    df['market_open'] = (
                        (df.us_eastern_timestamp.dt.time>=pd.to_datetime('09:30:00').time())
                        & (df.us_eastern_timestamp.dt.time < pd.to_datetime('16:00:00').time()))
                    df.set_index('us_eastern_timestamp', inplace=True)
                    datasets[sym]['1min'] = df.copy()
                    log(f'1min - {df.columns}')
                    datasets[sym] = self.process_dataset_min(df, datasets[sym], durations=self.data_prep_technical_config['durations']['minute_level'])

                elif 'hour_level_data' in df_name:

                    df['market_open'] = (
                        (df.us_eastern_timestamp.dt.time>=pd.to_datetime('09:30:00').time())
                        & (df.us_eastern_timestamp.dt.time < pd.to_datetime('16:00:00').time()))
                    df.set_index('us_eastern_timestamp', inplace=True)
                    datasets[sym]['60min'] = df.copy()
                    log(f'60min - {df.columns}')
                    datasets[sym] = self.process_dataset_hr(df, datasets[sym], durations=self.data_prep_technical_config['durations']['hour_level'])

                elif 'day_level_data' in df_name:

                    df.set_index('us_eastern_timestamp', inplace=True)
                    datasets[sym]['1D'] = df.copy()
                    log(f'1D - {df.columns}')
                    datasets[sym] = self.process_dataset_day(df, datasets[sym], durations=self.data_prep_technical_config['durations']['day_level'])
        
        for sym in datasets.keys():
            for k in datasets[sym].keys():
                log(f'{k} - {datasets[sym][k].shape}')
            
        return datasets


    def save(self, dfs):
        all_durations = []
        for sym in dfs.keys():
            for dur in dfs[sym].keys():
                all_durations.append(dur)
        all_durations = set(all_durations)
        log(f'all_durations: {all_durations}')

        for dur in all_durations:
            dur_df = pd.DataFrame()
            for sym in dfs.keys():
                if dur in dfs[sym].keys():
                    dur_df = pd.concat([dur_df, dfs[sym][dur].assign(symbol=sym)])
            # dur_df.to_parquet(f'{self.bucket_loc}/{self.data_folder}/data_prep/stock_bars_{dur}.parquet')
            save_df_as_parquet(dur_df, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/data_prep/stock_bars_{dur}.parquet')


if __name__=='__main__':
    data_prep = DataPrep()
    data_prep.run()