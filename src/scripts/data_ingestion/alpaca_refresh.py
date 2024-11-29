import misc
from misc import utils
from misc.utils import BaseClass, get_alpaca_secrets, load_config, log, read_and_duplicate, save_df_as_parquet

import re
import pandas as pd
import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import datetime


class AlpacaRefresh(BaseClass):
    def __init__(self):
        super().__init__()
        self.name = 'AlpacaDownload'
        log(f'running {self.name}')
        self.config = load_config()
        self.alpaca_data_ingestion_technical_config = self.config['technical_yaml']['data_ingestion']['alpaca_download']
        self.paths_config = self.config['paths_config']
        self.alpaca_key, self.alpaca_secret = get_alpaca_secrets()

        self.bucket_loc = f's3://{self.paths_config["s3_bucket"]}'
        self.base_folder = self.paths_config["base_folder"]
        self.data_folder = self.paths_config["data_folder"]


    # def get_alpaca_minute_level_data(start_date, end_date, alpaca_key, alpaca_secret, symbols):
    @staticmethod
    def get_alpaca_minute_level_data(start_date: datetime.datetime, end_date: datetime.datetime, alpaca_key: str, alpaca_secret: str, symbols: str):
        stock_historical_data_client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=start_date,
            end=end_date, 
        )
        df = stock_historical_data_client.get_stock_bars(req).df
        df = df.reset_index()
        return df

    @staticmethod
    def get_alpaca_hour_level_data(start_date: datetime.datetime, end_date: datetime.datetime, alpaca_key: str, alpaca_secret: str, symbols: str):
        stock_historical_data_client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Hour,
            start=start_date,
            end=end_date,
        )
        df = stock_historical_data_client.get_stock_bars(req).df
        df = df.reset_index()
        return df

    @staticmethod
    def get_alpaca_day_level_data(start_date: datetime.datetime, end_date: datetime.datetime, alpaca_key: str, alpaca_secret: str, symbols: str):
        stock_historical_data_client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
        )
        df = stock_historical_data_client.get_stock_bars(req).df
        df = df.reset_index()
        return df

    def extract(self):

        self.refresh = self.alpaca_data_ingestion_technical_config['refresh']
        self.start_date = self.alpaca_data_ingestion_technical_config['start_date']
        self.end_date = self.alpaca_data_ingestion_technical_config['end_date']
        self.window = self.alpaca_data_ingestion_technical_config['window']

        if self.refresh:
            if 'week' in self.window:
                self.end_date = datetime.datetime.now()
                self.end_date = datetime.datetime(self.end_date.year, self.end_date.month, self.end_date.day)
                self.start_date = self.end_date - datetime.timedelta(weeks=int(re.search(r'\d+', self.window).group()))
            else:
                raise NotImplementedError('Not yet implemented for refreshing other than N weeks')
        else:
            raise NotImplementedError('Not yet implemented for getting all data')

        log(f'start date: {self.start_date}, end date: {self.end_date}, window: {self.window}')

        alpaca_new_minute_level_data = self.get_alpaca_minute_level_data(
            self.start_date, self.end_date, self.alpaca_key, self.alpaca_secret,
            self.alpaca_data_ingestion_technical_config['symbols'])
        log(f'alpaca_new_minute_level_data shape: {alpaca_new_minute_level_data.shape}')

        alpaca_new_hour_level_data = self.get_alpaca_hour_level_data(
            self.start_date, self.end_date, self.alpaca_key, self.alpaca_secret,
            self.alpaca_data_ingestion_technical_config['symbols'])
        log(f'alpaca_new_hour_level_data shape: {alpaca_new_hour_level_data.shape}')

        alpaca_new_day_level_data = self.get_alpaca_day_level_data(
            self.start_date, self.end_date, self.alpaca_key, self.alpaca_secret,
            self.alpaca_data_ingestion_technical_config['symbols'])
        log(f'alpaca_new_day_level_data shape: {alpaca_new_day_level_data.shape}')

        alpaca_current_min_level_data = read_and_duplicate(f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.paths_config["parquet_folder"]}/stock_bars_minute.parquet')
        log(f'alpaca_current_min_level_data shape: {alpaca_current_min_level_data.shape}')
        
        alpaca_current_hour_level_data = read_and_duplicate(f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.paths_config["parquet_folder"]}/stock_bars_hour.parquet')
        log(f'alpaca_current_hour_level_data shape: {alpaca_current_hour_level_data.shape}')
        
        alpaca_current_day_level_data = read_and_duplicate(f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.paths_config["parquet_folder"]}/stock_bars_day.parquet')
        log(f'alpaca_current_day_level_data shape: {alpaca_current_day_level_data.shape}')

        return {
            "alpaca_new_minute_level_data": alpaca_new_minute_level_data,
            "alpaca_new_hour_level_data": alpaca_new_hour_level_data,
            "alpaca_new_day_level_data": alpaca_new_day_level_data,
            "alpaca_current_min_level_data": alpaca_current_min_level_data,
            "alpaca_current_hour_level_data": alpaca_current_hour_level_data,
            "alpaca_current_day_level_data": alpaca_current_day_level_data
        }

    def transform(self, dfs):
        alpaca_current_min_level_data = dfs["alpaca_current_min_level_data"]
        alpaca_current_hour_level_data = dfs["alpaca_current_hour_level_data"]
        alpaca_current_day_level_data = dfs["alpaca_current_day_level_data"]

        alpaca_current_min_level_data = alpaca_current_min_level_data[alpaca_current_min_level_data.timestamp.dt.tz_convert(None) < pd.to_datetime(self.start_date)]
        log(f'alpaca_current_min_level_data shape after filtering: {alpaca_current_min_level_data.shape}')

        alpaca_current_hour_level_data = alpaca_current_hour_level_data[alpaca_current_hour_level_data.timestamp.dt.tz_convert(None) < pd.to_datetime(self.start_date)]
        log(f'alpaca_current_hour_level_data shape after filtering: {alpaca_current_hour_level_data.shape}')

        alpaca_current_day_level_data = alpaca_current_day_level_data[alpaca_current_day_level_data.timestamp.dt.tz_convert(None) < pd.to_datetime(self.start_date)]
        log(f'alpaca_current_day_level_data shape after filtering: {alpaca_current_day_level_data.shape}')

        min_level_data = pd.concat([alpaca_current_min_level_data, dfs["alpaca_new_minute_level_data"]], axis=0)
        log(f'min_level_data shape: {min_level_data.shape}')
        hour_level_data = pd.concat([alpaca_current_hour_level_data, dfs["alpaca_new_hour_level_data"]], axis=0)
        log(f'hour_level_data shape: {hour_level_data.shape}')
        day_level_data = pd.concat([alpaca_current_day_level_data, dfs["alpaca_new_day_level_data"]], axis=0)
        log(f'day_level_data shape: {day_level_data.shape}')

        del dfs, alpaca_current_min_level_data, alpaca_current_hour_level_data, alpaca_current_day_level_data

        return {
            "min_level_data": min_level_data,
            "hour_level_data": hour_level_data,
            "day_level_data": day_level_data
        }

    def save(self, dfs):
        alpaca_minute_level_data = dfs["min_level_data"]
        alpaca_hour_level_data = dfs["hour_level_data"]
        alpaca_day_level_data = dfs["day_level_data"]

        save_df_as_parquet(alpaca_minute_level_data, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.paths_config["parquet_folder"]}/stock_bars_minute.parquet', index=False)
        save_df_as_parquet(alpaca_hour_level_data, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.paths_config["parquet_folder"]}/stock_bars_hour.parquet', index=False)
        save_df_as_parquet(alpaca_day_level_data, f'{self.bucket_loc}/{self.base_folder}/{self.data_folder}/{self.paths_config["parquet_folder"]}/stock_bars_day.parquet', index=False)


if __name__=='__main__':
    alpaca_download = AlpacaRefresh()
    alpaca_download.run()