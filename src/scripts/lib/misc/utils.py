import os
import sys
import yaml
import datetime

import pandas as pd
import numpy as np

from abc import ABC

class BaseClass(ABC):
    def __init__(self):
        super().__init__()
        self.name = 'BaseClass'

    def extract(self):
        raise NotImplementedError()

    def transform(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def run(self):
        dfs = self.extract()
        dfs = self.transform(dfs)
        self.save(dfs)

# def load_secrets():
#     path_secrets_yaml = 'config/secrets.yaml'
#     with open(path_secrets_yaml, 'r') as file:
#         secrets = yaml.load(file)
#     return secrets

def load_yaml(path):
    with open(path, 'r') as file:
        file = yaml.safe_load(file)
    return file

def load_config():
    config = {}

    path_yaml = 'config/spy_30min_v1/paths.yaml'
    config['paths_config'] = load_yaml(path_yaml)

    technical_yaml = 'config/spy_30min_v1/technical.yaml'
    config['technical_yaml'] = load_yaml(technical_yaml)

    return config

def get_alpaca_secrets():
    import boto3
    import json
    from botocore.exceptions import ClientError

    secret_name = "AthenaInsightsAlpaca"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    secret = get_secret_value_response['SecretString']

    alpaca_key = json.loads(secret)['alpaca_key']
    alpaca_secret = json.loads(secret)['alpaca_secret']

    return alpaca_key, alpaca_secret

def log(s):
    print(s)

def read_and_duplicate(path):
    if path.endswith('.parquet'):
        log(f'reading from {path}')
        df = pd.read_parquet(path)
        df.to_parquet(path.replace('.parquet', '_backup.parquet'))
        return df

def read_df(path):
    if path.endswith('.parquet'):
        log(f'reading from {path}')
        df = pd.read_parquet(path)
        return df

    if path.endswith('.csv'):
        log(f'reading from {path}')
        df = pd.read_csv(path)
        return df

def save_df_as_parquet(df, path, index=True):
    try:
        log(f"writing to file {path}")
        df.to_parquet(path, index=index)
    except:
        log(f"ERROR WITH {path}! NOT ABLE TO SAVE FILE! REPLACING WITH BACKUP")
        df = pd.read_parquet(path.replace('.parquet', '_backup.parquet'),)
        df.to_parquet(path.replace('_backup.parquet', '.parquet'), index=index)

def save_df_as_csv(df, path, index=True):
    try:
        log(f"writing to file {path}")
        df.to_csv(path, index=index)
    except:
        log(f"ERROR WITH {path}! NOT ABLE TO SAVE FILE! REPLACING WITH BACKUP")
        df = pd.read_csv(path.replace('.csv', '_backup.csv'),)
        df.to_parquet(path.replace('_backup.csv', '.csv'), index=index)

def explore_loc(client, bucket, loc):
    return client.list_objects_v2(
            Bucket=bucket,
            Prefix=loc)
    
def get_all_paths_from_loc(client, bucket, loc, filters=None):
    paths = []
    response = explore_loc(client, bucket, loc)
    for content in response.get('Contents', []):
        paths.append(f"s3://{bucket}/{content['Key']}")
    if filters:
        paths = [x for x in paths if filters in x]
    return paths

def get_name_and_type(path):
    name = path.split('/')[-1].split('.')[0]
    df_type = path.split('/')[-1].split('.')[1]
    return name, df_type