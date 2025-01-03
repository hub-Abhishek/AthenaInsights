import os
import re
import sys
import yaml
import datetime

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

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

    import os
import re

def find_newest_version_folder(base_path):
    pattern = r"spy_30min_v(\d+)$"
    folders = os.listdir(base_path)
    max_version = -1
    newest_folder = None
    for folder in folders:
        match = re.match(pattern, folder)
        if match:
            version = int(match.group(1))
            if version > max_version:
                max_version = version
                newest_folder = folder
    return newest_folder


def load_config():
    config = {}
    newest_folder = find_newest_version_folder('config')

    path_yaml = f'config/{newest_folder}/paths.yaml'
    config['paths_config'] = load_yaml(path_yaml)

    technical_yaml = f'config/{newest_folder}/technical.yaml'
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
        # log(f'reading from {path}')
        df = read_df(path)
        df.to_parquet(path.replace('.parquet', '_backup.parquet'))
        return df

def read_df(path, columns=None):
    if path.endswith('.parquet'):
        log(f'reading from {path}')
        # df = pd.read_parquet(path)
        df = pq.read_pandas(path, columns=columns).to_pandas()
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

def extract_properties_from_name(name):
    if len(name.split('_'))>=5:
        features_type = name.split('_')[-1]
        base_or_diff = name.split('_')[-2]
        duration = name.split('_')[-3]
        duration, duration_unit = re.match(r'^(\d+)([a-zA-Z]+)$', duration).groups()
        duration = int(duration)
        return name, features_type, base_or_diff, duration, duration_unit
    else:
        log("name doesn't follow convention")

def get_name_and_type(path):
    file_name = path.split('/')[-1]
    
    df_type = file_name.split('/')[-1].split('.')[1]
    name = file_name.split('.')[0]
    if len(name.split('_'))>=5:
        name, features_type, base_or_diff, duration, duration_unit = extract_properties_from_name(name)
        return name, df_type, features_type, base_or_diff, duration, duration_unit
    else:
        features_type = None
        base_or_diff = None
        duration = name.split('_')[-1]
        log(name)
        log(duration)
        duration, duration_unit = re.match(r'^(\d+)([a-zA-Z]+)$', duration).groups()
        duration = int(duration)
        return name, df_type, features_type, base_or_diff, duration, duration_unit
        