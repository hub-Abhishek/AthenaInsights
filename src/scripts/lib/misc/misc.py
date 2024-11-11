import os
import sys
from abc import ABC
import yaml

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
        self.extract()
        self.transform()
        self.save()

def load_secrets():
    path_secrets_yaml = 'config/secrets.yaml'
    with open(path_secrets_yaml, 'r') as file:
        secrets = yaml.load(file)
    return secrets

def load_config():
    config = {}
    path_yaml = 'config/paths.yaml'
    with open(path_yaml, 'r') as file:
        paths = yaml.load(file)
    config['paths'] = paths
    return config

def check_first_run(option_name, secrets=None, config=None):
    if option_name == 'alpaca_download':
        config['paths']