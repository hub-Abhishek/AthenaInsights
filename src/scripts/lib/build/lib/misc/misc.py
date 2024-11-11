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
    with open(path_secrets_yaml, 'rb') as file:
        secrets = yaml.load(file)
    return secrets

def load_config():
    pass
