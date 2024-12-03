import os
import yaml
import pandas as pd
import numpy as np
from misc.utils import log, load_yaml

def calc():
    pass

def update_config(df, loc, model_name='spy_30min_v1', feature_store=True):
    file_name = loc.split('/')[-1]
    
    base_class = 'feature_store' if feature_store else 'dependent_variable'
    name = file_name.split('.')[0]
    path = loc
    time_period = name.split('_')[-3]
    columns = list(df.columns)
    
    config_loc = f'config/{model_name}/features.yaml'
    
    if os.path.exists(config_loc):
        log(f'reading from {config_loc}')
        features_config = load_yaml(config_loc)
        if base_class in features_config.keys():
            if time_period in features_config[base_class].keys():
                features_config[base_class][time_period][name] = {'cols': columns, 'path': path}
            else:
                features_config[base_class][time_period] = {name: {'cols': columns, 'path': path}}
        else:
            features_config[base_class] = {time_period:{name: {'cols': columns, 'path': path}}}
    else:
        features_config = {base_class: {time_period:{name: {'cols': columns, 'path': path}}}}
    
    with open(config_loc, 'w+') as ff:
        yaml.dump(features_config, ff)
    log(f'updated {config_loc}')
    
def load_features_config(model_name='spy_30min_v1'):
    config_loc = f'config/{model_name}/features.yaml'
    if os.path.exists(config_loc):
        log(f'reading from {config_loc}')
        features_config = load_yaml(config_loc)
        return features_config
    else:
        return {'dependent_variable': None, 'feature_store': None}