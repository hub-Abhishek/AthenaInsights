from misc.utils import BaseClass, get_alpaca_secrets, load_config, log, read_and_duplicate, read_df, save_df_as_parquet, save_df_as_csv, get_all_paths_from_loc, get_name_and_type
from misc.features_calc import update_config, load_features_config, get_paths_and_cols_from_config, check_index_is_monotonic_increasing

class Model(BaseClass):
    def __init__(self, config=None, name='Model'):
        super().__init__()
        self.name = name
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
        self.calc_metrics = self.feature_prep['calc_metrics']
        
        self.features_listing_config = load_features_config()