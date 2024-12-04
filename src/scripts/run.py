import misc
from misc import utils
from misc.utils import load_config
from data_ingestion.alpaca_refresh import AlpacaRefresh
from data_ingestion.data_prep import DataPrep
from feature_prep.technical_feature_prep import TechnicalFeaturePrep
from feature_prep.model_custom_features_prep import ModelCustomFeaturePrep
from feature_prep.model_dependent_variable import ModelDependentFeaturePrep
from feature_prep.model_final_training_files import ModelTrainingFilePrep

def run():
    config = load_config()
    AlpacaRefresh(config).run()
    DataPrep(config).run()
    TechnicalFeaturePrep(config).run()
    ModelCustomFeaturePrep(config).run()    
    ModelDependentFeaturePrep(config).run()    
    ModelTrainingFilePrep(config).run()

if __name__=='__main__':
    run()