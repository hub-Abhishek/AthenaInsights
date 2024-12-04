from technical_feature_prep import TechnicalFeaturePrep
from model_custom_features import ModelCustomFeaturePrep
from model_dependent_variable import ModelDependentFeaturePrep
from model_final_training_files import ModelTrainingFilePrep
from misc.utils import load_config

def run():
    config = load_config()
    TechnicalFeaturePrep(config).run()
    ModelCustomFeaturePrep(config).run()    
    ModelDependentFeaturePrep(config).run()    
    ModelTrainingFilePrep(config).run()

if __name__ == '__main__':
    run()