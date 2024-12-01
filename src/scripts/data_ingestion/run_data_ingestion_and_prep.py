import misc
from misc import utils
from misc.utils import load_config
from alpaca_refresh import AlpacaRefresh
from data_prep import DataPrep

def run():
    config = load_config()
    AlpacaRefresh(config).run()
    DataPrep(config).run()
    
if __name__=='__main__':
    run()