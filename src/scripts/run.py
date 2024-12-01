from src.scripts.data_ingestion import alpaca_refresh
from src.scripts.data_ingestion import data_prep
from misc.utils import load_config

def run():
    config = load_config()
    alpaca_refresh.AlpacaRefresh(config).run()
    data_prep.DataPrep(config).run()
    
if __name__=="__main__":
    run()