from abc import ABC
from misc.misc import BaseClass
from misc.misc import load_secrets, load_config, check_first_run

class AlpacaDownload(BaseClass):
    def __init__(self):
        super().__init__()
        self.name = 'AlpacaDownload'
        self.config = load_config()
        self.secrets = load_secrets()
        self.first_run = check_first_run('alpaca_download')

if __name__=='__main__':
    alpaca_download = AlpacaDownload()
    print(alpaca_download.secrets)