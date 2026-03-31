import os
import random
import warnings

from dotenv import load_dotenv
from icecream import ic


def init_settings(verbose: bool=False):
    '''
    自前環境の初期設定をおこなう関数
    
    Args:
        verbose (bool): ログ出力の有無
    '''
    load_dotenv()
    
    if verbose:
        ic.enable()
        ic()
    else:
        ic.disable()
        warnings.filterwarnings('ignore')


def set_seeds(seed: int=42):
    '''
    乱数シードを設定する関数
    
    Args:
        seed (int): 乱数シードの値
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
