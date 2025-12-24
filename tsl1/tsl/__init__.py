from ._logger import logger
from .config import Config
from .lazy_loader import LazyLoader
import sys
sys.path.append("/home/data/zsy/tsl1/tsl1")

data = LazyLoader('data', globals(), 'tsl1.data')
datasets = LazyLoader('datasets', globals(), 'tsl1.datasets')
nn = LazyLoader('nn', globals(), 'tsl1.nn')
engines = LazyLoader('engines', globals(), 'tsl1.engines')

__version__ = '0.9.3'

epsilon = 5e-8
config = Config()

__all__ = [
    '__version__',
    'config',
    'epsilon',
    'logger',
    'data',
    'datasets',
    'nn',
    'engines',
]
