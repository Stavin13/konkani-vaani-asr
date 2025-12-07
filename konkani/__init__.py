"""
Konkani NLP - Core Package
Main package for Konkani language processing.
"""

__version__ = "0.1.0"
__author__ = "Stavin Fernandes"

from . import core
from . import models
from . import training
from . import inference
from . import data
from . import utils

__all__ = [
    'core',
    'models',
    'training',
    'inference',
    'data',
    'utils',
]
