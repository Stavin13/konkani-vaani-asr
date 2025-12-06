"""
Configuration package for Konkani NLP.
Centralized configuration management for paths, models, and training.
"""

from .paths import Paths
from .model_config import BiLSTMConfig, TransformerConfig, ASRConfig
from .training_config import TrainingConfig

__all__ = [
    'Paths',
    'BiLSTMConfig',
    'TransformerConfig',
    'ASRConfig',
    'TrainingConfig',
]
