"""
Centralized path management for Konkani NLP project.
"""

from pathlib import Path
from typing import Optional


class Paths:
    """Centralized path configuration for the project."""
    
    # Base directories
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    SCRIPTS_DIR = PROJECT_ROOT / "scripts"
    DOCS_DIR = PROJECT_ROOT / "docs"
    
    # Data subdirectories
    DATA_RAW = DATA_DIR / "raw"
    DATA_PROCESSED = DATA_DIR / "processed"
    DATA_CACHE = DATA_DIR / "cache"
    
    # Raw data
    DATA_RAW_SENTIMENT = DATA_RAW / "sentiment"
    DATA_RAW_ASR = DATA_RAW / "asr"
    
    # Processed data
    DATA_PROCESSED_SENTIMENT = DATA_PROCESSED / "sentiment"
    DATA_PROCESSED_ASR = DATA_PROCESSED / "asr"
    
    # Model directories
    MODELS_SENTIMENT = MODELS_DIR / "sentiment"
    MODELS_ASR = MODELS_DIR / "asr"
    
    # Output directories
    OUTPUTS_LOGS = OUTPUTS_DIR / "logs"
    OUTPUTS_CHECKPOINTS = OUTPUTS_DIR / "checkpoints"
    OUTPUTS_REPORTS = OUTPUTS_DIR / "reports"
    OUTPUTS_VISUALIZATIONS = OUTPUTS_DIR / "visualizations"
    
    @classmethod
    def ensure_dirs(cls):
        """Create all necessary directories if they don't exist."""
        dirs = [
            cls.DATA_RAW_SENTIMENT,
            cls.DATA_RAW_ASR,
            cls.DATA_PROCESSED_SENTIMENT,
            cls.DATA_PROCESSED_ASR,
            cls.DATA_CACHE,
            cls.MODELS_SENTIMENT,
            cls.MODELS_ASR,
            cls.OUTPUTS_LOGS,
            cls.OUTPUTS_CHECKPOINTS,
            cls.OUTPUTS_REPORTS,
            cls.OUTPUTS_VISUALIZATIONS,
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_sentiment_dataset(cls, filename: str) -> Path:
        """Get path to sentiment dataset file."""
        return cls.DATA_PROCESSED_SENTIMENT / filename
    
    @classmethod
    def get_asr_dataset(cls, filename: str) -> Path:
        """Get path to ASR dataset file."""
        return cls.DATA_PROCESSED_ASR / filename
    
    @classmethod
    def get_model_path(cls, model_type: str, model_name: str) -> Path:
        """Get path to model directory."""
        if model_type == "sentiment":
            return cls.MODELS_SENTIMENT / model_name
        elif model_type == "asr":
            return cls.MODELS_ASR / model_name
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @classmethod
    def get_checkpoint_path(cls, experiment_name: str) -> Path:
        """Get path to checkpoint directory for an experiment."""
        checkpoint_dir = cls.OUTPUTS_CHECKPOINTS / experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir
    
    @classmethod
    def get_log_path(cls, experiment_name: str) -> Path:
        """Get path to log file for an experiment."""
        cls.OUTPUTS_LOGS.mkdir(parents=True, exist_ok=True)
        return cls.OUTPUTS_LOGS / f"{experiment_name}.log"
