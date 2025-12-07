"""
Training configuration for Konkani NLP.
"""

from dataclasses import dataclass
import torch


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    warmup_steps: int = 0
    
    # Early stopping
    patience: int = 3
    min_delta: float = 0.001
    
    # Optimization
    optimizer: str = "adam"  # adam, adamw, sgd
    scheduler: str = "none"  # none, linear, cosine, plateau
    gradient_clip: float = 1.0
    
    # Device
    device: str = "auto"  # auto, cpu, cuda, mps
    
    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 100
    save_best_only: bool = True
    save_total_limit: int = 3
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Validate and process configuration."""
        # Validate values
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.patience > 0, "patience must be positive"
        
        # Auto-detect device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
    
    def get_device(self) -> torch.device:
        """Get torch device object."""
        return torch.device(self.device)
