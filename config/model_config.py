"""
Model configuration classes for Konkani NLP.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BiLSTMConfig:
    """Configuration for BiLSTM sentiment model."""
    
    vocab_size: int = 10000
    embedding_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 2
    num_classes: int = 3
    dropout: float = 0.3
    bidirectional: bool = True
    max_length: int = 128
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.embedding_dim > 0, "embedding_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.num_classes > 0, "num_classes must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"


@dataclass
class TransformerConfig:
    """Configuration for transformer-based sentiment model."""
    
    model_name: str = "distilbert-base-multilingual-cased"
    num_classes: int = 3
    max_length: int = 128
    dropout: float = 0.1
    freeze_base: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.num_classes > 0, "num_classes must be positive"
        assert self.max_length > 0, "max_length must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"


@dataclass
class ASRConfig:
    """Configuration for ASR model."""
    
    # Audio processing
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    
    # Model architecture
    encoder_dim: int = 512
    decoder_dim: int = 512
    num_encoder_layers: int = 12
    num_decoder_layers: int = 6
    num_attention_heads: int = 8
    dropout: float = 0.1
    
    # Vocabulary
    vocab_size: int = 5000
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.sample_rate > 0, "sample_rate must be positive"
        assert self.n_mels > 0, "n_mels must be positive"
        assert self.encoder_dim > 0, "encoder_dim must be positive"
        assert self.decoder_dim > 0, "decoder_dim must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
