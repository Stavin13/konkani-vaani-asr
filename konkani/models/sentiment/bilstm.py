"""
BiLSTM model for Konkani sentiment analysis.
"""

import torch
import torch.nn as nn
from typing import Tuple


class CustomKonkaniSentimentModel(nn.Module):
    """
    Custom Neural Network for Konkani Sentiment Analysis
    Architecture: Embedding → BiLSTM → Attention → Dense → Output
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(CustomKonkaniSentimentModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Linear(lstm_output_dim, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def attention_layer(
        self,
        lstm_output: torch.Tensor,
        lengths: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention mechanism"""
        # Calculate attention weights
        attention_weights = torch.tanh(self.attention(lstm_output))
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended = torch.sum(attention_weights * lstm_output, dim=1)
        
        return attended, attention_weights
    
    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: list = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embedding
        embedded = self.embedding(input_ids)  # [batch, seq_len, embedding_dim]
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)  # [batch, seq_len, hidden_dim*2]
        
        # Attention
        attended, attention_weights = self.attention_layer(lstm_out, lengths)
        
        # Fully connected layers
        x = self.relu(self.fc1(attended))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits, attention_weights
