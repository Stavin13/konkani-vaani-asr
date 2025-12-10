"""
Custom Konkani Emotion Detection Model
Built from scratch with BiLSTM + Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Attention mechanism for emotion detection"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
        
        Returns:
            context: (batch, hidden_dim)
            attention_weights: (batch, seq_len)
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len)
        
        # Apply attention weights
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output  # (batch, seq_len, hidden_dim)
        ).squeeze(1)  # (batch, hidden_dim)
        
        return context, attention_weights


class CustomEmotionModel(nn.Module):
    """
    Custom BiLSTM + Attention model for Konkani emotion detection
    Similar to NER model but for sequence classification
    """
    
    def __init__(
        self,
        vocab_size,
        num_emotions=7,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = AttentionLayer(lstm_output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_emotions)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        # Initialize embeddings
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[0].fill_(0)  # Padding token
        
        # Initialize LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize linear layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len) - token indices
            attention_mask: (batch, seq_len) - attention mask (1 for real tokens, 0 for padding)
        
        Returns:
            logits: (batch, num_emotions) - emotion logits
            attention_weights: (batch, seq_len) - attention weights
        """
        # Embedding
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Pack sequence if attention_mask is provided
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
        
        # BiLSTM
        lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # Unpack sequence
        if attention_mask is not None:
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_output, batch_first=True
            )
        
        # Layer normalization
        lstm_output = self.layer_norm(lstm_output)
        
        # Attention
        context, attention_weights = self.attention(lstm_output)
        context = self.dropout(context)
        
        # Classification layers
        hidden_output = F.relu(self.fc1(context))
        hidden_output = self.dropout(hidden_output)
        logits = self.fc2(hidden_output)
        
        return logits, attention_weights
    
    @torch.no_grad()
    def predict(self, input_ids, attention_mask=None):
        """
        Predict emotion for input text
        
        Args:
            input_ids: (batch, seq_len) - token indices
            attention_mask: (batch, seq_len) - attention mask
        
        Returns:
            predictions: (batch,) - predicted emotion indices
            probabilities: (batch, num_emotions) - emotion probabilities
            attention_weights: (batch, seq_len) - attention weights
        """
        self.eval()
        logits, attention_weights = self.forward(input_ids, attention_mask)
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        return predictions, probabilities, attention_weights


class EmotionLoss(nn.Module):
    """Custom loss for emotion detection with label smoothing"""
    
    def __init__(self, num_emotions, smoothing=0.1):
        super().__init__()
        self.num_emotions = num_emotions
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch, num_emotions)
            targets: (batch,) - emotion indices
        
        Returns:
            loss: scalar
        """
        log_probs = F.log_softmax(logits, dim=-1)
        
        # One-hot encode targets with label smoothing
        batch_size = targets.size(0)
        smooth_targets = torch.full(
            (batch_size, self.num_emotions),
            self.smoothing / (self.num_emotions - 1),
            device=targets.device
        )
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Calculate loss
        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        
        return loss


def create_custom_emotion_model(vocab_size, num_emotions=7, config=None):
    """
    Create custom emotion detection model
    
    Args:
        vocab_size: size of vocabulary
        num_emotions: number of emotion classes
        config: model configuration dict
    
    Returns:
        model: CustomEmotionModel
    """
    if config is None:
        config = {
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.3,
            'bidirectional': True
        }
    
    model = CustomEmotionModel(
        vocab_size=vocab_size,
        num_emotions=num_emotions,
        **config
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing Custom Emotion Model...")
    
    vocab_size = 5000
    num_emotions = 7
    
    model = create_custom_emotion_model(vocab_size, num_emotions)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 50
    
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    logits, attention_weights = model(input_ids, attention_mask)
    print(f"Logits shape: {logits.shape}")  # Should be (batch, num_emotions)
    print(f"Attention weights shape: {attention_weights.shape}")  # Should be (batch, seq_len)
    
    # Test prediction
    predictions, probabilities, attn = model.predict(input_ids, attention_mask)
    print(f"Predictions shape: {predictions.shape}")  # Should be (batch,)
    print(f"Probabilities shape: {probabilities.shape}")  # Should be (batch, num_emotions)
    
    # Test loss
    targets = torch.randint(0, num_emotions, (batch_size,))
    criterion = EmotionLoss(num_emotions)
    loss = criterion(logits, targets)
    print(f"Loss: {loss.item():.4f}")
    
    print("\n✅ Model test passed!")
