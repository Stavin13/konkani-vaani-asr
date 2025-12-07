"""
Custom NER Model for Konkani
BiLSTM-CRF architecture for Named Entity Recognition
"""

import torch
import torch.nn as nn
from torchcrf import CRF


class KonkaniNER(nn.Module):
    """
    BiLSTM-CRF model for Konkani NER
    
    Architecture:
    - Character-level embeddings
    - Word embeddings
    - BiLSTM encoder
    - CRF decoder
    """
    
    def __init__(self, vocab_size, char_vocab_size, num_tags,
                 embedding_dim=128, char_embedding_dim=32,
                 hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        
        # Word embeddings
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Character embeddings + CNN
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        self.char_cnn = nn.Conv1d(char_embedding_dim, 50, kernel_size=3, padding=1)
        
        # BiLSTM
        lstm_input_dim = embedding_dim + 50  # word + char features
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer to tag space
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        
        # CRF layer
        self.crf = CRF(num_tags, batch_first=True)
    
    def _get_char_features(self, char_ids):
        """
        Extract character-level features using CNN
        
        Args:
            char_ids: (batch, seq_len, max_char_len)
        Returns:
            char_features: (batch, seq_len, char_feature_dim)
        """
        batch_size, seq_len, max_char_len = char_ids.size()
        
        # Reshape for embedding
        char_ids_flat = char_ids.view(batch_size * seq_len, max_char_len)
        
        # Embed characters: (batch*seq_len, max_char_len, char_emb_dim)
        char_embeds = self.char_embedding(char_ids_flat)
        
        # Transpose for CNN: (batch*seq_len, char_emb_dim, max_char_len)
        char_embeds = char_embeds.transpose(1, 2)
        
        # Apply CNN: (batch*seq_len, 50, max_char_len)
        char_cnn_out = self.char_cnn(char_embeds)
        
        # Max pooling: (batch*seq_len, 50)
        char_features = torch.max(char_cnn_out, dim=2)[0]
        
        # Reshape back: (batch, seq_len, 50)
        char_features = char_features.view(batch_size, seq_len, -1)
        
        return char_features
    
    def forward(self, word_ids, char_ids, tags=None, mask=None):
        """
        Forward pass
        
        Args:
            word_ids: (batch, seq_len)
            char_ids: (batch, seq_len, max_char_len)
            tags: (batch, seq_len) - optional, for training
            mask: (batch, seq_len) - padding mask
        
        Returns:
            If training (tags provided): loss
            If inference: predicted tags
        """
        # Word embeddings
        word_embeds = self.word_embedding(word_ids)
        word_embeds = self.dropout(word_embeds)
        
        # Character features
        char_features = self._get_char_features(char_ids)
        char_features = self.dropout(char_features)
        
        # Concatenate word and char features
        combined = torch.cat([word_embeds, char_features], dim=2)
        
        # BiLSTM
        lstm_out, _ = self.lstm(combined)
        lstm_out = self.dropout(lstm_out)
        
        # Project to tag space
        emissions = self.hidden2tag(lstm_out)
        
        if tags is not None:
            # Training: compute CRF loss
            # CRF expects mask as ByteTensor
            if mask is None:
                mask = torch.ones_like(word_ids, dtype=torch.uint8)
            
            # CRF loss (negative log likelihood)
            loss = -self.crf(emissions, tags, mask=mask.byte(), reduction='mean')
            return loss
        else:
            # Inference: decode best path
            if mask is None:
                mask = torch.ones_like(word_ids, dtype=torch.uint8)
            
            predictions = self.crf.decode(emissions, mask=mask.byte())
            return predictions


class SimpleNER(nn.Module):
    """
    Simpler NER model without CRF (faster training)
    Use this if CRF installation fails
    """
    
    def __init__(self, vocab_size, num_tags, embedding_dim=128,
                 hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_tags)
    
    def forward(self, word_ids, tags=None, mask=None):
        """Simple forward pass"""
        embeds = self.embedding(word_ids)
        embeds = self.dropout(embeds)
        
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        
        logits = self.fc(lstm_out)
        
        if tags is not None:
            # Training: compute cross-entropy loss
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tags.view(-1))
            return loss
        else:
            # Inference: argmax
            predictions = torch.argmax(logits, dim=-1)
            return predictions


def create_ner_model(vocab_size, num_tags, use_crf=True, **kwargs):
    """
    Factory function to create NER model
    
    Args:
        vocab_size: Size of word vocabulary
        num_tags: Number of NER tags (usually 9: B/I for PER/ORG/LOC/MISC + O)
        use_crf: Whether to use CRF layer (better accuracy but slower)
        **kwargs: Additional model parameters
    
    Returns:
        NER model instance
    """
    if use_crf:
        try:
            # Try to create CRF model
            char_vocab_size = kwargs.get('char_vocab_size', 200)
            model = KonkaniNER(
                vocab_size=vocab_size,
                char_vocab_size=char_vocab_size,
                num_tags=num_tags,
                **{k: v for k, v in kwargs.items() if k != 'char_vocab_size'}
            )
            print("✅ Created BiLSTM-CRF model")
            return model
        except ImportError:
            print("⚠️  pytorch-crf not installed, using simple model")
            use_crf = False
    
    if not use_crf:
        model = SimpleNER(
            vocab_size=vocab_size,
            num_tags=num_tags,
            **kwargs
        )
        print("✅ Created BiLSTM model (without CRF)")
        return model
