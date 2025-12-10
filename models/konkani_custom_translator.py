"""
Custom Konkani-English Translation Model
Built from scratch like the ASR model
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CustomTranslationModel(nn.Module):
    """
    Custom Seq2Seq Transformer for Konkani→English Translation
    Similar architecture to the ASR model
    """
    
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=512
    ):
        super().__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        """
        Args:
            src: (batch, src_len) - source tokens
            tgt: (batch, tgt_len) - target tokens
            src_mask: (src_len, src_len) - source attention mask
            tgt_mask: (tgt_len, tgt_len) - target attention mask
            src_padding_mask: (batch, src_len) - source padding mask
            tgt_padding_mask: (batch, tgt_len) - target padding mask
        
        Returns:
            output: (batch, tgt_len, tgt_vocab_size)
        """
        # Embed and add positional encoding
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # Transformer
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        # Project to vocabulary
        output = self.output_projection(output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """Generate mask for decoder (prevents looking ahead)"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def encode(self, src, src_mask=None, src_padding_mask=None):
        """Encode source sequence"""
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_padding_mask
        )
        return memory
    
    def decode(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        """Decode target sequence"""
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        output = self.transformer.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask
        )
        return self.output_projection(output)
    
    @torch.no_grad()
    def translate(self, src, src_padding_mask=None, max_len=100, sos_idx=1, eos_idx=2):
        """
        Greedy decoding for translation
        
        Args:
            src: (batch, src_len) - source tokens
            src_padding_mask: (batch, src_len) - padding mask
            max_len: maximum length of translation
            sos_idx: start of sequence token index
            eos_idx: end of sequence token index
        
        Returns:
            translations: (batch, tgt_len) - translated tokens
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        memory = self.encode(src, src_padding_mask=src_padding_mask)
        
        # Initialize target with SOS token
        tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        
        # Generate tokens one by one
        for _ in range(max_len - 1):
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
            output = self.decode(tgt, memory, tgt_mask=tgt_mask, memory_padding_mask=src_padding_mask)
            
            # Get next token (greedy)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if all sequences have EOS
            if (next_token == eos_idx).all():
                break
        
        return tgt


def create_custom_translation_model(src_vocab_size, tgt_vocab_size, config=None):
    """
    Create custom translation model
    
    Args:
        src_vocab_size: size of source vocabulary (Konkani)
        tgt_vocab_size: size of target vocabulary (English)
        config: model configuration dict
    
    Returns:
        model: CustomTranslationModel
    """
    if config is None:
        config = {
            'd_model': 256,
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'max_len': 512
        }
    
    model = CustomTranslationModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        **config
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing Custom Translation Model...")
    
    src_vocab_size = 5000  # Konkani vocabulary
    tgt_vocab_size = 10000  # English vocabulary
    
    model = create_custom_translation_model(src_vocab_size, tgt_vocab_size)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 4
    src_len = 20
    tgt_len = 25
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    # Create masks
    tgt_mask = model.generate_square_subsequent_mask(tgt_len)
    
    output = model(src, tgt, tgt_mask=tgt_mask)
    print(f"Output shape: {output.shape}")  # Should be (batch, tgt_len, tgt_vocab_size)
    
    # Test translation
    translations = model.translate(src, max_len=30)
    print(f"Translation shape: {translations.shape}")
    
    print("\n✅ Model test passed!")
