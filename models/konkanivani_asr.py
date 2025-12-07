"""
KonkaniVani Custom ASR Model
============================
Transformer-based ASR architecture with Conformer encoder and Transformer decoder.
Uses hybrid CTC + Attention loss for robust training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConformerBlock(nn.Module):
    """
    Conformer block: Feed-forward + Multi-head self-attention + Convolution + Feed-forward
    """
    def __init__(self, d_model=256, num_heads=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        
        # First feed-forward module
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Convolution module
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, conv_kernel_size, padding=conv_kernel_size//2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout)
        )
        
        # Second feed-forward module
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Feed-forward 1 (with residual)
        x = x + 0.5 * self.ff1(x)
        
        # Self-attention (with residual)
        attn_out, _ = self.self_attn(
            self.self_attn_norm(x), 
            self.self_attn_norm(x), 
            self.self_attn_norm(x),
            key_padding_mask=mask
        )
        x = x + self.attn_dropout(attn_out)
        
        # Convolution (with residual)
        conv_in = self.conv_norm(x).transpose(1, 2)  # (B, T, D) -> (B, D, T)
        conv_out = self.conv(conv_in).transpose(1, 2)  # (B, D, T) -> (B, T, D)
        x = x + conv_out
        
        # Feed-forward 2 (with residual)
        x = x + 0.5 * self.ff2(x)
        
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    """
    Conformer encoder for audio features
    """
    def __init__(self, input_dim=80, d_model=256, num_layers=12, num_heads=4, 
                 conv_kernel_size=31, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])
        
        self.d_model = d_model
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, time, features) - mel-spectrogram features
            lengths: (batch,) - actual lengths of sequences
        Returns:
            encoded: (batch, time, d_model)
            mask: (batch, time) - padding mask
        """
        # Create padding mask
        mask = None
        if lengths is not None:
            batch_size, max_len = x.size(0), x.size(1)
            mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through Conformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        return x, mask


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for generating Konkani text
    """
    def __init__(self, vocab_size, d_model=256, num_layers=6, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt: (batch, tgt_len) - target token indices
            memory: (batch, src_len, d_model) - encoder output
            tgt_mask: (tgt_len, tgt_len) - causal mask
            memory_mask: (batch, src_len) - encoder padding mask
        Returns:
            output: (batch, tgt_len, vocab_size)
        """
        # Embed and add positional encoding
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # Decode
        output = self.decoder(
            tgt_emb, 
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_mask
        )
        
        # Project to vocabulary
        return self.output_proj(output)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class KonkaniVaniASR(nn.Module):
    """
    Complete KonkaniVani ASR model with CTC + Attention hybrid training
    """
    def __init__(self, vocab_size, input_dim=80, d_model=256, 
                 encoder_layers=12, decoder_layers=6, num_heads=4,
                 conv_kernel_size=31, dropout=0.1):
        super().__init__()
        
        # Encoder (Conformer)
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=encoder_layers,
            num_heads=num_heads,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout
        )
        
        # CTC head for alignment-free training
        self.ctc_head = nn.Linear(d_model, vocab_size)
        
        # Decoder (Transformer)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=decoder_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.vocab_size = vocab_size
        self.d_model = d_model
    
    def forward(self, audio_features, audio_lengths=None, 
                target_tokens=None, target_lengths=None):
        """
        Forward pass for training
        
        Args:
            audio_features: (batch, time, features) - mel-spectrogram
            audio_lengths: (batch,) - actual audio lengths
            target_tokens: (batch, tgt_len) - target tokens (for attention training)
            target_lengths: (batch,) - target lengths
        
        Returns:
            ctc_logits: (batch, time, vocab_size) - for CTC loss
            attn_logits: (batch, tgt_len, vocab_size) - for attention loss (if target provided)
        """
        # Encode audio
        encoder_out, encoder_mask = self.encoder(audio_features, audio_lengths)
        
        # CTC head
        ctc_logits = self.ctc_head(encoder_out)
        
        # Attention decoder (only during training with targets)
        attn_logits = None
        if target_tokens is not None:
            # Create causal mask for decoder
            tgt_len = target_tokens.size(1)
            causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=target_tokens.device), 
                diagonal=1
            ).bool()
            
            # Decode
            attn_logits = self.decoder(
                target_tokens,
                encoder_out,
                tgt_mask=causal_mask,
                memory_mask=encoder_mask
            )
        
        return ctc_logits, attn_logits
    
    def recognize(self, audio_features, audio_lengths=None, beam_size=5):
        """
        Inference with beam search
        
        Args:
            audio_features: (batch, time, features)
            audio_lengths: (batch,)
            beam_size: beam width for decoding
        
        Returns:
            predictions: List of predicted token sequences
        """
        self.eval()
        with torch.no_grad():
            # Encode
            encoder_out, encoder_mask = self.encoder(audio_features, audio_lengths)
            
            # CTC greedy decoding (simple baseline)
            ctc_logits = self.ctc_head(encoder_out)
            ctc_preds = ctc_logits.argmax(dim=-1)
            
            # TODO: Implement beam search with attention decoder
            # For now, return CTC predictions
            return ctc_preds


def create_konkanivani_model(vocab_size, config=None):
    """
    Factory function to create KonkaniVani ASR model
    
    Args:
        vocab_size: Size of Konkani character vocabulary
        config: Optional config dict with model hyperparameters
    
    Returns:
        model: KonkaniVaniASR instance
    """
    if config is None:
        config = {
            'input_dim': 80,  # 80-dim mel-spectrogram
            'd_model': 256,
            'encoder_layers': 12,
            'decoder_layers': 6,
            'num_heads': 4,
            'conv_kernel_size': 31,
            'dropout': 0.1
        }
    
    model = KonkaniVaniASR(
        vocab_size=vocab_size,
        **config
    )
    
    return model
