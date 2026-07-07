import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

class PositionalEncoding(nn.Module):
    """Standard positional encoding for Transformers"""
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

class ConformerBlock(nn.Module):
    """
    Conformer block: Feed-forward + Multi-head self-attention + Convolution + Feed-forward
    """
    def __init__(self, d_model=256, num_heads=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        
        # First half-step feed-forward module
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
        
        # Second half-step feed-forward module
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
        # mask is key_padding_mask (True for padding)
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
    def __init__(self, input_dim=80, d_model=256, num_layers=12, num_heads=4, 
                 conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])
        self.gradient_checkpointing = False
    
    def forward(self, x, lengths=None):
        # Create padding mask (True for padding indices)
        mask = None
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)
            
        return x, mask

class ConformerCTC(nn.Module):
    """
    Pure Conformer-CTC model for ASR.
    High performance, memory efficient, no decoder required.
    """
    def __init__(self, vocab_size, input_dim=80, d_model=256, 
                 num_layers=12, num_heads=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout
        )
        self.ctc_head = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        
        # For gradient checkpointing support if needed
        self.gradient_checkpointing = False

    def forward(self, x, lengths=None):
        """
        x: (Batch, Time, Features) - Mel-spectrogram
        lengths: (Batch,) - Actual lengths of sequences
        """
        enc_out, mask = self.encoder(x, lengths)
        logits = self.ctc_head(enc_out)
        return logits, lengths # CTC models usually don't change the time dimension length significantly

    @torch.no_grad()
    def decode(self, x, lengths=None):
        """Greedy CTC Decoding"""
        logits, _ = self.forward(x, lengths)
        probs = F.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1)

def create_model(vocab_size, d_model=256, num_layers=12):
    config = {
        'vocab_size': vocab_size,
        'input_dim': 80,
        'd_model': d_model,
        'num_layers': num_layers,
        'num_heads': 4,
        'conv_kernel_size': 31,
        'dropout': 0.1
    }
    return ConformerCTC(**config)

if __name__ == "__main__":
    # Quick test
    model = create_model(vocab_size=81, d_model=128, num_layers=4)
    test_input = torch.randn(2, 100, 80)
    test_lengths = torch.LongTensor([100, 80])
    out, out_lens = model(test_input, test_lengths)
    print(f"Output shape: {out.shape}") # Should be (2, 100, 81)
    print(f"Success!")
