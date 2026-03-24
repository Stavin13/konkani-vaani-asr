"""
Conformer-CTC v2
Upgrades over v1:
  - Subword/BPE tokenizer support (SentencePiece)
  - Beam search decoding (pyctcdecode / ctcdecode)
  - SpecAugment (frequency + time masking)
  - INT8 quantization export
  - Gradient checkpointing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math


# ─────────────────────────────────────────────────────────────
# POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Adding pe (FP32) to x (potentially FP16/BF16) results in FP32,
        # which keeps the model's residual 'backbone' in float32 for stability.
        return self.dropout((x + self.pe[:, :x.size(1)]).float())


# ─────────────────────────────────────────────────────────────
# SPEC AUGMENT
# ─────────────────────────────────────────────────────────────
class SpecAugment(nn.Module):
    """
    SpecAugment: frequency masking + time masking applied on mel features.
    Applied only during training (model.train()).
    """
    def __init__(self, freq_mask_param=27, time_mask_param=100, num_freq_masks=2, num_time_masks=2):
        super().__init__()
        self.freq_masks = nn.ModuleList([
            torch.nn.Identity()  # placeholder, applied manually
            for _ in range(num_freq_masks)
        ])
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x):
        # x: (B, T, F)
        if not self.training:
            return x
        B, T, F = x.shape
        x = x.clone()
        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            f0 = torch.randint(0, max(1, F - f), (1,)).item()
            x[:, :, f0:f0 + f] = 0.0
        # Time masking
        for _ in range(self.num_time_masks):
            t = torch.randint(0, min(self.time_mask_param + 1, T), (1,)).item()
            t0 = torch.randint(0, max(1, T - t), (1,)).item()
            x[:, t0:t0 + t, :] = 0.0
        return x


# ─────────────────────────────────────────────────────────────
# CONFORMER BLOCK
# ─────────────────────────────────────────────────────────────
class ConformerBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, conv_kernel_size, padding=conv_kernel_size // 2, groups=d_model),
            nn.GroupNorm(num_groups=32, num_channels=d_model),  # stable vs BatchNorm for variable-len seqs
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout)
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.final_norm = nn.LayerNorm(d_model)

    def _forward_impl(self, x, mask=None):
        x = x + 0.5 * self.ff1(x)
        normed = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(normed, normed, normed, key_padding_mask=mask)
        x = x + self.attn_dropout(attn_out)
        conv_out = self.conv(self.conv_norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + conv_out
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)

    def forward(self, x, mask=None):
        return self._forward_impl(x, mask)


# ─────────────────────────────────────────────────────────────
# CONFORMER ENCODER
# ─────────────────────────────────────────────────────────────
class ConformerEncoder(nn.Module):
    def __init__(self, input_dim=80, d_model=256, num_layers=12,
                 num_heads=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])
        self.use_gradient_checkpointing = False

    def forward(self, x, lengths=None):
        mask = None
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        # Project then cast to float32 to start the residual backbone in high precision
        x = self.pos_encoding(self.input_proj(x).float())
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)
        return x, mask


# ─────────────────────────────────────────────────────────────
# CONFORMER CTC v2
# ─────────────────────────────────────────────────────────────
class ConformerCTCv2(nn.Module):
    """
    Pure Conformer-CTC with:
      - SpecAugment
      - Gradient checkpointing
      - Beam search decode (via pyctcdecode if available, else greedy)
      - INT8 quantization export
    """
    def __init__(self, vocab_size, input_dim=80, d_model=256,
                 num_layers=12, num_heads=4, conv_kernel_size=31, dropout=0.1,
                 freq_mask_param=27, time_mask_param=100):
        super().__init__()
        self.spec_augment = SpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            num_freq_masks=2,
            num_time_masks=2
        )
        self.encoder = ConformerEncoder(
            input_dim=input_dim, d_model=d_model, num_layers=num_layers,
            num_heads=num_heads, conv_kernel_size=conv_kernel_size, dropout=dropout
        )
        self.ctc_head = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def enable_gradient_checkpointing(self):
        self.encoder.use_gradient_checkpointing = True

    def forward(self, x, lengths=None):
        """x: (B, T, F) mel features"""
        x = self.spec_augment(x)
        enc_out, mask = self.encoder(x, lengths)
        logits = self.ctc_head(enc_out)
        return logits, lengths

    @torch.no_grad()
    def greedy_decode(self, x, lengths, idx2char):
        logits, _ = self.forward(x, lengths)
        # Use float32 for argmax to be safe across precisions
        preds = torch.argmax(logits.float(), dim=-1)
        results = []
        for i in range(preds.size(0)):
            seq = preds[i, :lengths[i]].tolist()
            chars, prev = [], -1
            for idx in seq:
                if idx != prev and idx != 0:
                    chars.append(idx2char.get(str(idx), idx2char.get(idx, '')))
                prev = idx
            results.append(''.join(chars))
        return results

    @torch.no_grad()
    def beam_decode(self, x, lengths, idx2char, beam_width=10):
        """
        Beam search decode. Uses pyctcdecode if available, falls back to greedy.
        """
        logits, _ = self.forward(x, lengths)
        # Force log_softmax to float32 — critical for stability
        log_probs = F.log_softmax(logits.float(), dim=-1)
        try:
            from pyctcdecode import build_ctcdecoder
            # Build vocab list ordered by index
            max_idx = max(int(k) for k in idx2char.keys()) if isinstance(list(idx2char.keys())[0], str) else max(idx2char.keys())
            labels = [idx2char.get(str(i), idx2char.get(i, '')) for i in range(max_idx + 1)]
            labels[0] = ''  # blank token must be empty string at index 0
            decoder = build_ctcdecoder(labels)
            results = []
            for i in range(log_probs.size(0)):
                # Cast to float before numpy conversion (numpy doesn't support bf16)
                lp = log_probs[i, :lengths[i]].cpu().float().numpy()
                results.append(decoder.decode(lp, beam_width=beam_width))
            return results
        except ImportError:
            # Fallback to greedy
            return self.greedy_decode(x, lengths, idx2char)

    def quantize_int8(self):
        """Return INT8 quantized version for inference."""
        self.eval()
        quantized = torch.quantization.quantize_dynamic(
            self, {nn.Linear}, dtype=torch.qint8
        )
        return quantized


def create_model_v2(vocab_size, d_model=256, num_layers=12,
                    freq_mask_param=27, time_mask_param=100):
    return ConformerCTCv2(
        vocab_size=vocab_size,
        input_dim=80,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=4,
        conv_kernel_size=31,
        dropout=0.1,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
    )
