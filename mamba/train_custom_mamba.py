#!/usr/bin/env python3
# =============================================================================
# Option A: Custom Tiny Mamba (~2.5M params) — ASR Post-Correction
# =============================================================================
# - Pure PyTorch: zero CUDA kernel dependency (no mamba-ssm needed)
# - Uses your existing 81-char Konkani vocab from data/vocab.json
# - Trains a causal corrector: hyp_greedy -> ref
# - Augments with synthetic noise so the 52k identical rows also help
# - Runs in ~2-3 hours on Kaggle T4, ~1-2 hours on Lightning A10G
# =============================================================================

import os, json, math, random, warnings, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")   # headless — works on Kaggle with no display
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.gridspec import GridSpec

# Publication-quality plot style
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize":10,
    "figure.dpi":     150,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
})
_C = {  # color palette
    "train":"#2196F3", "val":"#F44336", "cer":"#4CAF50",
    "lr":"#FF9800",    "gap":"#9C27B0", "sub":"#E53935",
    "ins":"#FB8C00",   "del":"#43A047", "before":"#EF9A9A", "after":"#A5D6A7",
}

warnings.filterwarnings("ignore")

# =============================================================================
#  CONFIG
# =============================================================================
CONFIG = {
    # --- data ---
    "csv_path":       "./train_audit.csv",
    "vocab_path":     "../data/vocab.json",   # your 81-char Devanagari vocab

    # --- model ---
    # d_model=256, expand=2, n_layers=6 → ~2.5M params, fits T4 easily
    # d_model=384, expand=4, n_layers=8 → ~15M params (if you want bigger)
    "d_model":        256,    # embedding / SSM width
    "n_layers":       6,      # number of Mamba blocks
    "d_state":        16,     # SSM state dimension
    "d_conv":         4,      # causal conv kernel size
    "expand":         2,      # d_inner = expand * d_model (keep low for T4)
    "dropout":        0.1,

    # --- sequence ---
    "max_len":        256,    # max chars per sample — covers all Konkani utterances

    # --- training ---
    "batch_size":     8,
    "grad_accum":     4,      # effective batch = 32
    "epochs":         20,
    "lr":             1e-4,
    "weight_decay":   0.01,
    "warmup_steps":   200,
    "grad_clip":      1.0,
    "num_workers":    4,      # parallel data loading
    "seed":           42,

    # --- augmentation ---
    "augment_ratio":  0.3,    # fraction of identical rows to augment with noise
    "noise_prob":     0.1,    # per-char noise probability for augmentation

    # --- output ---
    "output_dir":     "./outputs_custom_mamba",
    "save_every":     5,      # save checkpoint every N epochs
}

# Special token ids we add on top of the 81-char vocab
# We use indices beyond 80 to avoid collisions
PAD_ID  = 0   # already in vocab
SEP_ID  = 81  # separates prompt from response
EOS_ID  = 82
VOCAB_SIZE = 83  # 81 original + SEP + EOS

# =============================================================================
#  TOKENIZER — character-level using your existing vocab.json
# =============================================================================
class KonkaniCharTokenizer:
    """
    Simple character tokenizer backed by the project's vocab.json.
    Falls back to <unk> (id=4) for unseen chars.
    Adds SEP (81) and EOS (82) on top of the original 81 chars.
    """
    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.char2idx: dict = vocab["char2idx"]
        self.idx2char: dict = {int(k): v for k, v in vocab["idx2char"].items()}
        # Register extra tokens
        self.char2idx["<sep>"] = SEP_ID
        self.char2idx["<eos>"] = EOS_ID
        self.idx2char[SEP_ID] = "<sep>"
        self.idx2char[EOS_ID] = "<eos>"
        self.pad_id  = PAD_ID
        self.sep_id  = SEP_ID
        self.eos_id  = EOS_ID
        self.unk_id  = self.char2idx.get("<unk>", 4)

    def encode(self, text: str) -> list[int]:
        return [self.char2idx.get(c, self.unk_id) for c in text]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i == self.eos_id:
                break
            if i not in (self.pad_id, self.sep_id):
                out.append(self.idx2char.get(i, "?"))
        return "".join(out)

    def encode_pair(self, src: str, tgt: str, max_len: int):
        """
        Format: [src_chars] <sep> [tgt_chars] <eos> [<pad>...]
        Labels: -100 for src+sep portion, tgt_chars + eos for supervision.
        """
        src_ids = self.encode(src)
        tgt_ids = self.encode(tgt) + [self.eos_id]
        full    = src_ids + [self.sep_id] + tgt_ids

        prompt_len = len(src_ids) + 1  # src + sep

        # Truncate to max_len
        full = full[:max_len]
        labels = [-100] * min(prompt_len, len(full))
        labels += full[prompt_len:]          # supervise only tgt portion
        labels = labels[:max_len]

        # Pad
        pad_len = max_len - len(full)
        full   += [self.pad_id] * pad_len
        labels += [-100]       * pad_len

        mask = [1 if t != self.pad_id else 0 for t in full]
        return full, labels, mask


# =============================================================================
#  DATASET — real correction pairs + synthetic noise augmentation
# =============================================================================
def augment_with_noise(text: str, char2idx: dict, noise_prob: float = 0.1) -> str:
    """
    Simulate ASR errors on a clean reference:
      - random char swap (30%), insertion (30%), deletion (40%)
    Used to turn identical (hyp==ref) rows into synthetic correction pairs.
    """
    chars = list(text)
    vocab_chars = [c for c in char2idx if len(c) == 1]  # single chars only
    result = []
    for ch in chars:
        r = random.random()
        if r < noise_prob:
            op = random.random()
            if op < 0.3 and vocab_chars:
                result.append(random.choice(vocab_chars))  # substitution
            elif op < 0.6 and vocab_chars:
                result.append(random.choice(vocab_chars))  # insertion before
                result.append(ch)
            # else: deletion (just don't append ch)
        else:
            result.append(ch)
    return "".join(result) if result else text


class CorrectionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: KonkaniCharTokenizer,
                 max_len: int, augment_ratio: float = 0.3,
                 noise_prob: float = 0.1, is_train: bool = True):
        self.tokenizer   = tokenizer
        self.max_len     = max_len
        self.noise_prob  = noise_prob
        self.pairs       = []  # list of (src, tgt)

        for _, row in df.iterrows():
            src = str(row["hyp_greedy"]).strip()
            tgt = str(row["ref"]).strip()

            if src != tgt:
                # Real correction pair — use as-is
                self.pairs.append((src, tgt))
            elif is_train and random.random() < augment_ratio:
                # Identical row — synthesise a noisy version as src
                noisy = augment_with_noise(tgt, tokenizer.char2idx, noise_prob)
                if noisy != tgt:
                    self.pairs.append((noisy, tgt))
                # else skip (noise didn't change anything)

        print(f"  Dataset size: {len(self.pairs)} pairs "
              f"(from {len(df)} rows, real errors: {(df['hyp_greedy']!=df['ref']).sum()})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        ids, labels, mask = self.tokenizer.encode_pair(src, tgt, self.max_len)
        return {
            "input_ids":      torch.tensor(ids,    dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask,   dtype=torch.long),
        }


# =============================================================================
#  MAMBA BLOCK — pure PyTorch, no custom CUDA kernels required
# =============================================================================
class MambaBlock(nn.Module):
    """
    One Mamba (selective SSM) block.
    Implements the S6 core via a sequential scan — works on any device/CUDA version.
    For production speed you can swap _ssm() for a parallel scan later.
    """
    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state

        self.norm    = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Strictly causal depthwise conv: pad left only, no automatic padding.
        # Input will be F.pad(..., (d_conv-1, 0)) before this layer so output
        # length stays exactly L and each position only sees past+current tokens.
        self.conv1d  = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=0,
            groups=self.d_inner, bias=True
        )

        # SSM projections
        self.x_proj  = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A: log-parameterised for numerical stability (always negative after exp)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.log_A = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1).clone())

        # D: skip connection scale
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

        # Initialise dt bias so initial delta values are sensible (~0.01–0.3)
        nn.init.uniform_(self.dt_proj.bias, -4.0, -1.0)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, L, D)
        residual = x
        x = self.norm(x)

        # Apply attention mask to zero out padding contributions in SSM
        if attention_mask is not None:
            # attention_mask: (B, L) with 1 for real tokens, 0 for padding
            x = x * attention_mask.unsqueeze(-1)

        # Split into SSM stream and gate
        xz          = self.in_proj(x)                      # (B, L, 2*d_inner)
        x_ssm, gate = xz.chunk(2, dim=-1)                  # each (B, L, d_inner)

        # Causal conv: left-pad by (d_conv-1), no right pad → strictly causal
        # Output length = L, each position only sees current + past tokens
        x_conv = F.pad(x_ssm.transpose(1, 2), (self.conv1d.kernel_size[0] - 1, 0))
        x_conv = self.conv1d(x_conv).transpose(1, 2)       # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # Apply mask again after conv (padding positions should not contribute)
        if attention_mask is not None:
            x_conv = x_conv * attention_mask.unsqueeze(-1)

        # SSM core
        y = self._ssm(x_conv)

        # Gated output
        out = y * F.silu(gate)
        out = self.drop(self.out_proj(out))
        return out + residual

    # ------------------------------------------------------------------
    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        A = -torch.exp(self.log_A.float())                  # (d_inner, d_state)

        # Input-dependent B, C, delta
        bcdt              = self.x_proj(x)                  # (B, L, 2*d_state+1)
        B_mat, C_mat, dt  = bcdt.split([self.d_state, self.d_state, 1], dim=-1)
        delta = F.softplus(self.dt_proj(dt))                # (B, L, d_inner)

        # Discretise via Zero-Order Hold
        # dA: (B, L, d_inner, d_state) — A is negative so dA stays in (0,1)
        dA = torch.exp(delta.unsqueeze(-1) * A)
        dB_u = delta.unsqueeze(-1) * B_mat.unsqueeze(2) * x.unsqueeze(-1)

        # Sequential scan in float32 — numerically stable, no log-space needed.
        # ponytail: O(L) loop — fine for L=256. Upgrade to mamba-ssm parallel
        #           scan if L > 512 or throughput becomes a bottleneck.
        x_f   = x.float()
        dA_f  = dA.float()
        dBu_f = dB_u.float()
        h = torch.zeros(x_f.size(0), self.d_inner, self.d_state,
                        device=x.device, dtype=torch.float32)
        ys = []
        for t in range(x_f.size(1)):
            h = dA_f[:, t] * h + dBu_f[:, t] * x_f[:, t].unsqueeze(-1)
            ys.append((h * C_mat[:, t].float().unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1).to(x.dtype)
        return y + x * self.D                                # skip connection


# =============================================================================
#  FULL MODEL — stacked Mamba blocks + embedding + LM head
# =============================================================================
class TinyMambaCorrectorModel(nn.Module):
    """
    Decoder-only Mamba language model (~2.5M params at default CONFIG).
    Trained as a causal corrector: given [noisy_src <sep>], predict [clean_tgt <eos>].
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.layers     = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out   = nn.LayerNorm(d_model)
        self.lm_head    = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embedding and lm_head weights (standard practice, saves params)
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor,
                labels: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None):
        # input_ids: (B, L)
        x = self.embedding(input_ids)           # (B, L, d_model)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.norm_out(x)
        logits = self.lm_head(x)                # (B, L, vocab_size)

        loss = None
        if labels is not None:
            # Shift: predict token[t+1] from token[t]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, src_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                 max_new: int = 200, temperature: float = 0.8, top_k: int = 40) -> list[int]:
        """
        Top-k sampling generation after the <sep> token.
        src_ids: (1, L) — encoded source already including <sep>
        temperature: < 1.0 = more conservative/greedy, > 1.0 = more random
        """
        was_training = self.training
        self.eval()

        ids = src_ids.clone()
        # Build attention mask for generated sequence (all 1s since we generate one at a time)
        if attention_mask is None:
            gen_mask = torch.ones_like(ids)
        else:
            gen_mask = attention_mask.clone()

        for _ in range(max_new):
            # Pass full attention mask
            logits, _ = self(ids, attention_mask=gen_mask)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, top_k)
                threshold = topk_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float('-inf'))
            probs    = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            ids      = torch.cat([ids, next_tok], dim=1)
            gen_mask = torch.cat([gen_mask, torch.ones((1, 1), dtype=gen_mask.dtype, device=gen_mask.device)], dim=1)
            if next_tok.item() == EOS_ID:
                break

        if was_training:
            self.train()
        return ids[0, src_ids.size(1):].tolist()

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
#  TRAINING LOOP
# =============================================================================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss, steps = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask   = batch["attention_mask"].to(device)
            _, loss = model(ids, labels=labels, attention_mask=mask)
            if loss is not None:
                total_loss += loss.item()
                steps += 1
    model.train()
    return total_loss / max(steps, 1)


def compute_cer(pred: str, ref: str) -> float:
    """Character error rate via edit distance."""
    if not ref:
        return 1.0  # undefined: return worst case
    if not pred:
        return 1.0
    n, m = len(pred), len(ref)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            prev, dp[j] = dp[j], prev if pred[i-1] == ref[j-1]                 else 1 + min(prev, dp[j], dp[j-1])
    return dp[m] / m


def quick_cer_eval(model, df_sample, tokenizer, device, n=200) -> float:
    """Run CER on a random sample of real error pairs (n=200 for stable estimates)."""
    error_rows = df_sample[df_sample["hyp_greedy"] != df_sample["ref"]]
    sample = error_rows.sample(min(n, len(error_rows)), random_state=0)
    cer_scores = []
    model.eval()
    with torch.no_grad():
        for _, row in sample.iterrows():
            src  = str(row["hyp_greedy"]).strip()
            ref  = str(row["ref"]).strip()
            src_ids = tokenizer.encode(src) + [SEP_ID]
            src_t   = torch.tensor([src_ids], dtype=torch.long, device=device)
            mask    = torch.ones_like(src_t)
            out_ids = model.generate(src_t, attention_mask=mask, max_new=len(ref) + 20)
            pred    = tokenizer.decode(out_ids)
            cer_scores.append(compute_cer(pred, ref))
    model.train()
    return float(np.mean(cer_scores)) if cer_scores else 1.0


def train():
    set_seed(CONFIG["seed"])
    torch.set_float32_matmul_precision("medium")  # faster matmuls on Ampere+

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # ---- Load vocab & data ----
    tokenizer = KonkaniCharTokenizer(CONFIG["vocab_path"])
    print(f"Vocab size (with special tokens): {VOCAB_SIZE}")

    df = pd.read_csv(CONFIG["csv_path"]).dropna(subset=["hyp_greedy", "ref"])
    df["hyp_greedy"] = df["hyp_greedy"].astype(str).str.strip()
    df["ref"]        = df["ref"].astype(str).str.strip()
    print(f"Total rows: {len(df):,}  |  Real errors: {(df['hyp_greedy']!=df['ref']).sum():,}")

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=CONFIG["seed"])

    print("\nBuilding train dataset...")
    train_ds = CorrectionDataset(train_df, tokenizer, CONFIG["max_len"],
                                 CONFIG["augment_ratio"], CONFIG["noise_prob"], is_train=True)
    print("Building val dataset...")
    val_ds   = CorrectionDataset(val_df,   tokenizer, CONFIG["max_len"],
                                 augment_ratio=0.0, noise_prob=0.0, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=CONFIG["num_workers"],
                              pin_memory=device.type=="cuda")
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=CONFIG["num_workers"],
                              pin_memory=device.type=="cuda")

    # ---- Build model ----
    model = TinyMambaCorrectorModel(
        vocab_size = VOCAB_SIZE,
        d_model    = CONFIG["d_model"],
        n_layers   = CONFIG["n_layers"],
        d_state    = CONFIG["d_state"],
        d_conv     = CONFIG["d_conv"],
        expand     = CONFIG["expand"],
        dropout    = CONFIG["dropout"],
    ).to(device)
    print(f"\nModel parameters: {model.count_params():,}")

    # torch.compile for extra speed (PyTorch 2.0+, skips gracefully if unavailable)
    try:
        model = torch.compile(model)
        print("torch.compile enabled")
    except Exception:
        print("torch.compile not available, running eagerly")

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    total_steps   = len(train_loader) * CONFIG["epochs"] // CONFIG["grad_accum"]
    warmup_steps  = CONFIG["warmup_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))  # cosine decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ---- Training ----
    best_val_loss = float("inf")
    global_step   = 0
    log_path      = os.path.join(CONFIG["output_dir"], "training_log.csv")
    log_rows      = []

    print(f"\nStarting training for {CONFIG['epochs']} epochs...\n")

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss, n_steps = 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            ids    = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask   = batch["attention_mask"].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                _, loss = model(ids, labels=labels, attention_mask=mask)
                loss    = loss / CONFIG["grad_accum"]

            # Skip NaN batches — can happen from log-space SSM instability
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * CONFIG["grad_accum"]
            n_steps    += 1

            if (step + 1) % CONFIG["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        train_loss = epoch_loss / max(n_steps, 1)
        val_loss   = evaluate(model, val_loader, device)
        cer        = quick_cer_eval(model, val_df, tokenizer, device, n=200)

        print(f"Epoch {epoch:3d}/{CONFIG['epochs']}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  CER={cer:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        log_rows.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "cer": cer})
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": val_loss, "cer": cer, "config": CONFIG},
                       os.path.join(CONFIG["output_dir"], "best_model.pt"))
            print(f"  → Saved best model (val_loss={val_loss:.4f})")

        # Periodic checkpoint
        if epoch % CONFIG["save_every"] == 0:
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(), "config": CONFIG},
                       os.path.join(CONFIG["output_dir"], f"checkpoint_epoch{epoch:03d}.pt"))

    print(f"\nDone. Best val_loss: {best_val_loss:.4f}")
    print(f"Logs saved to: {log_path}")
    print(f"Best model:    {CONFIG['output_dir']}/best_model.pt")

    # Auto-generate all research paper figures
    print("\nGenerating paper figures...")
    generate_plots(
        log_rows  = log_rows,
        df_data   = df,          # full dataset (has hyp_greedy, ref)
        model     = model,
        tokenizer = tokenizer,
        val_df    = val_df,
        device    = device,
        out_dir   = os.path.join(CONFIG["output_dir"], "figures"),
    )


# =============================================================================
#  RESEARCH PAPER FIGURE GENERATION
# =============================================================================
def _savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def _edit_ops(hyp: str, ref: str):
    """Returns (dist, n_sub, n_ins, n_del) and backpointer matrix for alignment."""
    n, m = len(hyp), len(ref)
    # dp[i][j] = (distance, subs, ins, dels)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    for j in range(1, m + 1):
        dp[0][j] = (j, 0, j, 0)
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, 0, i)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if hyp[i-1] == ref[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                s = dp[i-1][j-1]
                ins = dp[i][j-1]
                d = dp[i-1][j]
                cands = [
                    (s[0]+1, s[1]+1, s[2], s[3]),      # substitution
                    (ins[0]+1, ins[1], ins[2]+1, ins[3]), # insertion
                    (d[0]+1, d[1], d[2], d[3]+1)        # deletion
                ]
                dp[i][j] = min(cands)
    return dp[n][m]


def _get_alignment(hyp: str, ref: str):
    """Returns list of (hyp_char, ref_char) aligned pairs using DP backtracking.
    None indicates insertion/deletion."""
    n, m = len(hyp), len(ref)
    dp = [[0]*(m+1) for _ in range(n+1)]

    for j in range(m+1):
        dp[0][j] = j
    for i in range(n+1):
        dp[i][0] = i

    for i in range(1, n+1):
        for j in range(1, m+1):
            if hyp[i-1] == ref[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j])

    # Backtrack
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and hyp[i-1] == ref[j-1]:
            alignment.append((hyp[i-1], ref[j-1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append((hyp[i-1], ref[j-1]))  # substitution
            i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            alignment.append((None, ref[j-1]))  # insertion in ref
            j -= 1
        else:
            alignment.append((hyp[i-1], None))  # deletion from ref
            i -= 1

    alignment.reverse()
    return alignment


def generate_plots(log_rows, df_data, model, tokenizer, val_df, device, out_dir):
    """Generate all 13 publication figures from training data + trained model."""
    df = pd.DataFrame(log_rows)
    fig_dir = out_dir
    os.makedirs(fig_dir, exist_ok=True)

    def fp(name): return os.path.join(fig_dir, name)

    # ------------------------------------------------------------------
    # 1. Train / Val Loss Curves
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_loss"], color=_C["train"], lw=2, marker="o", ms=3, label="Train Loss")
    ax.plot(df["epoch"], df["val_loss"],   color=_C["val"],   lw=2, marker="s", ms=3, label="Val Loss", ls="--")
    best_ep = df.loc[df["val_loss"].idxmin(), "epoch"]
    best_vl = df["val_loss"].min()
    ax.axvline(best_ep, color="gray", ls=":", alpha=0.7)
    ax.annotate(f"Best\n{best_vl:.4f}", xy=(best_ep, best_vl),
                xytext=(best_ep+0.6, best_vl+0.02), fontsize=9, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training and Validation Loss"); ax.legend()
    _savefig(fig, fp("01_loss_curves.pdf"))

    # ------------------------------------------------------------------
    # 2. CER Curve
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    cer_pct = df["cer"] * 100
    ax.plot(df["epoch"], cer_pct, color=_C["cer"], lw=2, marker="^", ms=4)
    ax.fill_between(df["epoch"], cer_pct, alpha=0.1, color=_C["cer"])
    best_cer = cer_pct.min()
    ax.axhline(best_cer, color="gray", ls=":", alpha=0.6)
    ax.set_xlabel("Epoch"); ax.set_ylabel("CER (%)")
    ax.set_title(f"CER During Training  (best = {best_cer:.1f}%)")
    _savefig(fig, fp("02_cer_curve.pdf"))

    # ------------------------------------------------------------------
    # 3. Learning Rate Schedule (reconstructed from warmup+cosine)
    # ------------------------------------------------------------------
    total_ep  = len(df)
    warmup_ep = max(1, CONFIG["warmup_steps"] // max(1, total_ep))
    lr_vals   = []
    for ep in range(total_ep):
        if ep < warmup_ep:
            lr_vals.append(CONFIG["lr"] * ep / warmup_ep)
        else:
            prog = (ep - warmup_ep) / max(total_ep - warmup_ep, 1)
            lr_vals.append(CONFIG["lr"] * max(0.1, 0.5*(1 + math.cos(math.pi*prog))))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(df["epoch"], lr_vals, color=_C["lr"], lw=2)
    ax.fill_between(df["epoch"], lr_vals, alpha=0.1, color=_C["lr"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (Warmup + Cosine Decay)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    _savefig(fig, fp("03_lr_schedule.pdf"))

    # ------------------------------------------------------------------
    # 4. Generalisation Gap
    # ------------------------------------------------------------------
    gap = df["val_loss"] - df["train_loss"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(df["epoch"], gap, color=[_C["gap"] if g>0 else _C["train"] for g in gap],
           alpha=0.75, width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val − Train Loss")
    ax.set_title("Generalisation Gap (positive = overfitting)")
    _savefig(fig, fp("04_generalisation_gap.pdf"))

    # ------------------------------------------------------------------
    # 5. ΔCER per Epoch
    # ------------------------------------------------------------------
    delta = -df["cer"].diff().fillna(0) * 100
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(df["epoch"], delta,
           color=[_C["cer"] if d>=0 else _C["val"] for d in delta], alpha=0.8, width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("ΔCER (pp)")
    ax.set_title("CER Improvement per Epoch")
    _savefig(fig, fp("05_cer_delta.pdf"))

    # ------------------------------------------------------------------
    # 6. Error Type Distribution — computed on real error rows
    # ------------------------------------------------------------------
    error_rows = df_data[df_data["hyp_greedy"] != df_data["ref"]].sample(
        min(500, (df_data["hyp_greedy"]!=df_data["ref"]).sum()), random_state=42)
    tot_s=tot_i=tot_d = 0
    for _, row in error_rows.iterrows():
        _, s, i, d = _edit_ops(str(row["hyp_greedy"]).strip(), str(row["ref"]).strip())
        tot_s+=s; tot_i+=i; tot_d+=d
    counts = [tot_s, tot_i, tot_d]
    labels = ["Substitutions","Insertions","Deletions"]
    colors = [_C["sub"],_C["ins"],_C["del"]]
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    bars = axes[0].bar(labels, counts, color=colors, alpha=0.85, width=0.5)
    total_err = max(sum(counts),1)
    for bar, cnt in zip(bars, counts):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                     f"{cnt/total_err*100:.1f}%", ha="center", fontsize=9)
    axes[0].set_ylabel("Count"); axes[0].set_title("Error Type Distribution")
    axes[1].pie(counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(linewidth=1, edgecolor="white"))
    axes[1].set_title("Error Proportions")
    fig.suptitle("ASR Error Analysis (Before Correction)", fontsize=13)
    _savefig(fig, fp("06_error_type_distribution.pdf"))

    # ------------------------------------------------------------------
    # 7. CER Before vs After — using REAL model predictions
    # ------------------------------------------------------------------
    eval_rows = error_rows.head(min(200, len(error_rows)))
    cer_before, cer_after, predictions = [], [], []
    for _, row in eval_rows.iterrows():
        src = str(row["hyp_greedy"]).strip()
        ref = str(row["ref"]).strip()
        src_ids = tokenizer.encode(src) + [SEP_ID]
        src_t   = torch.tensor([src_ids], dtype=torch.long, device=device)
        mask    = torch.ones_like(src_t)
        out_ids = model.generate(src_t, attention_mask=mask, max_new=len(ref)+20)
        pred    = tokenizer.decode(out_ids)
        cer_before.append(compute_cer(src, ref))
        cer_after.append(compute_cer(pred, ref))
        predictions.append(pred)

    mb = np.mean(cer_before)*100; ma = np.mean(cer_after)*100
    fig, axes = plt.subplots(1,2, figsize=(11,4))
    bars = axes[0].bar(["Before\n(ASR output)","After\n(TinyMamba)"],
                       [mb, ma], color=[_C["before"],_C["after"]], alpha=0.9, width=0.4)
    for bar, val in zip(bars,[mb,ma]):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Mean CER (%)"); axes[0].set_ylim(0, max(mb,ma)*1.3)
    axes[0].set_title(f"CER Reduction: {(mb-ma)/mb*100:.1f}%")
    axes[1].hist([c*100 for c in cer_before], bins=20, alpha=0.6, color=_C["before"],
                 label="Before", density=True)
    axes[1].hist([c*100 for c in cer_after],  bins=20, alpha=0.6, color=_C["after"],
                 label="After",  density=True)
    axes[1].set_xlabel("CER (%)"); axes[1].set_ylabel("Density")
    axes[1].set_title("CER Distribution: Before vs After"); axes[1].legend()
    fig.suptitle("Post-Correction Quality — Real Model Predictions", fontsize=13)
    _savefig(fig, fp("07_cer_before_after.pdf"))

    # ------------------------------------------------------------------
    # 8. Edit Distance Scatter — before vs after (real predictions)
    # ------------------------------------------------------------------
    ed_before = [_edit_ops(str(r["hyp_greedy"]).strip(), str(r["ref"]).strip())[0]
                 for _, r in eval_rows.iterrows()]
    ed_after  = [_edit_ops(pred, str(r["ref"]).strip())[0]
                 for pred, (_, r) in zip(predictions, eval_rows.iterrows())]
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.scatter(ed_before, ed_after, alpha=0.4, s=18,
                    c=[b-a for b,a in zip(ed_before,ed_after)], cmap="RdYlGn")
    mx = max(max(ed_before),max(ed_after))+1
    ax.plot([0,mx],[0,mx],"k--",alpha=0.4,lw=1,label="No change")
    ax.set_xlabel("Edit Distance Before"); ax.set_ylabel("Edit Distance After")
    ax.set_title("Sample Edit Distance: Before vs After (Real Predictions)")
    fig.colorbar(sc, ax=ax, label="Improvement (chars)")
    ax.legend()
    _savefig(fig, fp("08_edit_distance_scatter.pdf"))

    # ------------------------------------------------------------------
    # 9. Character Confusion Heatmap (top-20) — FIXED: uses DP alignment
    # ------------------------------------------------------------------
    confusions = defaultdict(int)
    for _, row in error_rows.iterrows():
        hyp = str(row["hyp_greedy"]).strip()
        ref = str(row["ref"]).strip()
        alignment = _get_alignment(hyp, ref)
        for h_char, r_char in alignment:
            if h_char is not None and r_char is not None and h_char != r_char:
                confusions[(h_char, r_char)] += 1
            elif h_char is not None and r_char is None:
                # deletion: hyp char deleted
                pass
            elif h_char is None and r_char is not None:
                # insertion
                pass

    top = sorted(confusions.items(), key=lambda x:x[1], reverse=True)[:20]
    if top:
        lab=[f"{h}→{r}" for (h,r),_ in top]; cnt=[c for _,c in top]
        fig, ax = plt.subplots(figsize=(6,7))
        cols = plt.cm.Reds(np.linspace(0.3,0.9,len(cnt)))
        bars = ax.barh(range(len(lab)), cnt, color=cols, alpha=0.9)
        ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=9)
        ax.invert_yaxis(); ax.set_xlabel("Count")
        ax.set_title("Top-20 Character Confusions (ASR Errors)")
        for bar,c in zip(bars,cnt):
            ax.text(bar.get_width()+0.5,bar.get_y()+bar.get_height()/2,
                    str(c),va="center",fontsize=8)
        _savefig(fig, fp("09_confusion_heatmap.pdf"))

    # ------------------------------------------------------------------
    # 10. Parameter Breakdown
    # ------------------------------------------------------------------
    dm = CONFIG["d_model"]; ex = CONFIG["expand"]; nl = CONFIG["n_layers"]
    # More accurate parameter counting
    emb_p  = VOCAB_SIZE * dm
    # Per block:
    # in_proj: d_model -> 2*d_inner
    in_proj_p = dm * (2 * dm * ex)
    # conv1d: d_inner * d_conv (depthwise)
    conv_p = (dm * ex) * CONFIG["d_conv"]
    # x_proj: d_inner -> (2*d_state + 1)
    x_proj_p = (dm * ex) * (2 * CONFIG["d_state"] + 1)
    # dt_proj: 1 -> d_inner
    dt_proj_p = 1 * (dm * ex) + (dm * ex)  # weight + bias
    # log_A: d_inner * d_state
    log_A_p = (dm * ex) * CONFIG["d_state"]
    # D: d_inner
    D_p = dm * ex
    # out_proj: d_inner -> d_model
    out_proj_p = (dm * ex) * dm
    blk_p = (in_proj_p + conv_p + x_proj_p + dt_proj_p + log_A_p + D_p + out_proj_p) * nl
    norm_p = dm
    sizes  = [emb_p, blk_p, norm_p]
    lab    = ["Embedding\n(tied w/ LM head)", f"Mamba Blocks\n({nl} layers)", "Output Norm"]
    fig, axes = plt.subplots(1,2, figsize=(11,4.5))
    axes[0].pie(sizes, labels=lab, autopct="%1.1f%%", startangle=90,
                colors=["#FFCDD2","#90CAF9","#C5E1A5"],
                wedgeprops=dict(linewidth=1,edgecolor="white"))
    axes[0].set_title("Parameter Distribution")
    axes[1].barh(lab,[s/1e6 for s in sizes],color=["#FFCDD2","#90CAF9","#C5E1A5"],alpha=0.9)
    for i,val in enumerate(sizes):
        axes[1].text(val/1e6+0.1, i, f"{val/1e6:.1f}M", va="center", fontweight="bold")
    axes[1].set_xlabel("Parameters (M)")
    axes[1].set_title(f"Total: {sum(sizes)/1e6:.1f}M params")
    fig.suptitle(f"TinyMamba Size (d={dm}, expand={ex}, layers={nl})", fontsize=13)
    _savefig(fig, fp("10_param_breakdown.pdf"))

    # ------------------------------------------------------------------
    # 11. Convergence — train vs val on same axis (good for ablation slot)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.plot(df["epoch"], df["val_loss"], color=_C["train"], lw=2.5, marker="o", ms=3,
            label=f"d_model={dm} (ours)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Convergence")
    ax.legend()
    _savefig(fig, fp("11_convergence.pdf"))

    # ------------------------------------------------------------------
    # 12. Qualitative Examples Table (real predictions)
    # ------------------------------------------------------------------
    sample = eval_rows.head(min(8, len(eval_rows)))
    table_data = [["#","ASR Output","Model Correction","Ground Truth"]]
    for i, ((_, row), pred) in enumerate(zip(sample.iterrows(), predictions[:8]), 1):
        table_data.append([str(i),
                           str(row["hyp_greedy"])[:45],
                           pred[:45],
                           str(row["ref"])[:45]])
    fig, ax = plt.subplots(figsize=(13,5))
    ax.axis("off")
    tbl = ax.table(cellText=table_data, cellLoc="left", loc="center",
                   colWidths=[0.04,0.35,0.3,0.31])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1,2.3)
    for j in range(4):
        tbl[(0,j)].set_facecolor("#BBDEFB")
        tbl[(0,j)].set_text_props(weight="bold")
    ax.set_title("Qualitative Correction Examples (Real Model Output)", fontsize=13, pad=20)
    _savefig(fig, fp("12_qualitative_examples.pdf"))

    # ------------------------------------------------------------------
    # 13. Dashboard — one-page summary
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14,9))
    gs  = GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.35)
    a1=fig.add_subplot(gs[0,0])
    a1.plot(df["epoch"],df["train_loss"],color=_C["train"],lw=2,label="Train")
    a1.plot(df["epoch"],df["val_loss"],  color=_C["val"],  lw=2,label="Val",ls="--")
    a1.set_title("Loss Curves"); a1.set_xlabel("Epoch"); a1.legend(fontsize=8)
    a2=fig.add_subplot(gs[0,1])
    a2.plot(df["epoch"],df["cer"]*100,color=_C["cer"],lw=2,marker="^",ms=3)
    a2.fill_between(df["epoch"],df["cer"]*100,alpha=0.1,color=_C["cer"])
    a2.set_title("CER (%)"); a2.set_xlabel("Epoch")
    a3=fig.add_subplot(gs[0,2])
    a3.bar(df["epoch"],gap,color=[_C["gap"] if g>0 else _C["train"] for g in gap],alpha=0.75,width=0.6)
    a3.axhline(0,color="black",lw=0.8); a3.set_title("Generalisation Gap"); a3.set_xlabel("Epoch")
    a4=fig.add_subplot(gs[1,0])
    a4.bar(df["epoch"],delta,color=[_C["cer"] if d>=0 else _C["val"] for d in delta],alpha=0.8,width=0.6)
    a4.axhline(0,color="black",lw=0.8); a4.set_title("ΔCER / Epoch"); a4.set_xlabel("Epoch")
    a5=fig.add_subplot(gs[1,1])
    a5.bar(["Before","After"],[mb,ma],color=[_C["before"],_C["after"]],alpha=0.9,width=0.4)
    for pos,val in zip([0,1],[mb,ma]):
        a5.text(pos,val+0.3,f"{val:.1f}%",ha="center",fontsize=10,fontweight="bold")
    a5.set_ylabel("Mean CER (%)"); a5.set_title("Before vs After Correction")
    a6=fig.add_subplot(gs[1,2]); a6.axis("off")
    best_vl2 = df["val_loss"].min(); best_cer2=df["cer"].min()*100
    best_ep2 = df.loc[df["val_loss"].idxmin(),"epoch"]
    summary=(f"Training Summary\n{'─'*24}\n"
             f"Epochs:           {len(df)}\n"
             f"Best val loss:    {best_vl2:.4f}\n"
             f"Best epoch:       {best_ep2}\n"
             f"Best CER:         {best_cer2:.1f}%\n"
             f"Mean CER before:  {mb:.1f}%\n"
             f"Mean CER after:   {ma:.1f}%\n"
             f"CER reduction:    {(mb-ma)/mb*100:.1f}%\n"
             f"Model params:     {sum(sizes)/1e6:.1f}M")
    a6.text(0.05,0.95,summary,transform=a6.transAxes,fontsize=9.5,
            verticalalignment="top",fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5",facecolor="#F5F5F5",edgecolor="#BDBDBD"))
    fig.suptitle("TinyMamba Konkani ASR Post-Correction — Training Dashboard",
                 fontsize=14,fontweight="bold",y=1.01)
    _savefig(fig, fp("00_dashboard.pdf"))

    print(f"\nAll 13 figures saved to {fig_dir}/")
    print("Ready for LaTeX:  \\includegraphics[width=\\linewidth]{figures/01_loss_curves.pdf}")


if __name__ == "__main__":
    # Optional: add argparse for CLI usage
    parser = argparse.ArgumentParser(description="TinyMamba ASR Post-Correction for Konkani")
    parser.add_argument("--csv", default=CONFIG["csv_path"], help="Path to training CSV")
    parser.add_argument("--vocab", default=CONFIG["vocab_path"], help="Path to vocab.json")
    parser.add_argument("--output", default=CONFIG["output_dir"], help="Output directory")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"], help="Batch size")
    parser.add_argument("--lr", type=float, default=CONFIG["lr"], help="Learning rate")
    args = parser.parse_args()

    # Update config with CLI args
    CONFIG["csv_path"] = args.csv
    CONFIG["vocab_path"] = args.vocab
    CONFIG["output_dir"] = args.output
    CONFIG["epochs"] = args.epochs
    CONFIG["batch_size"] = args.batch_size
    CONFIG["lr"] = args.lr

    train()