#!/usr/bin/env python3
# =============================================================================
# Option B: Knowledge Distillation — mamba-130m-hf Teacher → TinyMamba Student
# =============================================================================
# HOW IT WORKS:
#   1. Load the existing LoRA-trained mamba-130m-hf (teacher) — already in
#      outputs_fullrun/final_model from mambatrain.py, or we download it fresh.
#   2. Run the teacher ONCE on every sample to produce soft logit distributions.
#   3. Train the TinyMamba student (~25M params from train_custom_mamba.py)
#      with a COMBINED loss:
#         loss = α * CE(student, hard_labels)      # learn from ground truth
#              + β * KL(student, teacher_softmax)  # mimic teacher's distribution
#   4. The student is pure PyTorch — no CUDA kernel dependency at inference time.
#
# PREREQUISITES:
#   - Run mambatrain.py first (or fix its kernel issue) to get the teacher.
#   - OR: point TEACHER_PATH to "state-spaces/mamba-130m-hf" to use the raw
#         pretrained model as teacher (weaker signal, but still works).
#   - pip install transformers peft mamba-ssm causal-conv1d sentencepiece
#     (teacher loading requires mamba-ssm; student inference does NOT)
#
# NOTES ON THE KERNEL ISSUE:
#   The teacher is loaded once for soft-label generation. If mamba-ssm fails to
#   install, use CPU offload: set TEACHER_DEVICE = "cpu" below. It's slow for
#   generation but the student trains on GPU fine.
# =============================================================================

import os, json, math, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Reuse shared pieces from Option A
from train_custom_mamba import (
    KonkaniCharTokenizer, TinyMambaCorrectorModel, augment_with_noise,
    set_seed, evaluate, quick_cer_eval, compute_cer,
    SEP_ID, EOS_ID, PAD_ID, VOCAB_SIZE, CONFIG as BASE_CONFIG
)

# =============================================================================
#  DISTILLATION CONFIG (extends BASE_CONFIG)
# =============================================================================
CONFIG = {**BASE_CONFIG,  # inherit all base settings

    # --- teacher ---
    "teacher_path":    "./outputs_fullrun/final_model",  # from mambatrain.py
    # Fallback: "state-spaces/mamba-130m-hf"  (downloads ~500MB)
    "teacher_device":  "cuda",   # set "cpu" if mamba-ssm kernel fails on GPU
    "teacher_dtype":   "float16",

    # --- distillation ---
    "temperature":     4.0,      # softer distributions = richer signal
    "alpha":           0.5,      # weight for hard-label CE loss
    "beta":            0.5,      # weight for KL distillation loss
    "soft_label_file": "./outputs_distill/soft_labels.pt",  # cache teacher outputs

    # --- student (same arch as Option A) ---
    "output_dir":      "./outputs_distill",
    "epochs":          15,       # fewer epochs needed with distillation
    "lr":              2e-4,
    "batch_size":      8,
    "grad_accum":      4,
}

# =============================================================================
#  TEACHER MODEL LOADER (Mamba-130M from HuggingFace / LoRA)
# =============================================================================
def load_teacher_model(path_or_id: str, device: str, dtype: str):
    """
    Loads the teacher (mamba-130m-hf) from either:
      - A local dir containing saved model files (outputs_fullrun/final_model)
      - HuggingFace model ID (state-spaces/mamba-130m-hf)
    """
    print(f"Loading teacher from {path_or_id}...")
    from transformers import AutoModelForCausalLM
    import sentencepiece as spm

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            path_or_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        print(f"Teacher loaded successfully on {device}.")
    except Exception as e:
        print(f"ERROR loading teacher: {e}")
        print("If this is a CUDA kernel issue, try CONFIG['teacher_device']='cpu'")
        raise

    # Load the SentencePiece tokenizer from same folder
    try:
        sp_path = os.path.join(path_or_id, "konkani_spm.model")
        if not os.path.exists(sp_path):
            sp_path = "./outputs_fullrun/konkani_tokenizer/konkani_spm.model"
        sp_proc = spm.SentencePieceProcessor()
        sp_proc.Load(sp_path)
        print(f"Teacher tokenizer loaded from {sp_path}")
    except:
        print("WARNING: Teacher tokenizer not found. Using char tokenizer as fallback.")
        sp_proc = None

    return model, sp_proc


# =============================================================================
#  SOFT LABEL GENERATION — run teacher once, cache results
# =============================================================================
def generate_soft_labels(df: pd.DataFrame, teacher_model, sp_proc,
                          char_tokenizer: KonkaniCharTokenizer,
                          teacher_device: str, max_len: int,
                          temperature: float, save_path: str):
    """
    Run the teacher on every sample and save the top-K soft logits.
    We only store top-32 logits per position to keep file size manageable.
    Saves to save_path as a .pt file so you only run this once.
    """
    if os.path.exists(save_path):
        print(f"Soft labels cache found at {save_path}, skipping generation.")
        return torch.load(save_path)

    print(f"Generating soft labels with teacher (T={temperature})...")
    print("  This runs the teacher model once — takes ~10-20 min on T4.")

    TOPK = 32  # store top-K per position to save memory

    # We only generate for error rows (identical rows get zero KL benefit)
    error_df = df[df["hyp_greedy"] != df["ref"]].reset_index(drop=True)
    print(f"  Generating for {len(error_df):,} error rows...")

    soft_labels = {}  # idx -> {"topk_vals": tensor, "topk_ids": tensor}

    teacher_model.eval()
    with torch.no_grad():
        for i, (_, row) in enumerate(error_df.iterrows()):
            src = str(row["hyp_greedy"]).strip()
            tgt = str(row["ref"]).strip()

            # Encode with teacher's own tokenizer if available, else char-level
            if sp_proc is not None:
                prompt    = f"Correct this Konkani ASR text: {src}\nCorrected: "
                full_text = prompt + tgt + "\n"
                full_ids  = sp_proc.EncodeAsIds(full_text)[:max_len]
                prompt_len = len(sp_proc.EncodeAsIds(prompt))
            else:
                src_ids   = char_tokenizer.encode(src) + [SEP_ID]
                tgt_ids   = char_tokenizer.encode(tgt) + [EOS_ID]
                full_ids  = (src_ids + tgt_ids)[:max_len]
                prompt_len = len(src_ids)

            input_t = torch.tensor([full_ids], dtype=torch.long).to(teacher_device)

            try:
                out = teacher_model(input_ids=input_t)
                logits = out.logits[0]  # (L, V)

                # Only supervise the target portion
                logits_tgt = logits[prompt_len:, :]         # (tgt_len, V)
                logits_soft = logits_tgt / temperature       # apply temperature
                probs       = F.softmax(logits_soft, dim=-1) # (tgt_len, V)

                topk_probs, topk_ids = probs.topk(TOPK, dim=-1)
                soft_labels[i] = {
                    "topk_probs": topk_probs.cpu().half(),   # (tgt_len, TOPK)
                    "topk_ids":   topk_ids.cpu(),            # (tgt_len, TOPK)
                    "prompt_len": prompt_len,
                    "full_len":   len(full_ids),
                }
            except Exception as e:
                # If teacher fails on a sample, skip gracefully
                soft_labels[i] = None

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(error_df)} rows processed...")

    # Attach original dataframe index mapping
    soft_labels["_index_mapping"] = error_df.index.tolist()
    torch.save(soft_labels, save_path)
    print(f"Soft labels saved to {save_path}")
    return soft_labels


# =============================================================================
#  DISTILLATION DATASET
# =============================================================================
class DistillationDataset(Dataset):
    """
    Like CorrectionDataset but also returns soft labels where available.
    For rows without soft labels (identical src/tgt or teacher failure),
    the KL term is simply skipped — only CE applies.
    """
    def __init__(self, df: pd.DataFrame, tokenizer: KonkaniCharTokenizer,
                 soft_labels: dict, max_len: int,
                 augment_ratio: float = 0.3, noise_prob: float = 0.1,
                 is_train: bool = True):
        self.tokenizer   = tokenizer
        self.max_len     = max_len
        self.pairs       = []   # (src, tgt, soft_label_or_None)
        index_mapping    = soft_labels.get("_index_mapping", [])
        # Reverse map: original df index -> soft_label dict
        sl_map = {orig_idx: soft_labels.get(i)
                  for i, orig_idx in enumerate(index_mapping)}

        for df_idx, row in df.iterrows():
            src = str(row["hyp_greedy"]).strip()
            tgt = str(row["ref"]).strip()
            sl  = sl_map.get(df_idx)   # None for identical rows

            if src != tgt:
                self.pairs.append((src, tgt, sl))
            elif is_train and random.random() < augment_ratio:
                noisy = augment_with_noise(tgt, tokenizer.char2idx, noise_prob)
                if noisy != tgt:
                    self.pairs.append((noisy, tgt, None))  # no teacher signal for synthetic

        print(f"  Distillation dataset: {len(self.pairs)} pairs "
              f"({sum(1 for _,_,sl in self.pairs if sl is not None)} with soft labels)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt, sl = self.pairs[idx]
        ids, labels, mask = self.tokenizer.encode_pair(src, tgt, self.max_len)

        item = {
            "input_ids":      torch.tensor(ids,    dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask,   dtype=torch.long),
            "has_soft":       torch.tensor(sl is not None, dtype=torch.bool),
        }

        # Pack soft labels into fixed-size tensors
        if sl is not None:
            tgt_len = sl["topk_probs"].size(0)
            topk    = sl["topk_probs"].size(1)
            # Pad/truncate to max_len
            p = sl["topk_probs"].float()
            i = sl["topk_ids"]
            pad_rows = self.max_len - tgt_len
            if pad_rows > 0:
                p = torch.cat([p, torch.zeros(pad_rows, topk)], dim=0)
                i = torch.cat([i, torch.zeros(pad_rows, topk, dtype=torch.long)], dim=0)
            else:
                p = p[:self.max_len]
                i = i[:self.max_len]
            item["soft_probs"] = p[:self.max_len]   # (max_len, TOPK)
            item["soft_ids"]   = i[:self.max_len]   # (max_len, TOPK)
            item["soft_start"] = torch.tensor(sl["prompt_len"], dtype=torch.long)
        else:
            item["soft_probs"] = torch.zeros(self.max_len, 32)
            item["soft_ids"]   = torch.zeros(self.max_len, 32, dtype=torch.long)
            item["soft_start"] = torch.tensor(0, dtype=torch.long)

        return item


# =============================================================================
#  DISTILLATION LOSS
# =============================================================================
def distillation_loss(student_logits: torch.Tensor,
                      hard_labels:    torch.Tensor,
                      soft_probs:     torch.Tensor,
                      soft_ids:       torch.Tensor,
                      has_soft:       torch.Tensor,
                      soft_start:     torch.Tensor,
                      alpha: float, beta: float, temperature: float) -> torch.Tensor:
    """
    Combined loss:
      total = alpha * CE(student, hard_labels) + beta * KL(student_soft, teacher_soft)

    student_logits: (B, L, V)
    hard_labels:    (B, L)   — -100 for ignored positions
    soft_probs:     (B, L, K) — teacher's top-K probs
    soft_ids:       (B, L, K) — teacher's top-K token ids
    has_soft:       (B,)      — whether this sample has soft labels
    soft_start:     (B,)      — position where supervision starts
    """
    B, L, V = student_logits.shape

    # --- Hard CE loss (shift by 1) ---
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = hard_labels[:, 1:].contiguous()
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, V), shift_labels.view(-1), ignore_index=-100
    )

    # --- KL distillation loss ---
    kl_loss = torch.tensor(0.0, device=student_logits.device)
    n_soft  = has_soft.sum().item()

    if n_soft > 0 and beta > 0:
        # Only compute KL for samples that have soft labels
        soft_mask = has_soft.nonzero(as_tuple=True)[0]

        for b_idx in soft_mask:
            b = b_idx.item()
            start = soft_start[b].item()
            end   = min(start + soft_probs.size(1), L - 1)
            if end <= start:
                continue

            s_logits  = student_logits[b, start:end, :]   # (tgt_len, V)
            t_probs   = soft_probs[b, :end-start, :]       # (tgt_len, K)
            t_ids     = soft_ids[b,  :end-start, :]        # (tgt_len, K)

            # Build sparse teacher distribution over full vocab
            t_full = torch.zeros_like(s_logits)             # (tgt_len, V)
            t_full.scatter_(1, t_ids, t_probs)              # fill top-K slots

            # Student log-probs at temperature
            s_log_soft = F.log_softmax(s_logits / temperature, dim=-1)
            t_log      = torch.log(t_full + 1e-9)

            # KL(teacher || student) — teacher is "target"
            kl = F.kl_div(s_log_soft, t_full, reduction="batchmean")
            kl_loss = kl_loss + kl

        kl_loss = kl_loss / n_soft  # average over samples with soft labels

    total = alpha * ce_loss + beta * (temperature ** 2) * kl_loss
    return total, ce_loss.detach(), kl_loss.detach()


# =============================================================================
#  MAIN DISTILLATION TRAINING LOOP
# =============================================================================
def train():
    set_seed(CONFIG["seed"])
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Student device: {device}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # ---- Load tokenizer and data ----
    char_tokenizer = KonkaniCharTokenizer(CONFIG["vocab_path"])
    df = pd.read_csv(CONFIG["csv_path"]).dropna(subset=["hyp_greedy", "ref"])
    df["hyp_greedy"] = df["hyp_greedy"].astype(str).str.strip()
    df["ref"]        = df["ref"].astype(str).str.strip()
    print(f"Total rows: {len(df):,}")

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=CONFIG["seed"])
    train_df = train_df.reset_index(drop=True)  # keep original index for sl_map
    val_df   = val_df.reset_index(drop=True)

    # ---- Generate (or load cached) soft labels from teacher ----
    soft_label_path = CONFIG["soft_label_file"]
    os.makedirs(os.path.dirname(soft_label_path), exist_ok=True)

    if not os.path.exists(soft_label_path):
        teacher, sp_proc = load_teacher_model(
            CONFIG["teacher_path"], CONFIG["teacher_device"], CONFIG["teacher_dtype"]
        )
        soft_labels = generate_soft_labels(
            train_df, teacher, sp_proc, char_tokenizer,
            CONFIG["teacher_device"], CONFIG["max_len"],
            CONFIG["temperature"], soft_label_path
        )
        # Free teacher from GPU memory — student owns the GPU from here
        del teacher
        if CONFIG["teacher_device"] == "cuda":
            torch.cuda.empty_cache()
        print("Teacher unloaded. GPU memory freed for student training.")
    else:
        soft_labels = torch.load(soft_label_path)
        print(f"Loaded cached soft labels from {soft_label_path}")

    # ---- Build distillation datasets ----
    print("\nBuilding distillation datasets...")
    train_ds = DistillationDataset(train_df, char_tokenizer, soft_labels,
                                   CONFIG["max_len"], CONFIG["augment_ratio"],
                                   CONFIG["noise_prob"], is_train=True)
    # Val uses empty soft labels — just evaluate with CE + CER
    val_ds   = DistillationDataset(val_df, char_tokenizer, {},
                                   CONFIG["max_len"], augment_ratio=0.0,
                                   noise_prob=0.0, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=CONFIG["num_workers"],
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=CONFIG["num_workers"],
                              pin_memory=(device.type == "cuda"))

    # ---- Build student model ----
    student = TinyMambaCorrectorModel(
        vocab_size = VOCAB_SIZE,
        d_model    = CONFIG["d_model"],
        n_layers   = CONFIG["n_layers"],
        d_state    = CONFIG["d_state"],
        d_conv     = CONFIG["d_conv"],
        expand     = CONFIG["expand"],
        dropout    = CONFIG["dropout"],
    ).to(device)
    print(f"Student parameters: {student.count_params():,}")

    # ---- Optimizer + cosine scheduler ----
    optimizer = torch.optim.AdamW(student.parameters(),
                                  lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    total_steps  = len(train_loader) * CONFIG["epochs"] // CONFIG["grad_accum"]

    def lr_lambda(step):
        if step < CONFIG["warmup_steps"]:
            return step / max(CONFIG["warmup_steps"], 1)
        progress = (step - CONFIG["warmup_steps"]) / max(total_steps - CONFIG["warmup_steps"], 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ---- Training ----
    best_val_loss = float("inf")
    log_rows      = []
    log_path      = os.path.join(CONFIG["output_dir"], "distill_log.csv")

    print(f"\nStarting distillation training for {CONFIG['epochs']} epochs...\n")

    for epoch in range(1, CONFIG["epochs"] + 1):
        student.train()
        total_loss, ce_accum, kl_accum, n_steps = 0.0, 0.0, 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            ids        = batch["input_ids"].to(device)
            labels     = batch["labels"].to(device)
            soft_probs = batch["soft_probs"].to(device)
            soft_ids   = batch["soft_ids"].to(device)
            has_soft   = batch["has_soft"].to(device)
            soft_start = batch["soft_start"].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, _ = student(ids)   # ignore student's built-in CE here
                loss, ce, kl = distillation_loss(
                    logits, labels, soft_probs, soft_ids,
                    has_soft, soft_start,
                    CONFIG["alpha"], CONFIG["beta"], CONFIG["temperature"]
                )
                loss = loss / CONFIG["grad_accum"]

            scaler.scale(loss).backward()
            total_loss += loss.item() * CONFIG["grad_accum"]
            ce_accum   += ce.item()
            kl_accum   += kl.item()
            n_steps    += 1

            if (step + 1) % CONFIG["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), CONFIG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        train_loss = total_loss / max(n_steps, 1)
        val_loss   = evaluate(student, val_loader, device)
        cer        = quick_cer_eval(student, val_df, char_tokenizer, device, n=50)

        print(f"Epoch {epoch:3d}/{CONFIG['epochs']}  "
              f"total={train_loss:.4f}  CE={ce_accum/n_steps:.4f}  "
              f"KL={kl_accum/n_steps:.4f}  val={val_loss:.4f}  CER={cer:.3f}")

        log_rows.append({"epoch": epoch, "train_loss": train_loss,
                         "ce_loss": ce_accum/n_steps, "kl_loss": kl_accum/n_steps,
                         "val_loss": val_loss, "cer": cer})
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state": student.state_dict(),
                        "val_loss": val_loss, "cer": cer, "config": CONFIG},
                       os.path.join(CONFIG["output_dir"], "best_student.pt"))
            print(f"  → Saved best student (val_loss={val_loss:.4f})")

    print(f"\nDone. Best val_loss: {best_val_loss:.4f}")
    print(f"Logs: {log_path}")
    print(f"Best student: {CONFIG['output_dir']}/best_student.pt")
    print("\nThe saved student model is pure PyTorch — no mamba-ssm kernel needed at inference.")


if __name__ == "__main__":
    train()
