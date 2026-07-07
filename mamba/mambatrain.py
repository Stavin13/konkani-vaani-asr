#!/usr/bin/env python3
# =============================================================================
# Full-run training script for Mamba-130M ASR post-correction
# Optimised for 20 GB VRAM (also runs perfectly on T4 16 GB)
# =============================================================================

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import math
import random
import warnings
import shutil
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
import sentencepiece as spm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
#  CONFIG – tuned for 20 GB VRAM (safe, fast, and robust)
# =============================================================================
CONFIG = {
    "csv_path": "./train_audit.csv",
    "model_id": "state-spaces/mamba-130m-hf",
    "vocab_size": 4000,
    "max_length": 512,           # up from 128 – better long‑range corrections
    "batch_size": 4,             # up from 1 – faster training
    "grad_accum": 4,             # effective batch = 4*4 = 16 (same as before)
    "epochs": 3,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "grad_clip": 1.0,
    "num_workers": 2,            # parallel data loading (uses CPU RAM)
    "tokenizer_train_rows": 30000,
    "seed": 42,
    "output_dir": "./outputs_fullrun",
    "prompt_prefix": "Correct this Konkani ASR text: ",
    "response_prefix": "\nCorrected: ",
}

# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_env_summary():
    print("=" * 60)
    print("Environment summary")
    print("=" * 60)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU count:", torch.cuda.device_count())
        print("bf16 supported:", torch.cuda.is_bf16_supported())
        # Print initial VRAM usage
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"Initial VRAM allocated: {allocated:.2f} GB")
        print(f"Initial VRAM reserved:  {reserved:.2f} GB")
    print()

def print_vram_usage(stage=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"[VRAM {stage}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

# =============================================================================
#  TOKENIZER TRAINING & WRAPPER
# =============================================================================
def train_tokenizer(df_for_tokenizer, save_path):
    texts = df_for_tokenizer["hyp_greedy"].tolist() + df_for_tokenizer["ref"].tolist()
    with open("konkani_corpus.txt", "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
    spm.SentencePieceTrainer.train(
        input="konkani_corpus.txt",
        model_prefix=save_path,
        vocab_size=CONFIG["vocab_size"],
        character_coverage=0.9995,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols="।,॥",
        byte_fallback=True,
    )
    os.remove("konkani_corpus.txt")

class KonkaniTokenizer:
    def __init__(self, tokenizer_dir):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(os.path.join(tokenizer_dir, "konkani_spm.model"))
        self.pad_token_id = 0
        self.eos_token_id = 3
        self.bos_token_id = 2

    @property
    def vocab_size(self):
        return self.sp.GetPieceSize()

    def __call__(self, text, truncation=True, max_length=128, padding="max_length"):
        ids = self.sp.EncodeAsIds(text)
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        attention_mask = [1] * len(ids)
        if padding == "max_length":
            pad_len = max_length - len(ids)
            ids += [self.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
        return {"input_ids": ids, "attention_mask": attention_mask}

# =============================================================================
#  DATASET
# =============================================================================
class KonkaniCorrectionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._enc = self._preprocess()

    def _preprocess(self):
        ids_list, lbl_list, mask_list = [], [], []
        for _, row in self.data.iterrows():
            prompt = CONFIG["prompt_prefix"] + row["hyp_greedy"] + CONFIG["response_prefix"]
            full_text = prompt + row["ref"] + "\n"
            full_enc = self.tokenizer(
                full_text, truncation=True, max_length=self.max_length, padding="max_length"
            )
            prompt_enc = self.tokenizer(
                prompt, truncation=True, max_length=self.max_length, padding=False
            )
            prompt_len = min(len(prompt_enc["input_ids"]), self.max_length)
            ids = full_enc["input_ids"]
            mask = full_enc["attention_mask"]
            labels = [-100] * prompt_len + ids[prompt_len:]
            labels = [
                token if (token != self.tokenizer.pad_token_id and m == 1) else -100
                for token, m in zip(labels, mask)
            ]
            labels = (labels + [-100] * self.max_length)[: self.max_length]
            ids_list.append(ids)
            lbl_list.append(labels)
            mask_list.append(mask)
        return {
            "input_ids": torch.tensor(ids_list, dtype=torch.long),
            "labels": torch.tensor(lbl_list, dtype=torch.long),
            "attention_mask": torch.tensor(mask_list, dtype=torch.long),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self._enc.items()}

# =============================================================================
#  LIGHTNING MODULE
# =============================================================================
class MambaKonkaniModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.resize_token_embeddings(config["vocab_size"])

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["in_proj", "x_proj", "dt_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        out = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        self.log("train_loss", out.loss, on_step=True, on_epoch=True, prog_bar=True)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        self.log("val_loss", out.loss, on_step=False, on_epoch=True, prog_bar=True)
        return out.loss

    def configure_optimizers(self):
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(self.config["warmup_ratio"] * total_steps)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

# =============================================================================
#  MAIN
# =============================================================================
def main():
    global TOKENIZER_DIR
    output_dir = CONFIG["output_dir"]
    TOKENIZER_DIR = os.path.join(output_dir, "konkani_tokenizer")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    final_dir = os.path.join(output_dir, "final_model")
    logs_dir = os.path.join(output_dir, "logs")

    for path in [output_dir, TOKENIZER_DIR, checkpoint_dir, final_dir, logs_dir]:
        os.makedirs(path, exist_ok=True)

    set_seed(CONFIG["seed"])
    print_env_summary()

    # ---- Load data ----
    print("Loading CSV from:", CONFIG["csv_path"])
    df = pd.read_csv(CONFIG["csv_path"]).dropna(subset=["hyp_greedy", "ref"]).reset_index(drop=True)
    print("Total rows after cleaning:", f"{len(df):,}")

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=CONFIG["seed"])
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,}")

    # ---- Tokenizer ----
    tokenizer_path = os.path.join(TOKENIZER_DIR, "konkani_spm.model")
    if not os.path.exists(tokenizer_path):
        tokenizer_rows = min(CONFIG["tokenizer_train_rows"], len(train_df))
        tokenizer_df = train_df.sample(tokenizer_rows, random_state=CONFIG["seed"]).reset_index(drop=True)
        print(f"Training tokenizer on {tokenizer_rows:,} sampled rows...")
        # Pass the full path without extension to train_tokenizer
        train_tokenizer(tokenizer_df, os.path.join(TOKENIZER_DIR, "konkani_spm"))
    else:
        print("Tokenizer already exists. Reusing it.")

    tokenizer = KonkaniTokenizer(TOKENIZER_DIR)
    CONFIG["vocab_size"] = tokenizer.vocab_size
    print("Tokenizer vocab size:", tokenizer.vocab_size)

    # ---- Datasets & DataLoaders ----
    print("\nEncoding datasets into tensors...")
    train_dataset = KonkaniCorrectionDataset(train_df, tokenizer, CONFIG["max_length"])
    val_dataset = KonkaniCorrectionDataset(val_df, tokenizer, CONFIG["max_length"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if CONFIG["num_workers"] > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if CONFIG["num_workers"] > 0 else None,
    )

    # ---- Training statistics ----
    steps_per_epoch = math.ceil(len(train_loader) / CONFIG["grad_accum"])
    print("\nRun settings")
    print("max_length:", CONFIG["max_length"])
    print("batch_size:", CONFIG["batch_size"])
    print("grad_accum:", CONFIG["grad_accum"])
    print("effective batch size:", CONFIG["batch_size"] * CONFIG["grad_accum"])
    print("epochs:", CONFIG["epochs"])
    print("optimizer steps/epoch:", steps_per_epoch)

    # ---- Model ----
    module = MambaKonkaniModule(CONFIG)
    print_vram_usage("after model load")

    # ---- Callbacks & Logger ----
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="mamba-fullrun-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop = pl.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min", verbose=True)
    csv_logger = pl.loggers.CSVLogger(logs_dir, name="fullrun")

    # ---- Trainer ----
    trainer = pl.Trainer(
        max_epochs=CONFIG["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        strategy="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        accumulate_grad_batches=CONFIG["grad_accum"],
        gradient_clip_val=CONFIG["grad_clip"],
        logger=csv_logger,
        callbacks=[checkpoint_cb, early_stop],
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    # ---- Train ----
    print("\nStarting full training run...")
    trainer.fit(module, train_loader, val_loader)

    # ---- Save best model ----
    if checkpoint_cb.best_model_path:
        print("\nBest checkpoint:", checkpoint_cb.best_model_path)
        best_module = MambaKonkaniModule.load_from_checkpoint(
            checkpoint_cb.best_model_path,
            config=CONFIG,
        )
        best_module.model = best_module.model.merge_and_unload()
        best_module.model.save_pretrained(final_dir)
        shutil.copy(tokenizer_path, final_dir)
        with open(os.path.join(final_dir, "config.json"), "w") as f:
            json.dump(CONFIG, f, indent=2)
        print("Saved merged model to:", final_dir)

    print("\nFull run finished.")

if __name__ == "__main__":
    # Optional: improve performance on T4/Ampere GPUs
    torch.set_float32_matmul_precision('high')
    main()