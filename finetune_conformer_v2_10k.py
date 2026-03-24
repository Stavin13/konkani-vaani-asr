import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
import random
from tqdm import tqdm
from jiwer import wer, cer
from typing import List, Dict, Tuple
from models.conformer_ctc_v2 import create_model_v2
from train_conformer_v2 import Tokenizer, KonkaniDataset, collate_fn, compute_mel, build_mel_transform

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
CONFIG = {
    'train_manifest': 'data/konkani-10k/train_manifest.json',
    'val_manifest': 'data/konkani-10k/val_manifest.json',
    'bpe_vocab': 'data/bpe_tokenizer/bpe_vocab.json',
    'bpe_sp_model': 'data/bpe_tokenizer/konkani_bpe.model',
    'char_vocab': 'data/konkani-10k/vocab.json',
    
    'base_checkpoint': 'outputs/conformer_v2_200ep/best_model.pt',
    'output_dir': 'outputs/conformer_v2_finetune_10k',
    
    'batch_size': 3, # Lower for RTX 3060 6GB stability
    'accum_steps': 20, 
    'epochs': 30,
    'learning_rate': 2e-5, # Very low for fine-tuning
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    
    'sampling_rate': 16000,
    'max_duration': 15.0,
}

# ─────────────────────────────────────────────────────────────
# CUSTOM DATASET FOR 10K PATHS
# ─────────────────────────────────────────────────────────────
class FineTuneDataset(KonkaniDataset):
    def load_samples(self, manifest_path):
        samples = []
        # Path correction for 10k dataset
        old_prefix = "/Volumes/data&proj/konkani/KonkaniRawSpeechCorpus/"
        new_prefix = "E:/konkani/KonkaniRawSpeechCorpus/"
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                path = item['audio_filepath'].replace(old_prefix, new_prefix)
                path = path.replace("\\", "/") # Normalize to forward slashes
                
                if not os.path.exists(path):
                    continue
                    
                samples.append({
                    'path': path,
                    'text': item['text']
                })
        return samples

# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def finetune():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Fine-Tuning Conformer v2 | Mode: 10K Polishing")
    print(f"  Device: {device} | Base: {CONFIG['base_checkpoint']}")
    print(f"{'='*60}")
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    tokenizer = Tokenizer(CONFIG['bpe_vocab'], CONFIG['bpe_sp_model'], CONFIG['char_vocab'])
    mel_transform = build_mel_transform(device)
    
    # 1. Initialize Model
    model = create_model_v2(vocab_size=tokenizer.vocab_size)
    
    # 2. Load Base Checkpoint
    print(f"Loading base model from {CONFIG['base_checkpoint']}...")
    ckpt = torch.load(CONFIG['base_checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    
    # 3. Data Loaders
    train_ds = FineTuneDataset(CONFIG['train_manifest'], tokenizer, int(CONFIG['sampling_rate']*CONFIG['max_duration']), augment=True)
    val_ds   = FineTuneDataset(CONFIG['val_manifest'], tokenizer, int(CONFIG['sampling_rate']*CONFIG['max_duration']), augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples  : {len(val_ds)}")
    
    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=CONFIG['min_lr'])
    scaler = torch.amp.GradScaler('cuda')
    
    # 5. Fine-Tuning Loop
    best_wer = 100.0
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        for batch_idx, batch in enumerate(pbar):
            audio = batch['audio'].to(device)
            audio_lens = batch['audio_lens'].to(device)
            labels = batch['text'].to(device)
            label_lens = batch['text_lens'].to(device)
            
            with torch.amp.autocast('cuda'):
                mel, mel_lens = compute_mel(audio, audio_lens, mel_transform)
                logits, out_lens = model(mel, mel_lens)
                
                # CTC Loss expects: (T, N, C)
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                loss = F.ctc_loss(log_probs, labels, out_lens, label_lens, blank=tokenizer.blank_id, zero_infinity=True)
                loss = loss / CONFIG['accum_steps']
            
            if not torch.isnan(loss):
                scaler.scale(loss).backward()
                train_loss += loss.item() * CONFIG['accum_steps']
            
            if (batch_idx + 1) % CONFIG['accum_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            pbar.set_postfix({'loss': f"{loss.item()*CONFIG['accum_steps']:.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
            
        scheduler.step()
        
        # Validation
        val_wer = evaluate(model, tokenizer, val_loader, mel_transform, device)
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch} summary: Loss={avg_loss:.4f}, Val WER={val_wer:.2%}")
        
        # Save checkpoints
        save_path = os.path.join(CONFIG['output_dir'], f"latest_checkpoint.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'wer': val_wer,
            'config': CONFIG
        }, save_path)
        
        if val_wer < best_wer:
            best_wer = val_wer
            best_path = os.path.join(CONFIG['output_dir'], f"best_model_ft.pt")
            torch.save({'model_state_dict': model.state_dict(), 'wer': best_wer}, best_path)
            print(f"  *** New Best WER: {best_wer:.2%} ***")

@torch.no_grad()
def evaluate(model, tokenizer, loader, mel_transform, device):
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        audio = batch['audio'].to(device)
        audio_lens = batch['audio_lens'].to(device)
        t_strs = batch['text_strs']
        
        mel, mel_lens = compute_mel(audio, audio_lens, mel_transform)
        logits, out_lens = model(mel, mel_lens)
        
        preds_ids = torch.argmax(logits, dim=-1)
        for i in range(preds_ids.size(0)):
            ids = preds_ids[i, :out_lens[i]].tolist()
            decoded = tokenizer.decode_ctc(ids)
            all_preds.append(decoded)
        all_targets.extend(t_strs)
        
    return wer(all_targets, all_preds)

if __name__ == '__main__':
    finetune()
