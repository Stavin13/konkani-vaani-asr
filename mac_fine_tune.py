#!/usr/bin/env python3
"""
🚀 Mac M4 Optimized Fine-tuning Script (Konkani ASR)
Tailored for 16GB Unified Memory and MPS acceleration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import librosa
import json, os, csv, time
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from jiwer import wer, cer
import numpy as np

# Import project modules
from models.conformer_ctc import create_model
from data.audio_processing.audio_processor import AudioProcessor
from data.audio_processing.text_tokenizer import KonkaniTokenizer

# ─────────────────────────────────────────────────────────────
# 🍏 MAC M4 OPTIMIZED CONFIGURATION
# ─────────────────────────────────────────────────────────────
CONFIG = {
    'batch_size': 4,              # Balanced for 16GB Unified Memory
    'grad_accum': 8,              # Effective batch size = 32
    'd_model': 256,
    'num_layers': 12,
    'lr': 5e-5,                   # Conservative for fine-tuning
    'max_audio_len': 16000 * 12,  # 12 seconds
    'num_workers': 0,             # Best for MPS
    'output_root': 'outputs/mac_finetune',
    'checkpoint': 'best_model.pt', # Will fallback to run1 if not found
    'vocab': 'data/konkani-10k/vocab.json'
}

# ─────────────────────────────────────────────────────────────
# MPS DEVICE SETUP
# ─────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("🍏 Optimized for Apple Silicon (MPS detected)")
else:
    device = torch.device('cpu')
    print("⚠️  Warning: MPS not available, falling back to CPU")

# Extra stability for Mac
torch.set_num_threads(8) # Use all M4 performance cores

BASE_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────
def find_audio_path(rel_path: str) -> str:
    """Intelligently map manifest paths to local filesystem"""
    # 1. Standardize path separators (Normalize to forward slashes first)
    normalized_rel = rel_path.replace("\\", "/")
    
    # 2. Try direct absolute path
    if os.path.exists(normalized_rel): return normalized_rel
    
    # 3. Clean common prefixes
    clean_p = normalized_rel.replace("/Volumes/data&proj/konkani/KonkaniRawSpeechCorpus/", "")
    clean_p = clean_p.replace("/Volumes/data&proj/konkani/", "")
    clean_p = clean_p.replace("Data/", "", 1) # Sometimes paths have extra 'Data/'
    
    # 4. Try variants relative to project root / KonkaniRawSpeechCorpus
    paths_to_try = [
        BASE_DIR / normalized_rel,
        BASE_DIR / "KonkaniRawSpeechCorpus" / normalized_rel,
        BASE_DIR / "KonkaniRawSpeechCorpus" / clean_p,
        BASE_DIR / "KonkaniRawSpeechCorpus" / "Data" / clean_p,
        BASE_DIR / "data" / "audio" / clean_p
    ]
    
    for p in paths_to_try:
        # Final check with local OS separator
        if p.exists(): return str(p)
    return ""

def clean_konkani_text(text: str) -> str:
    """Extract actual transcript if the text is in LDC-IL metadata format"""
    if "RECORDED TEXT ::" in text:
        try:
            # Extract content between 'RECORDED TEXT ::' and any subsequent field
            parts = text.split("RECORDED TEXT ::")
            if len(parts) > 1:
                subcontent = parts[1].strip()
                # Split at next metadata marker if exists
                for marker in ["TEXT TRANSLITERATION ::", "DIALECT ::", "RECORDING DATE ::"]:
                    subcontent = subcontent.split(marker)[0].strip()
                return subcontent.strip()
        except: pass
    return text.strip()

class KonkaniMacDataset(Dataset):
    def __init__(self, manifest_path, vocab_path, max_audio_len):
        self.max_len = max_audio_len
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.char2idx = vocab['char2idx']
        self.resamplers = {} # Cache for resamplers
        
        self.samples = []
        if not os.path.exists(manifest_path):
            print(f"❌ Manifest not found: {manifest_path}")
            return
            
        print(f"📂 Loading manifest: {manifest_path}")
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                try:
                    s = json.loads(line)
                    # 1. Clean Text
                    s['text'] = clean_konkani_text(s.get('text', ''))
                    
                    # 2. Remap path
                    p = find_audio_path(s['audio_filepath'])
                    if p:
                        s['audio_filepath'] = p
                        self.samples.append(s)
                    else:
                        if i < 1: # Print the first failure in detail
                            print(f"DEBUG: Failed to map path '{s['audio_filepath']}'")
                            print(f"DEBUG: BASE_DIR is '{BASE_DIR}'")
                            # Explicitly check if it exists at the most likely place
                            norm = s['audio_filepath'].replace("\\", "/")
                            maybe = BASE_DIR / "KonkaniRawSpeechCorpus" / norm
                            print(f"DEBUG: Check exists({maybe}) -> {maybe.exists()}")
                except Exception as e:
                    if i < 1: print(f"❌ Error at line {i}: {e}")
                    continue
        
        if len(self.samples) == 0:
            raise RuntimeError(f"❌ ERROR: Loaded 0 valid samples from {manifest_path}. Please check your KonkaniRawSpeechCorpus folder.")
        
        print(f"📊 Dataset Loaded: {len(self.samples)} valid samples found.")

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        wav, sr = torchaudio.load(s['audio_filepath'])
        # 1. Mono conversion
        if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
        # 2. Cached Resampling
        if sr != 16000:
             if sr not in self.resamplers:
                 self.resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
             wav = self.resamplers[sr](wav)
        
        audio = wav.squeeze(0)
        if len(audio) > self.max_len: audio = audio[:self.max_len]
        
        # 3. Label mapping
        ids = [self.char2idx.get(c, 0) for c in s.get('text', '')]
        return {'audio': audio, 'text': torch.LongTensor(ids), 'text_str': s.get('text', '')}

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x['audio']), reverse=True)
    lengths = torch.LongTensor([len(b['audio']) for b in batch])
    padded_audio = torch.nn.utils.rnn.pad_sequence([b['audio'] for b in batch], batch_first=True)
    t_lengths = torch.LongTensor([len(b['text']) for b in batch])
    padded_text = torch.nn.utils.rnn.pad_sequence([b['text'] for b in batch], batch_first=True, padding_value=0)
    return {
        'audio': padded_audio, 'audio_lengths': lengths,
        'text': padded_text, 'text_lengths': t_lengths,
        'text_strs': [b['text_str'] for b in batch]
    }

def decode_greedy(logits, mel_lens, tokenizer, blank_id=0):
    """Consistent greedy decoding targeting blank ID from model training"""
    preds = torch.argmax(logits, dim=-1)
    decoded = []
    pad_id = tokenizer.pad_id
    
    for i in range(preds.size(0)):
        p = preds[i, :mel_lens[i]].tolist()
        tokens = []
        prev = -1
        for idx in p:
            # Skip if same as prev, or if it's the blank/pad token
            if idx != prev and idx != blank_id and idx != pad_id:
                tokens.append(idx)
            prev = idx
        decoded.append(tokenizer.decode(tokens))
    return decoded

def save_plots(stats_path, output_dir):
    epochs, t_loss, v_loss, wers, cers = [], [], [], [], []
    if not os.path.exists(stats_path): return
    with open(stats_path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            epochs.append(int(r['epoch']))
            t_loss.append(float(r['train_loss']))
            v_loss.append(float(r['val_loss']))
            wers.append(float(r['wer']) * 100)
            cers.append(float(r['cer']) * 100)
            
    if not epochs: return
    plt.style.use('bmh')
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🍏 ASR Training Dashboard (Mac M4 Optimized)', fontsize=16, fontweight='bold')
    
    # Loss
    axs[0,0].plot(epochs, t_loss, label='Train', marker='o', markersize=4)
    axs[0,0].plot(epochs, v_loss, label='Val', color='red', marker='o', markersize=4)
    axs[0,0].set_title('Loss History'); axs[0,0].legend(); axs[0,0].grid(True, alpha=0.3)
    
    # WER
    axs[0,1].plot(epochs, wers, color='green', marker='s', markersize=4)
    axs[0,1].set_title('Word Error Rate (WER) %')
    axs[0,1].set_ylabel('%'); axs[0,1].grid(True, alpha=0.3)
    
    # CER
    axs[1,0].plot(epochs, cers, color='orange', marker='s', markersize=4)
    axs[1,0].set_title('Character Error Rate (CER) %')
    axs[1,0].set_ylabel('%'); axs[1,0].grid(True, alpha=0.3)
    
    # Learning Rate (from scheduler logic - we'll visualize progress)
    axs[1,1].text(0.5, 0.5, f"Latest CER: {cers[-1]:.2f}%\nTarget <20%", 
                  ha='center', va='center', fontsize=20, color='blue', alpha=0.6)
    axs[1,1].set_title('Status')
    axs[1,1].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'training_dashboard.png'), dpi=150)
    plt.close()

# ─────────────────────────────────────────────────────────────
# TRAINING ENGINE
# ─────────────────────────────────────────────────────────────
def run_training(manifest_path, output_tag, epochs=10):
    start_time = time.time()
    output_dir = os.path.join(CONFIG['output_root'], output_tag)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Setup Processors
    tokenizer = KonkaniTokenizer(CONFIG['vocab'])
    # CRITICAL: run1 was trained with blank=0. 
    # Use local blank_id variable for decoder/loss
    model_blank_id = 0 
    
    processor = AudioProcessor()
    
    # 2. Setup Model
    model = create_model(vocab_size=tokenizer.vocab_size, d_model=CONFIG['d_model'], num_layers=CONFIG['num_layers'])
    
    # 3. Intelligent Resume
    baseline = 'outputs/conformer_ctc_run1/best_conformer_ctc.pt'
    resume_p = CONFIG['checkpoint'] if os.path.exists(CONFIG['checkpoint']) else baseline
    
    if os.path.exists(resume_p):
        print(f"🔄 Resuming from: {resume_p}")
        checkpoint = torch.load(resume_p, map_location='cpu')
        sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Detect architecture to prevent size mismatch
        if 'encoder.layers.0.ff1.1.weight' in sd:
            d_model = sd['encoder.layers.0.ff1.1.weight'].shape[1]
            if d_model != CONFIG['d_model']: 
                print(f"🧩 Auto-adjusting d_model to {d_model}"); CONFIG['d_model'] = d_model
        
        # Adjust vocab if needed
        v_size = sd['ctc_head.weight'].shape[0]
        if v_size != tokenizer.vocab_size:
             model = create_model(vocab_size=v_size, d_model=CONFIG['d_model'], num_layers=CONFIG['num_layers'])
        
        msg = model.load_state_dict(sd, strict=False)
        print(f"✅ Weights loaded from {resume_p}")
        print(f"🔍 Missing: {len(msg.missing_keys)} | Unexpected: {len(msg.unexpected_keys)}")
        if len(msg.missing_keys) > 0:
             print(f"🔍 Sample Missing: {msg.missing_keys[:3]}")
    
    model = model.to(device)
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, n_fft=400, hop_length=160).to(device)
    
    train_ds = KonkaniMacDataset(manifest_path, CONFIG['vocab'], CONFIG['max_audio_len'])
    val_ds = KonkaniMacDataset('data/konkani-10k/val_manifest.json', CONFIG['vocab'], CONFIG['max_audio_len'])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=2, persistent_workers=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-3)
    # Using model_blank_id which is 0 for run1
    criterion = nn.CTCLoss(blank=model_blank_id, zero_infinity=True) 
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG['lr'], steps_per_epoch=len(train_loader)//CONFIG['grad_accum'], epochs=epochs)
    
    stats_path = os.path.join(output_dir, 'stats.csv')
    best_cer = 1.0
    
    # Initialize stats file with header
    with open(stats_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'wer', 'cer'])
    
    print(f"\n🚀 Phase {output_tag} started on M4.")

    for epoch in range(epochs):
        model.train(); total_loss = 0; optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, batch in enumerate(pbar):
            audio, a_lens = batch['audio'].to(device), batch['audio_lengths'].to(device)
            target, t_lens = batch['text'].to(device), batch['text_lengths'].to(device)
            
            with torch.no_grad():
                mel = mel_transform(audio).transpose(1, 2)
                mel = torch.log(mel + 1e-9)
                mel_lens = (a_lens // 160) + 1
                
                if i == 0 and epoch == 0:
                     print(f"📊 Mel stats: Mean={mel.mean():.4f}, Std={mel.std():.4f}, Shape={mel.shape}")
                     print(f"📊 Target stats: MaxID={target.max()}, MinID={target.min()}")
            
            logits, _ = model(mel, mel_lens)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            # Move to CPU for CTC Loss (MPS doesn't support CTCLoss yet)
            loss = criterion(log_probs.cpu(), target.cpu(), mel_lens.cpu(), t_lens.cpu())
            loss = loss.to(device)
            
            # Use local variable for loss calculation
            (loss / CONFIG['grad_accum']).backward()
            
            if (i + 1) % CONFIG['grad_accum'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Validation Phase
        model.eval(); v_loss = 0; v_cer = []; v_wer = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                audio, a_lens = batch['audio'].to(device), batch['audio_lengths'].to(device)
                target, t_lens, t_strs = batch['text'].to(device), batch['text_lengths'].to(device), batch['text_strs']
                
                mel = mel_transform(audio).transpose(1, 2)
                mel = torch.log(mel + 1e-9)
                mel_lens = (a_lens // 160) + 1
                
                logits, _ = model(mel, mel_lens)
                # Move to CPU for validation loss
                v_l = criterion(F.log_softmax(logits, dim=-1).transpose(0, 1).cpu(), target.cpu(), mel_lens.cpu(), t_lens.cpu())
                v_loss += v_l.item()
                preds = decode_greedy(logits, mel_lens, tokenizer, blank_id=model_blank_id)
                v_cer.extend([cer(r, p) for r, p in zip(t_strs, preds)])
                v_wer.extend([wer(r, p) for r, p in zip(t_strs, preds)])
        
        avg_v_loss = v_loss / len(val_loader)
        avg_wer = np.mean(v_wer)
        avg_cer = np.mean(v_cer)
        print(f"🏁 Epoch {epoch+1}: CER: {avg_cer:.2%} | WER: {avg_wer:.2%} | V_Loss: {avg_v_loss:.4f}")
        
        # Sample debug output
        print(f"👉 Target: {t_strs[0]}")
        print(f"👉 Pred  : {preds[0]}")
        
        # Save Stats
        with open(stats_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, total_loss/len(train_loader), avg_v_loss, avg_wer, avg_cer])
            
        # Draw Dashboard
        save_plots(stats_path, output_dir)
            
        if avg_cer < best_cer:
            best_cer = avg_cer
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'cer': avg_cer,
                'wer': avg_wer
            }, os.path.join(output_dir, 'best_model.pt'))
            print("💾 Saved New Best Model!")

    print(f"\n✅ Phase {output_tag} Complete in {(time.time() - start_time)/60:.1f} mins.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, choices=['chunk6', 'mega'], default='chunk6')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    manifest = 'data/kaggle_manifests/kaggle_manifest_chunk_6.json' if args.phase == 'chunk6' else 'data/konkani-mega-dataset/manifests/train.json'
    run_training(manifest, args.phase, epochs=args.epochs)
