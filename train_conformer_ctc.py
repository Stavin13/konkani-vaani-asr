#!/usr/bin/env python3
"""
Deep Conformer-CTC Training Script (Optimized for RTX 3060)
This version includes the 4 mandatory ASR metrics dashboard.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import librosa
import json, argparse, os, sys, gc, logging, csv
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
from jiwer import wer, cer

# Import our new model
from models.conformer_ctc import create_model

# ─────────────────────────────────────────────────────────────
# CONFIGURATION (Optimized for 6GB VRAM)
# ─────────────────────────────────────────────────────────────
CONFIG = {
    'batch_size': 4,              
    'grad_accum': 8,              # Effective batch size = 32
    'd_model': 256,               
    'num_layers': 12,             
    'lr': 3e-4,                   
    'epochs': 50,
    'max_audio_len': 16000 * 10,  
    'num_workers': 0,             
    'output_dir': 'outputs/conformer_ctc_run1',
    'checkpoint': 'outputs/conformer_ctc_run1/best_conformer_ctc.pt'
}

# GPU STABILITY FLAGS
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.9)

# ─────────────────────────────────────────────────────────────
# PATH REMAPPING
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

def remap_path(unix_path: str) -> str:
    for prefix in ["/Volumes/data&proj/konkani/", "/Volumes/data&proj/konkani", "/Volumes/"]:
        if unix_path.startswith(prefix):
            rel = unix_path[len(prefix):]
            candidate = BASE_DIR / rel.replace("/", os.sep)
            if candidate.exists(): return str(candidate)
            parts = rel.split("/", 1)
            if len(parts) > 1:
                candidate2 = BASE_DIR / parts[1].replace("/", os.sep)
                if candidate2.exists(): return str(candidate2)
    if os.path.exists(unix_path): return unix_path
    fname = os.path.basename(unix_path)
    corpus_dir = BASE_DIR / "KonkaniRawSpeechCorpus"
    if corpus_dir.exists():
        for root, _, files in os.walk(corpus_dir):
            if fname in files: return os.path.join(root, fname)
    return ""

# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class KonkaniCTCDataset(Dataset):
    def __init__(self, manifest_path, vocab_path, max_audio_len):
        self.max_len = max_audio_len
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.char2idx = vocab['char2idx']
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.blank_idx = 0
        self.unk_idx = self.char2idx.get('<unk>', len(self.char2idx)-1)

        self.samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = json.loads(line)
                local_path = remap_path(s['audio_filepath'])
                if local_path:
                    s['audio_filepath'] = local_path
                    self.samples.append(s)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            audio, sr = librosa.load(s['audio_filepath'], sr=16000)
            audio = torch.FloatTensor(audio)
        except:
            audio = torch.zeros(16000)
        if len(audio) > self.max_len: audio = audio[:self.max_len]
        text_str = s.get('text', '')
        ids = [self.char2idx.get(c, self.unk_idx) for c in text_str]
        return {'audio': audio, 'text': torch.LongTensor(ids), 'text_str': text_str}

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x['audio']), reverse=True)
    audios = [b['audio'] for b in batch]
    texts = [b['text'] for b in batch]
    audio_lengths = torch.LongTensor([len(a) for a in audios])
    padded_audio = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
    text_lengths = torch.LongTensor([len(t) for t in texts])
    padded_text = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    return {
        'audio': padded_audio, 'audio_lengths': audio_lengths,
        'text': padded_text, 'text_lengths': text_lengths,
        'text_strs': [b['text_str'] for b in batch]
    }

# ─────────────────────────────────────────────────────────────
# DECODING & METRICS
# ─────────────────────────────────────────────────────────────
def greedy_decode(logits, mel_lens, idx2char):
    # logits: (B, T, V)
    preds = torch.argmax(logits, dim=-1) # (B, T)
    decoded_texts = []
    for i in range(preds.size(0)):
        p = preds[i, :mel_lens[i]]
        chars = []
        prev = -1
        for idx in p.tolist():
            if idx != prev and idx != 0: # 0 is blank
                chars.append(idx2char.get(idx, ''))
            prev = idx
        decoded_texts.append("".join(chars))
    return decoded_texts

def save_plots(stats_path, output_dir):
    epochs, t_loss, v_loss, wers, cers, lrs = [], [], [], [], [], []
    if not os.path.exists(stats_path): return
    
    with open(stats_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            t_loss.append(float(row['train_loss']))
            v_loss.append(float(row.get('val_loss', 0)))
            wers.append(float(row.get('wer', 1.0)))
            cers.append(float(row.get('cer', 1.0)))
            lrs.append(float(row['lr']))
            
    if not epochs: return

    # Using the style requested by user
    plt.style.use('bmh') # Clean look
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ASR Training Metrics - 50 Epochs', fontsize=16, fontweight='bold')

    # 1. Loss
    axs[0,0].plot(epochs, t_loss, label='Train Loss', marker='s', markersize=4)
    if any(v > 0 for v in v_loss):
        axs[0,0].plot(epochs, v_loss, label='Val Loss', color='red', marker='s', markersize=4)
    axs[0,0].set_title('Training and Validation Loss')
    axs[0,0].set_xlabel('Epoch')
    axs[0,0].set_ylabel('Loss')
    axs[0,0].legend()
    axs[0,0].grid(True, alpha=0.3)

    # 2. WER
    axs[0,1].plot(epochs, wers, color='green', marker='s', markersize=4)
    axs[0,1].set_title('Word Error Rate')
    axs[0,1].set_xlabel('Epoch')
    axs[0,1].set_ylabel('WER (%)')
    axs[0,1].set_ylim(0, 1.1)
    axs[0,1].grid(True, alpha=0.3)

    # 3. CER
    axs[1,0].plot(epochs, cers, color='orange', marker='s', markersize=4)
    axs[1,0].set_title('Character Error Rate')
    axs[1,0].set_xlabel('Epoch')
    axs[1,0].set_ylabel('CER (%)')
    axs[1,0].set_ylim(0, 1.1)
    axs[1,0].grid(True, alpha=0.3)

    # 4. Learning Rate
    axs[1,1].plot(epochs, lrs, color='purple', marker='s', markersize=4)
    axs[1,1].set_title('Learning Rate Schedule')
    axs[1,1].set_xlabel('Epoch')
    axs[1,1].set_ylabel('Learning Rate')
    axs[1,1].set_yscale('log') # Common to view LR in log scale
    axs[1,1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'training_progress.png'), dpi=150)
    plt.close()

# ─────────────────────────────────────────────────────────────
# TRAINING ENGINE
# ─────────────────────────────────────────────────────────────
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    vocab_path = 'data/konkani-10k/vocab.json'
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    char2idx = vocab['char2idx']
    idx2char = {idx: char for char, idx in char2idx.items()}
    vocab_size = len(char2idx)
    
    model = create_model(vocab_size=vocab_size, d_model=CONFIG['d_model'], num_layers=CONFIG['num_layers'])
    start_epoch = 0
    if os.path.exists(CONFIG['checkpoint']):
        print(f"Resuming from checkpoint: {CONFIG['checkpoint']}")
        checkpoint = torch.load(CONFIG['checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from Epoch {start_epoch}")
    model = model.to(device)

    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, n_fft=400, hop_length=160).to(device)
    
    train_ds = KonkaniCTCDataset('data/konkani-10k/train_manifest.json', vocab_path, CONFIG['max_audio_len'])
    val_ds = KonkaniCTCDataset('data/konkani-10k/val_manifest.json', vocab_path, CONFIG['max_audio_len'])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=CONFIG['num_workers'])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-2)
    if start_epoch > 0 and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for group in optimizer.param_groups: group.setdefault('initial_lr', CONFIG['lr'])
        except: pass

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    steps_per_epoch = len(train_loader) // CONFIG['grad_accum']
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG['lr'], steps_per_epoch=steps_per_epoch, epochs=CONFIG['epochs'], last_epoch=(start_epoch * steps_per_epoch) - 1 if start_epoch > 0 else -1)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    stats_path = os.path.join(CONFIG['output_dir'], 'training_stats.csv')
    if not os.path.exists(stats_path):
        with open(stats_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'wer', 'cer', 'lr', 'timestamp'])

    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, CONFIG['epochs']):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
        for i, batch in enumerate(pbar):
            audio, a_lens, target, t_lens = batch['audio'].to(device), batch['audio_lengths'].to(device), batch['text'].to(device), batch['text_lengths'].to(device)
            with torch.no_grad():
                mel = mel_transform(audio).transpose(1, 2)
                mel = torch.log(mel + 1e-9)
                mel_lens = (a_lens // 160) + 1
                mel_lens = torch.clamp(mel_lens, max=mel.size(1))
            
            logits, _ = model(mel, mel_lens)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = criterion(log_probs, target, mel_lens, t_lens) / CONFIG['grad_accum']
            loss.backward()
            
            if (i + 1) % CONFIG['grad_accum'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                if device.type == 'cuda': torch.cuda.synchronize()
                
            total_train_loss += loss.item() * CONFIG['grad_accum']
            pbar.set_postfix(loss=f"{loss.item() * CONFIG['grad_accum']:.4f}")
            
        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]")
            for batch in vbar:
                audio, a_lens, target, t_lens, t_strs = batch['audio'].to(device), batch['audio_lengths'].to(device), batch['text'].to(device), batch['text_lengths'].to(device), batch['text_strs']
                mel = mel_transform(audio).transpose(1, 2)
                mel = torch.log(mel + 1e-9)
                mel_lens = (a_lens // 160) + 1
                mel_lens = torch.clamp(mel_lens, max=mel.size(1))
                
                logits, _ = model(mel, mel_lens)
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                val_loss = criterion(log_probs, target, mel_lens, t_lens)
                total_val_loss += val_loss.item()
                
                # Metrics
                preds = greedy_decode(logits, mel_lens, idx2char)
                all_preds.extend(preds)
                all_targets.extend(t_strs)
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_wer = wer(all_targets, all_preds)
        avg_cer = cer(all_targets, all_preds)
        lr_now = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | WER: {avg_wer:.2%} | CER: {avg_cer:.2%}")
        
        # Log to CSV
        with open(stats_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, avg_wer, avg_cer, lr_now, datetime.now().strftime('%H:%M:%S')])
            
        save_plots(stats_path, CONFIG['output_dir'])
        
        # Save best model based on Val Loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(CONFIG['output_dir'], 'best_conformer_ctc.pt')
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss, 'wer': avg_wer, 'cer': avg_cer,
                'vocab_size': vocab_size, 'config': CONFIG
            }, ckpt_path)
            print(f"--> Saved New Best Model (Val Loss: {avg_val_loss:.4f})")

if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    train()
