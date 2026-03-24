#!/usr/bin/env python3
"""
Deep Conformer-CTC Training Script (Optimized for RTX 3060 - Windows/CUDA)
Updated for Phase 2: Refinement (Total 200 Epochs)
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

# Import the character-based Conformer model
from models.conformer_ctc import create_model

# ─────────────────────────────────────────────────────────────
# CONFIGURATION (Optimized for 6GB VRAM)
# ─────────────────────────────────────────────────────────────
CONFIG = {
    'batch_size': 8,              # Doubled from 4 thanks to AMP!
    'grad_accum': 8,              # Effective batch size = 64
    'd_model': 256,               
    'num_layers': 12,             
    'lr': 3e-4,                   
    'epochs': 200,                # Train for another 100 epochs on the 110 hr data
    'max_audio_len': 16000 * 10,  
    'num_workers': 0,             
    'output_dir': 'outputs/conformer_ctc_run1',
    'checkpoint': 'outputs/conformer_ctc_run1/best_conformer_ctc.pt',
    'use_amp': True               # Mixed Precision ON
}

# GPU STABILITY & OPTIMIZATION (Windows/CUDA Optimized)
torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Allowing some room for the OS on a 6GB card
    torch.cuda.set_per_process_memory_fraction(0.85)

# ─────────────────────────────────────────────────────────────
# PATH REMAPPING (Handles Windows/Unix cross-OS paths)
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
    plt.style.use('bmh')
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Konkani ASR Metrics (Target: {CONFIG["epochs"]} Epochs)', fontsize=16, fontweight='bold')
    axs[0,0].plot(epochs, t_loss, label='Train Loss')
    if any(v > 0 for v in v_loss): axs[0,0].plot(epochs, v_loss, label='Val Loss', color='red')
    axs[0,0].set_title('Loss Curve'); axs[0,0].legend()
    axs[0,1].plot(epochs, wers, color='green'); axs[0,1].set_title('WER (%)')
    axs[1,0].plot(epochs, cers, color='orange'); axs[1,0].set_title('CER (%)')
    axs[1,1].plot(epochs, lrs, color='purple'); axs[1,1].set_title('LR Schedule'); axs[1,1].set_yscale('log')
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
    char2idx = vocab['char2idx']; idx2char = {idx: char for char, idx in char2idx.items()}
    vocab_size = len(char2idx)
    
    model = create_model(vocab_size=vocab_size, d_model=CONFIG['d_model'], num_layers=CONFIG['num_layers'])
    start_epoch = 0
    if os.path.exists(CONFIG['checkpoint']):
        print(f"Resuming from checkpoint: {CONFIG['checkpoint']}")
        checkpoint = torch.load(CONFIG['checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from Epoch {start_epoch}")
    model = model.to(device)

    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, n_fft=400, hop_length=160).to(device)
    train_ds = KonkaniCTCDataset('data/konkani-ultimate/train.json', vocab_path, CONFIG['max_audio_len'])
    val_ds = KonkaniCTCDataset('data/konkani-ultimate/val.json', vocab_path, CONFIG['max_audio_len'])
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=CONFIG['num_workers'])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-2)
    if start_epoch > 0 and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    steps_per_epoch = len(train_loader) // CONFIG['grad_accum']
    # OneCycleLR logic for the full 200 epochs
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG['lr'], 
                                              steps_per_epoch=steps_per_epoch, 
                                              epochs=CONFIG['epochs'], 
                                              last_epoch=(start_epoch * steps_per_epoch) - 1 if start_epoch > 0 else -1)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    stats_path = os.path.join(CONFIG['output_dir'], 'training_stats.csv')
    if not os.path.exists(stats_path):
        with open(stats_path, 'w', newline='') as f:
            writer = csv.writer(f); writer.writerow(['epoch', 'train_loss', 'val_loss', 'wer', 'cer', 'lr', 'timestamp'])

    best_val_loss = float('inf')
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=CONFIG.get('use_amp', True))
    
    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
        for i, batch in enumerate(pbar):
            audio, a_lens, target, t_lens = batch['audio'].to(device), batch['audio_lengths'].to(device), batch['text'].to(device), batch['text_lengths'].to(device)
            with torch.no_grad():
                mel = torch.log(mel_transform(audio).transpose(1, 2) + 1e-9)
                mel_lens = torch.clamp((a_lens // 160) + 1, max=mel.size(1))
            
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=CONFIG.get('use_amp', True)):
                logits, _ = model(mel, mel_lens)
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                loss = criterion(log_probs, target, mel_lens, t_lens) / CONFIG['grad_accum']
            
            scaler.scale(loss).backward()
            
            if (i + 1) % CONFIG['grad_accum'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
            total_train_loss += loss.item() * CONFIG['grad_accum']
            pbar.set_postfix(loss=f"{loss.item() * CONFIG['grad_accum']:.4f}")
            
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval(); total_val_loss = 0; all_preds, all_targets = [], []
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]")
            for batch in vbar:
                audio, a_lens, target, t_lens, t_strs = batch['audio'].to(device), batch['audio_lengths'].to(device), batch['text'].to(device), batch['text_lengths'].to(device), batch['text_strs']
                mel = torch.log(mel_transform(audio).transpose(1, 2) + 1e-9)
                mel_lens = torch.clamp((a_lens // 160) + 1, max=mel.size(1))
                logits, _ = model(mel, mel_lens)
                val_loss = criterion(F.log_softmax(logits, dim=-1).transpose(0, 1), target, mel_lens, t_lens)
                total_val_loss += val_loss.item()
                preds = greedy_decode(logits, mel_lens, idx2char)
                all_preds.extend(preds); all_targets.extend(t_strs)
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_wer = wer(all_targets, all_preds); avg_cer = cer(all_targets, all_preds)
        lr_now = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | WER: {avg_wer:.2%} | CER: {avg_cer:.2%}")
        with open(stats_path, 'a', newline='') as f:
            writer = csv.writer(f); writer.writerow([epoch+1, avg_train_loss, avg_val_loss, avg_wer, avg_cer, lr_now, datetime.now().strftime('%H:%M:%S')])
        save_plots(stats_path, str(CONFIG['output_dir']))
        
        if avg_wer < best_val_loss: # Reusing best_val_loss variable name but tracking WER instead
            best_val_loss = avg_wer
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_val_loss, 'wer': avg_wer, 'cer': avg_cer, 'vocab_size': vocab_size, 'config': CONFIG}, 
                       os.path.join(str(CONFIG['output_dir']), 'best_conformer_ctc.pt'))
            print(f"--> Saved New Best Model (WER: {avg_wer:.2%})")
            
        # Always save the latest epoch so progress isn't lost if stopped!
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_val_loss, 'wer': avg_wer, 'cer': avg_cer, 'vocab_size': vocab_size, 'config': CONFIG}, 
                   os.path.join(str(CONFIG['output_dir']), 'latest_conformer_ctc.pt'))

        # Save a separate model backup and graph snapshot every 10 epochs
        if (epoch + 1) % 10 == 0:
            import shutil
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_val_loss, 'wer': avg_wer, 'cer': avg_cer, 'vocab_size': vocab_size, 'config': CONFIG}, 
                       os.path.join(str(CONFIG['output_dir']), f'conformer_ctc_epoch_{epoch+1}.pt'))
            shutil.copy(os.path.join(str(CONFIG['output_dir']), 'training_progress.png'), 
                        os.path.join(str(CONFIG['output_dir']), f'training_progress_epoch_{epoch+1}.png'))
            print(f"--> Saved Milestone Backup for Epoch {epoch+1}")

if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    train()
