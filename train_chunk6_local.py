#!/usr/bin/env python3
"""
Stage 2: Local Fine-tuning on 6GB Chunk
This script handles Phase 2 (Chunk 6) with extreme VRAM stability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import librosa
import json, os, csv
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from jiwer import wer, cer

from models.conformer_ctc import create_model

# ─────────────────────────────────────────────────────────────
# STAGE 2 CONFIGURATION (Ironclad for 6GB VRAM)
# ─────────────────────────────────────────────────────────────
CONFIG = {
    'batch_size': 1,              # Batch 1 is safest for 6GB VRAM with long audio
    'grad_accum': 32,             # Effective batch size = 32
    'd_model': 256,
    'num_layers': 12,
    'lr': 1e-4,
    'epochs': 20,
    'max_audio_len': 16000 * 15,  # Ignore anything > 15s to prevent crash
    'num_workers': 0,
    'output_dir': 'outputs/conformer_ctc_chunk6',
    'prev_checkpoint': 'outputs/conformer_ctc_run1/best_conformer_ctc.pt',
    'manifest': 'data/kaggle_manifests/kaggle_manifest_chunk_6.json',
    'vocab': 'data/konkani-10k/vocab.json'
}

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.enabled = False 

BASE_DIR = Path(__file__).resolve().parent

# ... (Helper Functions) ...
def remap_chunk_path(rel_path: str) -> str:
    full_path = BASE_DIR / "KonkaniRawSpeechCorpus" / rel_path.replace("/", os.sep)
    if full_path.exists(): return str(full_path)
    return ""

class KonkaniCTCDataset(Dataset):
    def __init__(self, manifest_path, vocab_path, max_audio_len):
        self.max_len = max_audio_len
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.char2idx = vocab['char2idx']
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = json.loads(line)
                local_p = remap_chunk_path(s['audio_filepath'])
                if local_p:
                    s['audio_filepath'] = local_p
                    self.samples.append(s)
        print(f"Loaded {len(self.samples)} samples for Stage 2 training.")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            audio, sr = librosa.load(s['audio_filepath'], sr=16000)
            audio = torch.FloatTensor(audio)
        except: audio = torch.zeros(16000)
        if len(audio) > self.max_len: audio = audio[:self.max_len]
        ids = [self.char2idx.get(c, 0) for c in s['text']]
        return {'audio': audio, 'text': torch.LongTensor(ids), 'text_str': s['text']}

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x['audio']), reverse=True)
    audios = [b['audio'] for b in batch]; texts = [b['text'] for b in batch]
    a_lens = torch.LongTensor([len(a) for a in audios]); padded_a = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
    t_lens = torch.LongTensor([len(t) for t in texts]); padded_t = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    return {'audio': padded_a, 'audio_lengths': a_lens, 'text': padded_t, 'text_lengths': t_lens, 'text_strs': [b['text_str'] for b in batch]}

def greedy_decode(logits, mel_lens, idx2char):
    preds = torch.argmax(logits, dim=-1)
    decoded = []
    for i in range(preds.size(0)):
        p = preds[i, :mel_lens[i]]; chars = []; prev = -1
        for idx in p.tolist():
            if idx != prev and idx != 0: chars.append(idx2char.get(idx, ''))
            prev = idx
        decoded.append("".join(chars))
    return decoded

def save_plots(stats_path, output_dir):
    epochs, t_loss, v_loss, wers, cers, lrs = [], [], [], [], [], []
    if not os.path.exists(stats_path): return
    with open(stats_path, 'r') as f:
        reader = csv.DictReader(f); [ (epochs.append(int(r['epoch'])), t_loss.append(float(r['train_loss'])), v_loss.append(float(r.get('val_loss',0))), wers.append(float(r.get('wer',1))), cers.append(float(r.get('cer',1))), lrs.append(float(r.get('lr',0)))) for r in reader]
    if not epochs: return
    plt.style.use('bmh')
    fig, axs = plt.subplots(2, 2, figsize=(15, 12)); fig.suptitle('Stage 2 Fine-Tuning Dashboard', fontsize=16, fontweight='bold')
    axs[0,0].plot(epochs, t_loss, label='Train', color='blue'); axs[0,0].plot(epochs, v_loss, label='Val', color='red'); axs[0,0].set_title('Loss'); axs[0,0].legend()
    axs[0,1].plot(epochs, wers, color='green'); axs[0,1].set_title('WER (%)'); axs[0,1].set_ylim(0,1.1)
    axs[1,0].plot(epochs, cers, color='orange'); axs[1,0].set_title('CER (%)'); axs[1,0].set_ylim(0,1.1)
    axs[1,1].plot(epochs, lrs, color='purple'); axs[1,1].set_title('LR (Log)'); axs[1,1].set_yscale('log')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(output_dir, 'stage2_progress.png'), dpi=150); plt.close()

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Stage 2 starting on device: {device}")
    with open(CONFIG['vocab'], 'r', encoding='utf-8') as f: vocab = json.load(f)
    char2idx = vocab['char2idx']; idx2char = {idx: char for char, idx in char2idx.items()}
    model = create_model(vocab_size=len(char2idx), d_model=CONFIG['d_model'], num_layers=CONFIG['num_layers'])
    if os.path.exists(CONFIG['prev_checkpoint']):
        print(f"Loading weights: {CONFIG['prev_checkpoint']}")
        model.load_state_dict(torch.load(CONFIG['prev_checkpoint'], map_location='cpu')['model_state_dict'], strict=False)
    model = model.to(device); mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, n_fft=400, hop_length=160).to(device)
    train_ds = KonkaniCTCDataset(CONFIG['manifest'], CONFIG['vocab'], CONFIG['max_audio_len'])
    from train_conformer_ctc import KonkaniCTCDataset as PilotDataset
    val_ds = PilotDataset('data/konkani-10k/val_manifest.json', CONFIG['vocab'], CONFIG['max_audio_len'])
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-3)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    os.makedirs(CONFIG['output_dir'], exist_ok=True); stats_p = os.path.join(CONFIG['output_dir'], 'stage2_stats.csv')
    if not os.path.exists(stats_p):
        with open(stats_p, 'w', newline='') as f: csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'wer', 'cer', 'lr'])
    best_val_loss = float('inf')
    for epoch in range(CONFIG['epochs']):
        model.train(); total_t_loss = 0; optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/20 [Train]")
        for i, batch in enumerate(pbar):
            a, al, t, tl = batch['audio'].to(device), batch['audio_lengths'].to(device), batch['text'].to(device), batch['text_lengths'].to(device)
            with torch.no_grad():
                mel = mel_transform(a).transpose(1, 2); mel = torch.log(mel + 1e-9); mel_lens = (al // 160) + 1
            logits, _ = model(mel, mel_lens); log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = criterion(log_probs, t, mel_lens, tl); (loss/CONFIG['grad_accum']).backward()
            if (i+1)%CONFIG['grad_accum']==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad()
            total_t_loss += loss.item(); pbar.set_postfix(loss=f"{loss.item():.4f}")
            if (i+1)%100==0: torch.cuda.empty_cache()
        model.eval(); total_v_loss = 0; all_p, all_t = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                a, al, t, tl, ts = batch['audio'].to(device), batch['audio_lengths'].to(device), batch['text'].to(device), batch['text_lengths'].to(device), batch['text_strs']
                mel = mel_transform(a).transpose(1, 2); mel = torch.log(mel + 1e-9); mel_lens = (al // 160) + 1
                logits, _ = model(mel, mel_lens); total_v_loss += criterion(F.log_softmax(logits, dim=-1).transpose(0, 1), t, mel_lens, tl).item()
                all_p.extend(greedy_decode(logits, mel_lens, idx2char)); all_t.extend(ts)
        avg_t = total_t_loss/len(train_loader); avg_v = total_v_loss/len(val_loader); aw, ac = wer(all_t, all_p), cer(all_t, all_p)
        print(f"Epoch {epoch+1}: T_Loss: {avg_t:.4f} | V_Loss: {avg_v:.4f} | WER: {aw:.2%}"); lr = optimizer.param_groups[0]['lr']
        with open(stats_p, 'a', newline='') as f: csv.writer(f).writerow([epoch+1, avg_t, avg_v, aw, ac, lr])
        save_plots(stats_p, CONFIG['output_dir']); scheduler.step()
        if avg_v < best_val_loss:
            best_val_loss = avg_v; torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'loss': avg_v}, os.path.join(CONFIG['output_dir'], 'best_stage2_model.pt'))

if __name__ == "__main__":
    import gc; gc.collect(); 
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    train()
