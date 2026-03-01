# ==============================================================================
# 🧠 KAGGLE MASTER ASR NOTEBOOK (CONFORMER-CTC)
# ==============================================================================
# INSTRUCTIONS:
# 1. Add Data -> Upload your 20GB Chunks as Datasets.
# 2. Add Data -> Upload your Pilot Model (best_conformer_ctc.pt) as a Dataset.
# 3. Add Data -> Upload your `vocab.json` as a Dataset.
# 4. Use "Settings" to enable GPU (T4 x2 is recommended).
# ==============================================================================

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import librosa
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from jiwer import wer, cer

# ==============================================================================
# 🛠️ 1. DYNAMIC CONFIGURATION (CHANGE CHUNK HERE)
# ==============================================================================
CHUNK_NUMBER = 1 

CONFIG = {
    'batch_size': 32,
    'grad_accum': 1,
    'd_model': 256,
    'num_layers': 12,
    'lr': 1e-4,
    'epochs': 10,
    'max_audio_len': 16000 * 15,
    'num_workers': 4,
}

KAG_PATHS = {
    'audio_root': f'/kaggle/input/konkani-chunk-{CHUNK_NUMBER}',
    'manifest': f'/kaggle/input/konkani-manifests/kaggle_manifest_chunk_{CHUNK_NUMBER}.json',
    'vocab': '/kaggle/input/konkani-vocab/vocab.json',
    'prev_ckpt': '/kaggle/input/konkani-best-model/best_conformer_ctc.pt',
    'output_dir': f'/kaggle/working/outputs_chunk_{CHUNK_NUMBER}'
}

# ==============================================================================
# 🏗️ 2. OVERRIDE MODEL DEFINITION
# ==============================================================================
import sys
sys.path.append('/kaggle/input/konkani-asr-code') 
from models.conformer_ctc import create_model

# ==============================================================================
# 📦 3. KAGGLE DATASET HANDLER
# ==============================================================================
class KaggleKonkaniDataset(Dataset):
    def __init__(self, manifest_path, audio_root, vocab_path):
        self.audio_root = audio_root
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.char2idx = vocab['char2idx']
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f: self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} samples from Chunk {CHUNK_NUMBER}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        path = os.path.join(self.audio_root, s['audio_filepath'])
        try:
            audio, _ = librosa.load(path, sr=16000); audio = torch.FloatTensor(audio)
        except: audio = torch.zeros(16000)
        if len(audio) > CONFIG['max_audio_len']: audio = audio[:CONFIG['max_audio_len']]
        ids = [self.char2idx.get(c, 0) for c in s['text']]
        return {'audio': audio, 'text': torch.LongTensor(ids), 'text_str': s['text']}

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x['audio']), reverse=True)
    a = [b['audio'] for b in batch]; t = [b['text'] for b in batch]
    al = torch.LongTensor([len(x) for x in a]); padded_a = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
    tl = torch.LongTensor([len(x) for x in t]); padded_t = torch.nn.utils.rnn.pad_sequence(t, batch_first=True, padding_value=0)
    return {'audio': padded_a, 'audio_lengths': al, 'text': padded_t, 'text_lengths': tl, 'text_strs': [b['text_str'] for b in batch]}

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
    epochs, t_loss, wers, cers, lrs = [], [], [], [], []
    if not os.path.exists(stats_path): return
    with open(stats_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch'])); t_loss.append(float(row['train_loss']))
            wers.append(float(row.get('wer', 1.0))); cers.append(float(row.get('cer', 1.0)))
            lrs.append(float(row.get('lr', 0)))
    if not epochs: return
    plt.style.use('bmh')
    fig, axs = plt.subplots(2, 2, figsize=(15, 12)); fig.suptitle(f'Kaggle Training: Chunk {CHUNK_NUMBER}', fontsize=16, fontweight='bold')
    axs[0,0].plot(epochs, t_loss, label='Train Loss', color='blue', marker='o'); axs[0,0].set_title('Loss'); axs[0,0].legend()
    axs[0,1].plot(epochs, wers, color='green', marker='o'); axs[0,1].set_title('Word Error Rate'); axs[0,1].set_ylim(0, 1.1)
    axs[1,0].plot(epochs, cers, color='orange', marker='o'); axs[1,0].set_title('Character Error Rate'); axs[1,0].set_ylim(0, 1.1)
    axs[1,1].plot(epochs, lrs, color='purple', marker='o'); axs[1,1].set_title('Learning Rate'); axs[1,1].set_yscale('log')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(output_dir, 'kaggle_dashboard.png')); plt.close()

# ==============================================================================
# 🚀 4. TRAINING ENGINE
# ==============================================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kaggle Session Started: {device}")
    with open(KAG_PATHS['vocab'], 'r') as f: vocab = json.load(f)
    char2idx = vocab['char2idx']; idx2char = {idx: char for char, idx in char2idx.items()}
    model = create_model(vocab_size=len(char2idx), d_model=CONFIG['d_model'], num_layers=CONFIG['num_layers'])
    if os.path.exists(KAG_PATHS['prev_ckpt']):
        print(f"Resuming: {KAG_PATHS['prev_ckpt']}")
        model.load_state_dict(torch.load(KAG_PATHS['prev_ckpt'], map_location='cpu')['model_state_dict'], strict=False)
    model = model.to(device); mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, n_fft=400, hop_length=160).to(device)
    loader = DataLoader(KaggleKonkaniDataset(KAG_PATHS['manifest'], KAG_PATHS['audio_root'], KAG_PATHS['vocab']), batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-3); criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    os.makedirs(KAG_PATHS['output_dir'], exist_ok=True); stats_p = os.path.join(KAG_PATHS['output_dir'], 'stats.csv')
    if not os.path.exists(stats_p):
        with open(stats_p, 'w', newline='') as f: csv.writer(f).writerow(['epoch', 'train_loss', 'wer', 'cer', 'lr'])
    
    for epoch in range(CONFIG['epochs']):
        model.train(); total_loss = 0; all_p, all_t = []
        pbar = tqdm(loader, desc=f"Chunk {CHUNK_NUMBER} | Epoch {epoch+1}")
        for i, batch in enumerate(pbar):
            a, al, t, tl, ts = batch['audio'].to(device), batch['audio_lengths'].to(device), batch['text'].to(device), batch['text_lengths'].to(device), batch['text_strs']
            with torch.no_grad(): mel = mel_tf(a).transpose(1, 2); mel = torch.log(mel + 1e-9); m_len = (al // 160) + 1
            logits, _ = model(mel, m_len); loss = criterion(F.log_softmax(logits, dim=-1).transpose(0, 1), t, m_len, tl)
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            total_loss += loss.item(); pbar.set_postfix(loss=f"{loss.item():.4f}")
            if i % 50 == 0: all_p.extend(greedy_decode(logits, m_len, idx2char)); all_t.extend(ts)
        avg_l, aw, ac = total_loss/len(loader), wer(all_t, all_p), cer(all_t, all_p)
        lr = optimizer.param_groups[0]['lr']
        with open(stats_p, 'a', newline='') as f: csv.writer(f).writerow([epoch+1, avg_l, aw, ac, lr])
        save_plots(stats_p, KAG_PATHS['output_dir'])
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'loss': avg_l, 'config': CONFIG}, os.path.join(KAG_PATHS['output_dir'], f'ckpt_epoch{epoch+1}.pt'))
        print(f"Saved Epoch {epoch+1} | WER: {aw:.2%}")

if __name__ == "__main__":
    train()
