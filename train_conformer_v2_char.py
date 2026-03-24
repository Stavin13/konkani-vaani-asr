#!/usr/bin/env python3
"""
Conformer-CTC v2 Training Script — Character Variant
Upgrades:
  - Pure CTC Conformer (no hybrid decoder)
  - Character-based tokenization (using provided vocab.json)
  - SpecAugment (freq + time masking)
  - Audio augmentation (speed perturb + noise)
  - Beam search decode for validation metrics
  - FP16 mixed precision
  - Gradient checkpointing
  - Larger effective batch (grad accumulation)
  - 200 epoch training
  - INT8 quantization export at end
  - Auto path remapping for Mac/Unix paths on Windows
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as T
import librosa
import json, os, sys, gc, csv, random
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from jiwer import wer, cer

from models.conformer_ctc_v2 import create_model_v2

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CONFIG = {
    # Data — 20GB chunk
    'train_manifest': 'data/konkani-20gb/train.json',
    'val_manifest':   'data/konkani-20gb/val.json',
    'char_vocab':     'data/konkani-10k/vocab.json',  # Using the provided E:\konkani\data\konkani-10k\vocab.json

    # Model
    'd_model':        256,
    'num_layers':     12,
    'freq_mask':      27,    # SpecAugment F param
    'time_mask':      100,   # SpecAugment T param

    # Training
    'epochs':         200,
    'batch_size':     4,     # Per-GPU batch (6GB VRAM)
    'grad_accum':     16,    # Effective batch = 4 * 16 = 64
    'lr':             3e-4,
    'warmup_epochs':  5,
    'weight_decay':   1e-2,
    'max_audio_len':  16000 * 15,   # 15s max
    'num_workers':    2,

    # Audio augmentation
    'speed_perturb':  True,
    'speed_factors':  [0.9, 1.0, 1.1],
    'noise_prob':     0.3,
    'noise_snr_db':   [10, 30],

    # Beam search
    'beam_width':     10,

    # Output
    'output_dir':     'outputs/conformer_char_v2_run1',
    'save_every':     10,    # Save checkpoint every N epochs
}

# GPU stability
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95)

BASE_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────
# PATH REMAPPING  (Mac /Volumes/... → local Windows paths)
# ─────────────────────────────────────────────────────────────
def remap_path(unix_path: str) -> str:
    if os.path.exists(unix_path):
        return unix_path
    # Strip Mac volume prefix
    for prefix in ['/Volumes/data&proj/konkani/', '/Volumes/data&proj/konkani', '/Volumes/']:
        if unix_path.startswith(prefix):
            rel = unix_path[len(prefix):]
            # Try direct relative
            candidate = BASE_DIR / rel.replace('/', os.sep)
            if candidate.exists():
                return str(candidate)
            # Try stripping first path component
            parts = rel.split('/', 1)
            if len(parts) > 1:
                candidate2 = BASE_DIR / parts[1].replace('/', os.sep)
                if candidate2.exists():
                    return str(candidate2)
    # Filename search in KonkaniRawSpeechCorpus
    fname = os.path.basename(unix_path)
    corpus_dir = BASE_DIR / 'KonkaniRawSpeechCorpus'
    if corpus_dir.exists():
        for root, _, files in os.walk(corpus_dir):
            if fname in files:
                return os.path.join(root, fname)
    return ''


# ─────────────────────────────────────────────────────────────
# TOKENIZER WRAPPER (Character-based)
# ─────────────────────────────────────────────────────────────
class CharTokenizer:
    def __init__(self, char_vocab_path):
        with open(char_vocab_path, encoding='utf-8') as f:
            vocab = json.load(f)
        
        self.char2idx   = vocab['char2idx']
        self.idx2char   = {int(k): v for k, v in vocab['idx2char'].items()}
        self.vocab_size = vocab.get('vocab_size', len(self.char2idx))
        
        # In the provided vocab.json: <pad>=0, <blank>=1
        # PyTorch CTC Loss expects a blank index
        self.blank_id = self.char2idx.get('<blank>', 1)
        self.pad_id   = self.char2idx.get('<pad>', 0)
        
        # Map indices to pieces (characters) for beam search
        self.id2piece = {str(i): c for i, c in self.idx2char.items()}
        print(f'[Tokenizer] Char vocab loaded from {char_vocab_path} — vocab_size={self.vocab_size}, blank_id={self.blank_id}')

    def encode(self, text: str):
        unk = self.char2idx.get('<unk>', 4)
        return [self.char2idx.get(c, unk) for c in text]

    def decode(self, ids):
        chars = []
        for idx in ids:
            if idx == self.blank_id:
                continue
            chars.append(self.idx2char.get(idx, ''))
        return ''.join(chars)

    def decode_ctc(self, ids):
        """CTC collapse: remove blanks and consecutive duplicates."""
        out, prev = [], -1
        for idx in ids:
            if idx != self.blank_id and idx != prev:
                out.append(idx)
            prev = idx
        return self.decode(out)


# ─────────────────────────────────────────────────────────────
# AUDIO AUGMENTATION
# ─────────────────────────────────────────────────────────────
def augment_audio(audio: torch.Tensor, sr=16000) -> torch.Tensor:
    """Speed perturbation + additive noise."""
    if CONFIG['speed_perturb']:
        factor = random.choice(CONFIG['speed_factors'])
        if factor != 1.0:
            effects = [['speed', str(factor)], ['rate', str(sr)]]
            try:
                augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                    audio.unsqueeze(0), sr, effects, channels_first=True
                )
                audio = augmented.squeeze(0)
            except Exception:
                pass

    if random.random() < CONFIG['noise_prob']:
        snr_db = random.uniform(*CONFIG['noise_snr_db'])
        signal_power = audio.pow(2).mean().clamp(min=1e-8)
        noise_power  = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(audio) * noise_power.sqrt()
        audio = audio + noise

    return audio


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class KonkaniDataset(Dataset):
    def __init__(self, manifest_path, tokenizer: CharTokenizer, max_audio_len, augment=False):
        self.tokenizer    = tokenizer
        self.max_len      = max_audio_len
        self.augment      = augment
        self.samples      = []

        with open(manifest_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    s = json.loads(line)
                    local = remap_path(s['audio_filepath'])
                    if local:
                        s['audio_filepath'] = local
                        self.samples.append(s)
                except Exception:
                    pass

        HOP = 160
        MIN_AUDIO_SAMPLES = int(0.5 * 16000)
        filtered, ctc_fail, too_short = [], 0, 0
        for s in self.samples:
            dur = s.get('duration', None)
            text = s.get('text', '')
            n_tokens = len(tokenizer.encode(text)) if text else 0
            n_samples = int(dur * 16000) if dur else max_audio_len
            n_samples = min(n_samples, max_audio_len)
            mel_frames = n_samples // HOP
            
            if n_samples < MIN_AUDIO_SAMPLES:
                too_short += 1
                continue
            if n_tokens == 0 or mel_frames < n_tokens:
                ctc_fail += 1
                continue
            filtered.append(s)
        self.samples = filtered
        print(f'  Dataset: {len(self.samples):,} usable samples (dropped {ctc_fail} CTC-incompat, {too_short} too-short) from {manifest_path}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            audio, _ = librosa.load(s['audio_filepath'], sr=16000, dtype=np.float32)
            audio = torch.from_numpy(audio)
        except Exception:
            audio = torch.zeros(16000)

        if len(audio) > self.max_len:
            audio = audio[:self.max_len]

        if self.augment:
            audio = augment_audio(audio)

        ids = self.tokenizer.encode(s.get('text', ''))
        return {
            'audio':    audio,
            'text':     torch.LongTensor(ids),
            'text_str': s.get('text', ''),
        }


def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x['audio']), reverse=True)
    audios      = [b['audio'] for b in batch]
    texts       = [b['text']  for b in batch]
    audio_lens  = torch.LongTensor([len(a) for a in audios])
    text_lens   = torch.LongTensor([len(t) for t in texts])
    # 0 is <pad> in the vocab
    padded_audio = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
    padded_text  = torch.nn.utils.rnn.pad_sequence(texts,  batch_first=True, padding_value=0)
    return {
        'audio':      padded_audio,
        'audio_lens': audio_lens,
        'text':       padded_text,
        'text_lens':  text_lens,
        'text_strs':  [b['text_str'] for b in batch],
    }


# ─────────────────────────────────────────────────────────────
# MEL FEATURES
# ─────────────────────────────────────────────────────────────
def build_mel_transform(device):
    return T.MelSpectrogram(
        sample_rate=16000, n_mels=80, n_fft=400, hop_length=160, win_length=400,
    ).to(device)


def compute_mel(audio, audio_lens, mel_transform):
    with torch.no_grad():
        mel = mel_transform(audio)
        mel = torch.log(mel.clamp(min=1e-7))
        mel_lens = (audio_lens // 160) + 1
        mel_lens = torch.clamp(mel_lens, max=mel.size(2))
        
        mask = torch.arange(mel.size(2), device=mel.device)[None, None, :] < mel_lens[:, None, None]
        valid_counts = mel_lens[:, None, None].float().clamp(min=1.0)
        mel_sum = (mel * mask).sum(dim=2, keepdim=True)
        mel_mean = mel_sum / valid_counts
        mel_var = (((mel - mel_mean) * mask) ** 2).sum(dim=2, keepdim=True) / valid_counts
        mel_std = mel_var.sqrt()
        mel = (mel - mel_mean) / (mel_std + 1e-5)
        mel = mel * mask
        mel = mel.transpose(1, 2).float()
    return mel, mel_lens


# ─────────────────────────────────────────────────────────────
# DECODING
# ─────────────────────────────────────────────────────────────
def greedy_decode_batch(logits, mel_lens, tokenizer):
    preds = torch.argmax(logits, dim=-1)
    results = []
    for i in range(preds.size(0)):
        ids = preds[i, :mel_lens[i]].tolist()
        results.append(tokenizer.decode_ctc(ids))
    return results


def _build_beam_decoder(tokenizer):
    try:
        from pyctcdecode import build_ctcdecoder
        max_idx = max(int(k) for k in tokenizer.id2piece.keys())
        labels  = [tokenizer.id2piece.get(str(i), '') for i in range(max_idx + 1)]
        # Key fix: ensure the blank token is empty string at its specific index
        labels[tokenizer.blank_id] = ''
        return build_ctcdecoder(labels)
    except ImportError:
        return None


def _beam_decode_batch_cached(logits, mel_lens, tokenizer, beam_width, decoder):
    if decoder is None:
        return greedy_decode_batch(logits, mel_lens, tokenizer)
    log_probs = F.log_softmax(logits, dim=-1)
    lens = mel_lens.cpu().tolist()
    results = []
    for i in range(log_probs.size(0)):
        lp = log_probs[i, :lens[i]].cpu().float().numpy()
        results.append(decoder.decode(lp, beam_width=beam_width))
    return results


# ─────────────────────────────────────────────────────────────
# PLOT METRICS
# ─────────────────────────────────────────────────────────────
def save_plots(stats_path, output_dir, total_epochs):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs, t_loss, v_loss, wers, cers, lrs = [], [], [], [], [], []
    with open(stats_path, encoding='utf-8') as f:
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
    fig.suptitle(f'Conformer-CTC Char Variant — {total_epochs} Epochs', fontsize=16, fontweight='bold')
    axs[0, 0].plot(epochs, t_loss, label='Train'); axs[0, 0].plot(epochs, v_loss, label='Val', color='red')
    axs[0, 0].set_title('CTC Loss'); axs[0, 0].legend(); axs[0, 0].grid(True, alpha=0.3)
    axs[0, 1].plot(epochs, wers, color='green'); axs[0, 1].set_title('WER'); axs[0, 1].set_ylim(0, 1.1)
    axs[1, 0].plot(epochs, cers, color='orange'); axs[1, 0].set_title('CER'); axs[1, 0].set_ylim(0, 1.1)
    axs[1, 1].plot(epochs, lrs, color='purple'); axs[1, 1].set_title('Learning Rate'); axs[1, 1].set_yscale('log')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'training_progress.png'), dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}\n  Conformer-CTC v2 [Char Variant] | Device: {device}\n{"="*60}\n')
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    tokenizer = CharTokenizer(CONFIG['char_vocab'])
    model = create_model_v2(
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG['d_model'],
        num_layers=CONFIG['num_layers'],
        freq_mask_param=CONFIG['freq_mask'],
        time_mask_param=CONFIG['time_mask'],
    )
    model.enable_gradient_checkpointing()

    start_epoch = 0
    ckpt = {}
    resume_ckpt = os.path.join(CONFIG['output_dir'], 'latest_checkpoint.pt')
    if os.path.exists(resume_ckpt):
        print(f'Resuming from: {resume_ckpt}')
        ckpt = torch.load(resume_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        start_epoch = ckpt['epoch'] + 1

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {total_params:,}')

    mel_transform = build_mel_transform(device)
    train_ds = KonkaniDataset(CONFIG['train_manifest'], tokenizer, CONFIG['max_audio_len'], augment=True)
    val_ds   = KonkaniDataset(CONFIG['val_manifest'],   tokenizer, CONFIG['max_audio_len'], augment=False)

    _persist = (CONFIG['num_workers'] > 0) and (sys.platform != 'win32')
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
        collate_fn=collate_fn, num_workers=CONFIG['num_workers'],
        pin_memory=(device.type == 'cuda'), drop_last=True,
        persistent_workers=_persist,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=CONFIG['num_workers'],
        pin_memory=(device.type == 'cuda'),
        persistent_workers=_persist,
    )

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    if start_epoch > 0 and 'optimizer_state_dict' in ckpt:
        try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception: pass
    for pg in optimizer.param_groups:
        pg['lr'] = CONFIG['lr']
        pg['initial_lr'] = CONFIG['lr']

    steps_per_epoch = max(1, len(train_loader) // CONFIG['grad_accum'])
    total_steps     = steps_per_epoch * CONFIG['epochs']
    warmup_steps    = steps_per_epoch * CONFIG['warmup_epochs']
    last_step       = (start_epoch * steps_per_epoch) - 1 if start_epoch > 0 else -1

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_step)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    if start_epoch > 0 and 'scaler_state_dict' in ckpt:
        try: scaler.load_state_dict(ckpt['scaler_state_dict'])
        except Exception: pass

    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    if device.type == 'cuda': torch.set_float32_matmul_precision('high')

    stats_path = str(Path(CONFIG['output_dir']).resolve() / 'training_stats.csv')
    if not os.path.exists(stats_path):
        with open(stats_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'wer', 'cer', 'lr', 'timestamp'])

    best_val_loss = float('inf')
    if start_epoch > 0 and 'best_val_loss' in ckpt:
        best_val_loss = ckpt['best_val_loss']

    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f'Ep {epoch+1:03d}/{CONFIG["epochs"]} [Train]', dynamic_ncols=True)

        for step, batch in enumerate(pbar):
            audio, audio_lens = batch['audio'].to(device), batch['audio_lens'].to(device)
            target, t_lens    = batch['text'].to(device), batch['text_lens'].to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                mel, mel_lens = compute_mel(audio, audio_lens, mel_transform)
                logits, _     = model(mel, mel_lens)
                log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
                loss = criterion(log_probs, target, mel_lens, t_lens) / CONFIG['grad_accum']

            if not torch.isfinite(loss):
                optimizer.zero_grad(); continue

            scaler.scale(loss).backward()
            if (step + 1) % CONFIG['grad_accum'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(); scheduler.step()

            total_train_loss += loss.item() * CONFIG['grad_accum']
            pbar.set_postfix(loss=f'{loss.item()*CONFIG["grad_accum"]:.4f}', lr=f'{scheduler.get_last_lr()[0]:.2e}')

        # Validation
        model.eval()
        total_val_loss, _val_bat = 0.0, 0
        all_preds, all_targets = [], []
        _val_decoder = _build_beam_decoder(tokenizer) if (epoch + 1) % 10 == 0 else None

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Ep {epoch+1:03d} [Val]'):
                audio, audio_lens = batch['audio'].to(device), batch['audio_lens'].to(device)
                target, t_lens    = batch['text'].to(device), batch['text_lens'].to(device)
                
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    mel, mel_lens = compute_mel(audio, audio_lens, mel_transform)
                    logits, _     = model(mel, mel_lens)
                    log_probs     = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
                    val_loss      = criterion(log_probs, target, mel_lens, t_lens)

                if torch.isfinite(val_loss):
                    total_val_loss += val_loss.item(); _val_bat += 1
                
                if (epoch + 1) % 10 == 0:
                    preds = _beam_decode_batch_cached(logits, mel_lens, tokenizer, CONFIG['beam_width'], _val_decoder)
                else:
                    preds = greedy_decode_batch(logits, mel_lens, tokenizer)
                
                all_preds.extend(preds); all_targets.extend(batch['text_strs'])

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / max(1, _val_bat)
        avg_wer = wer(all_targets, all_preds) if all_targets else 1.0
        avg_cer = cer(all_targets, all_preds) if all_targets else 1.0
        lr_now  = scheduler.get_last_lr()[0]

        print(f'  Ep {epoch+1:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | WER: {avg_wer:.2%} | CER: {avg_cer:.2%}')
        with open(stats_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch+1, avg_train_loss, avg_val_loss, avg_wer, avg_cer, lr_now, datetime.now().strftime('%H:%M:%S')])
        save_plots(stats_path, CONFIG['output_dir'], CONFIG['epochs'])

        ckpt_data = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(), 'val_loss': avg_val_loss, 'best_val_loss': best_val_loss,
            'config': CONFIG, 'vocab_size': tokenizer.vocab_size, 'use_bpe': False,
        }
        torch.save(ckpt_data, os.path.join(CONFIG['output_dir'], 'latest_checkpoint.pt'))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss; ckpt_data['best_val_loss'] = best_val_loss
            torch.save(ckpt_data, os.path.join(CONFIG['output_dir'], 'best_model.pt'))
        if (epoch + 1) % CONFIG['save_every'] == 0:
            torch.save(ckpt_data, os.path.join(CONFIG['output_dir'], f'ckpt_epoch{epoch+1:03d}.pt'))

    print(f'\nTraining complete. Best val loss: {best_val_loss:.4f}')

if __name__ == '__main__':
    train()
