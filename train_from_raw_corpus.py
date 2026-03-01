#!/usr/bin/env python3
"""
RTX 3060 (6GB VRAM) Optimized Training Script
Trains from: best_model (1).pt
Dataset:     KonkaniRawSpeechCorpus  (local, Windows)
Author-fixed: remaps /Volumes/data&proj/konkani/ -> e:/konkani/
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import librosa
import json, argparse, os, sys, gc, logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np

# ─────────────────────────────────────────────────────────────
# MEMORY SETTINGS  (RTX 3060 6 GB)
# ─────────────────────────────────────────────────────────────
SETTINGS = {
    'batch_size':                2,
    'gradient_accumulation':     8,     # effective batch = 16
    'max_audio_len':             16000 * 8,  # 8 s
    'max_mel_len':               800,
    'num_workers':               0,     # 0 = main-thread only (safest on Windows)
    'pin_memory':                False,
    # AMP disabled: BatchNorm1d in ConformerBlock does NOT support float16
    # Model is only 4.7M params — fits in 6GB at float32 easily
    'mixed_precision':           False,
    'gradient_checkpointing':    True,
}

# ─────────────────────────────────────────────────────────────
# PATH REMAPPER
# ─────────────────────────────────────────────────────────────
# The manifests were created on a Mac:
#   /Volumes/data&proj/konkani/KonkaniRawSpeechCorpus/…
# The corpus lives locally at (edit BASE_DIR if needed):
BASE_DIR = Path(__file__).resolve().parent  # e.g. e:\konkani

def remap_path(unix_path: str) -> str:
    """Convert mac manifest path to local Windows path."""
    # strip known Mac prefix
    for prefix in [
        "/Volumes/data&proj/konkani/",
        "/Volumes/data&proj/konkani",
        "/Volumes/",
    ]:
        if unix_path.startswith(prefix):
            rel = unix_path[len(prefix):]
            candidate = BASE_DIR / rel.replace("/", os.sep)
            if candidate.exists():
                return str(candidate)
            # try without leading segment (e.g. the volume name)
            parts = rel.split("/", 1)
            if len(parts) > 1:
                candidate2 = BASE_DIR / parts[1].replace("/", os.sep)
                if candidate2.exists():
                    return str(candidate2)
    # maybe it's already a valid local path
    if os.path.exists(unix_path):
        return unix_path
    # last resort: look only by filename inside KonkaniRawSpeechCorpus
    fname = os.path.basename(unix_path)
    corpus_dir = BASE_DIR / "KonkaniRawSpeechCorpus"
    if corpus_dir.exists():
        for root, _, files in os.walk(corpus_dir):
            if fname in files:
                return os.path.join(root, fname)
    return ""   # not found

# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class KonkaniDataset(Dataset):
    def __init__(self, manifest_path, vocab_path, max_audio_len=None, split_name="train"):
        self.max_len = max_audio_len or SETTINGS['max_audio_len']
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.char2idx = vocab['char2idx']
        self.unk = self.char2idx.get('<unk>', 4)

        self.samples = []
        missing = 0
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if s.get('duration', 0) > 10.0:
                    continue
                local = remap_path(s['audio_filepath'])
                if not local:
                    missing += 1
                    continue
                s['audio_filepath'] = local
                self.samples.append(s)

        print(f"[{split_name}] loaded {len(self.samples)} samples  "
              f"(skipped {missing} missing audio files)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            audio, sr = librosa.load(s['audio_filepath'], sr=16000, dtype=np.float32)
            audio = torch.FloatTensor(audio)
        except Exception:
            try:
                audio, sr = torchaudio.load(s['audio_filepath'])
                if sr != 16000:
                    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
                audio = audio.squeeze(0).float()
            except Exception as e:
                print(f"  ⚠ load error {s['audio_filepath']}: {e}")
                audio = torch.zeros(16000)

        # truncate / pad to max_len
        if len(audio) > self.max_len:
            audio = audio[:self.max_len]
        else:
            audio = F.pad(audio, (0, self.max_len - len(audio)))

        text_str = s.get('text', '')[:100]
        ids = [self.char2idx.get(c, self.unk) for c in text_str][:50]
        return {
            'audio':        audio,
            'text':         torch.LongTensor(ids),
            'text_length':  len(ids),
            'audio_length': len(audio),
        }


def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x['audio_length'])
    al = [b['audio_length'] for b in batch]
    tl = [b['text_length']  for b in batch]
    max_a = int(np.percentile(al, 95))
    max_t = min(max(tl), 50)

    audios, texts, als, tls = [], [], [], []
    for b in batch:
        a = b['audio']
        a = a[:max_a] if len(a) > max_a else F.pad(a, (0, max_a - len(a)))
        t = b['text']
        t = t[:max_t] if len(t) > max_t else F.pad(t, (0, max_t - len(t)))
        audios.append(a); texts.append(t)
        als.append(min(b['audio_length'], max_a))
        tls.append(min(b['text_length'],  max_t))

    return {
        'audio':        torch.stack(audios),
        'text':         torch.stack(texts),
        'audio_lengths': torch.LongTensor(als),
        'text_lengths':  torch.LongTensor(tls),
    }


# ─────────────────────────────────────────────────────────────
# MEL FEATURES
# ─────────────────────────────────────────────────────────────
_mel_cache = {}
def get_mel_transform(device):
    key = str(device)
    if key not in _mel_cache:
        _mel_cache[key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=80,
            n_fft=512, hop_length=256, win_length=512
        ).to(device)
    return _mel_cache[key]

def compute_mel(audio, device):
    mel_tf = get_mel_transform(device)
    with torch.no_grad():
        mel = mel_tf(audio)
        mel = torch.log(mel + 1e-8)
    L = SETTINGS['max_mel_len']
    if mel.size(-1) > L:
        mel = mel[..., :L]
    return mel.transpose(1, 2)   # (B, T, 80)


# ─────────────────────────────────────────────────────────────
# TRAIN / VALIDATE
# ─────────────────────────────────────────────────────────────
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def train_epoch(model, loader, optimizer, device, scaler, accum):
    model.train()
    total_loss, n = 0.0, 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="  train")
    for step, batch in enumerate(pbar):
        audio  = batch['audio'].to(device,  non_blocking=True)
        text   = batch['text'].to(device,   non_blocking=True)
        a_lens = batch['audio_lengths'].to(device, non_blocking=True)
        t_lens = batch['text_lengths'].to(device,  non_blocking=True)

        # Full float32 forward pass (BatchNorm1d has no half-precision support)
        mel = compute_mel(audio, device)
        # Only pass text to decoder if sequences are non-empty
        has_target = (t_lens > 0).all() and text.size(1) > 1
        if has_target:
            out = model(mel, a_lens, text, t_lens)
        else:
            out = model(mel, a_lens)  # CTC-only forward

        # ── Extract CTC logits ──
        if isinstance(out, (tuple, list)):
            enc = out[0]   # (B, T, vocab) – ctc_logits
        elif isinstance(out, dict):
            enc = out.get('ctc_logits') or out.get('encoder_outputs')
        else:
            enc = out

        if enc is not None and enc.dim() == 3:
            log_p = F.log_softmax(enc, dim=-1)  # already float32
            in_lens = torch.full((audio.size(0),), log_p.size(1),
                                 dtype=torch.long, device=device)
            ctc = F.ctc_loss(log_p.transpose(0, 1), text,
                             in_lens, t_lens,
                             blank=1, reduction='mean',
                             zero_infinity=True)
        else:
            ctc = torch.tensor(0.0, device=device, requires_grad=True)

        loss = ctc / accum

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += ctc.item();  n += 1
        mem = (f"{torch.cuda.memory_allocated()/1024**3:.1f}GB"
               if torch.cuda.is_available() else "cpu")
        pbar.set_postfix(loss=f"{ctc.item():.4f}", mem=mem)
        del audio, mel, out, ctc, loss

    return total_loss / max(n, 1)


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in tqdm(loader, desc="  val  "):
        audio  = batch['audio'].to(device,  non_blocking=True)
        text   = batch['text'].to(device,   non_blocking=True)
        a_lens = batch['audio_lengths'].to(device, non_blocking=True)
        t_lens = batch['text_lengths'].to(device,  non_blocking=True)

        # Full float32 — no autocast (BatchNorm1d safety)
        mel = compute_mel(audio, device)
        has_target = (t_lens > 0).all() and text.size(1) > 1
        if has_target:
            out = model(mel, a_lens, text, t_lens)
        else:
            out = model(mel, a_lens)

        if isinstance(out, (tuple, list)):
            enc = out[0]
        elif isinstance(out, dict):
            enc = out.get('ctc_logits') or out.get('encoder_outputs')
        else:
            enc = out

        if enc is not None and enc.dim() == 3:
            log_p = F.log_softmax(enc, dim=-1)
            in_lens = torch.full((audio.size(0),), log_p.size(1),
                                 dtype=torch.long, device=device)
            ctc = F.ctc_loss(log_p.transpose(0, 1), text,
                             in_lens, t_lens,
                             blank=1, reduction='mean',
                             zero_infinity=True)
            total_loss += ctc.item(); n += 1
        del audio, mel, out

    clear_memory()
    return total_loss / max(n, 1)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("RTX 3060 Konkani ASR Training")
    parser.add_argument('--checkpoint',    default='best_model (1).pt')
    parser.add_argument('--train_manifest',default='data/konkani-10k/train_manifest.json')
    parser.add_argument('--val_manifest',  default='data/konkani-10k/val_manifest.json')
    parser.add_argument('--vocab_file',    default='data/konkani-10k/vocab.json')
    parser.add_argument('--epochs',        type=int,   default=10)
    parser.add_argument('--lr',            type=float, default=3e-5)
    parser.add_argument('--output_dir',    default='rtx3060_output')
    args = parser.parse_args()

    # ── device ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  RTX 3060 Konkani ASR Training")
    print(f"{'='*60}")
    if device.type == 'cuda':
        print(f"  GPU  : {torch.cuda.get_device_name()}")
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
        torch.cuda.set_per_process_memory_fraction(0.95)
    print(f"  Device: {device}")

    # ── checkpoint ──
    ck_path = args.checkpoint
    if not os.path.exists(ck_path):
        for alt in ['best_model (1).pt', 'best_model.pt',
                    'checkpoints/best_model.pt', 'models/best_model.pt']:
            if os.path.exists(alt):
                ck_path = alt; break
    print(f"\n  Loading checkpoint: {ck_path}")
    checkpoint = torch.load(ck_path, map_location='cpu')
    print(f"  Checkpoint keys: {list(checkpoint.keys())[:6]}")

    # ── auto-detect architecture from state_dict ──
    sd = checkpoint.get('model_state_dict', {})
    if sd:
        # d_model
        d_model = sd.get('encoder.input_proj.weight', torch.zeros(128, 80)).shape[0]
        # vocab_size  — prefer ctc_head, fall back to decoder output_proj
        if 'ctc_head.weight' in sd:
            vocab_size = sd['ctc_head.weight'].shape[0]
        elif 'decoder.output_proj.weight' in sd:
            vocab_size = sd['decoder.output_proj.weight'].shape[0]
        else:
            vocab_size = 81
        # encoder layers
        enc_layers = len({int(k.split('.')[2]) for k in sd
                          if k.startswith('encoder.layers.')})
        enc_layers = max(enc_layers, 1)
        # decoder layers – the actual path is decoder.decoder.layers.N
        import re as _re
        dec_layer_nums = set()
        for k in sd:
            m = _re.search(r'decoder\.decoder\.layers\.(\d+)', k)
            if m:
                dec_layer_nums.add(int(m.group(1)))
        dec_layers = len(dec_layer_nums)
        # num_heads: in_proj_weight is (3*num_heads*head_dim, d_model)
        #            = (num_heads * 3 * head_dim, d_model)
        # PyTorch packs Q,K,V together → shape[0] = 3 * d_model always
        # → num_heads = d_model / head_dim.  Standard training used head_dim=?
        # Safest: derive from weight shape: shape[0] / shape[1] = num_heads if
        # head_dim == d_model / num_heads, but in_proj is (3*d_model, d_model)
        # so shape[0]//shape[1] = 3.  Instead count directly from the weight:
        # num_heads = d_model // head_dim.  head_dim not stored — infer from
        # the fact that in_proj rows = 3*d_model, not helpful.
        # Solution: try all common values and pick first whose head_dim is int.
        # For d_model=128 the standard is num_heads=4 (head_dim=32).
        # We know from inspection the checkpoint uses 3 heads (384/3=128 proj).
        # Better approach: look at the actual in_proj shape:
        #   shape[0] = 3 * d_model for Any num_heads  → always 384 for d_model=128
        # So we cannot infer num_heads from in_proj alone.
        # Hardcode the known value and verify by loading weights:
        num_heads = None   # will be determined below
        for nh in [3, 4, 8, 2, 1]:
            if d_model % nh == 0:
                num_heads = nh
                break
        # conv kernel
        conv_kernel = 31
        for k, v in sd.items():
            if 'depthwise_conv.weight' in k:
                conv_kernel = v.shape[-1]
                break
    else:
        d_model, vocab_size, enc_layers, dec_layers = 128, 81, 8, 0
        num_heads, conv_kernel = 4, 31

    # --- Verify num_heads by trying to load weights with each candidate ---
    # The in_proj_weight shape is (3*d_model, d_model) regardless of num_heads
    # so we determine by trying to build and load the model
    good_num_heads = None
    for nh_candidate in [3, 4, 8, 2, 1]:
        if d_model % nh_candidate != 0:
            continue
        try:
            sys.path.insert(0, str(BASE_DIR / 'archives' / 'kaggle_minimal'))
            from models.konkanivani_asr import create_konkanivani_model
            _test_model = create_konkanivani_model(
                vocab_size=vocab_size,
                config={
                    'input_dim': 80, 'd_model': d_model,
                    'encoder_layers': enc_layers, 'decoder_layers': dec_layers,
                    'num_heads': nh_candidate, 'dropout': 0.1,
                    'conv_kernel_size': conv_kernel,
                }
            )
            if sd:
                _test_model.load_state_dict(sd, strict=True)
            good_num_heads = nh_candidate
            del _test_model
            break
        except Exception:
            if '_test_model' in dir():
                del _test_model
            continue
    if good_num_heads is None:
        good_num_heads = num_heads  # fall back
    num_heads = good_num_heads

    print(f"  Auto-detected arch: d_model={d_model}, vocab={vocab_size}, "
          f"enc_layers={enc_layers}, dec_layers={dec_layers}, "
          f"num_heads={num_heads}, conv_kernel={conv_kernel}")

    # ── model ──
    try:
        sys.path.insert(0, str(BASE_DIR / 'archives' / 'kaggle_minimal'))
        from models.konkanivani_asr import create_konkanivani_model

        model = create_konkanivani_model(
            vocab_size = vocab_size,
            config = {
                'input_dim':        80,
                'd_model':          d_model,
                'encoder_layers':   enc_layers,
                'decoder_layers':   dec_layers,
                'num_heads':        num_heads,
                'dropout':          0.1,
                'conv_kernel_size': conv_kernel,
            }
        )
        if hasattr(model, 'gradient_checkpointing'):
            model.gradient_checkpointing = True

        if sd:
            inc, mis = [], []
            try:
                model.load_state_dict(sd, strict=True)
                print("  ✓ Weights loaded (strict=True)")
            except Exception:
                result = model.load_state_dict(sd, strict=False)
                print(f"  ✓ Weights loaded (strict=False)")
                if result.missing_keys:
                    print(f"    missing : {result.missing_keys[:4]}")
                if result.unexpected_keys:
                    print(f"    unexpected: {result.unexpected_keys[:4]}")
    except ImportError as e:
        print(f"\n  ❌ Cannot import model: {e}")
        print("  Cannot determine model class. Please ensure")
        print("  archives/kaggle_minimal/models/konkanivani_asr.py exists.")
        return

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # ── datasets ──
    print("\n  Building datasets…")
    train_ds = KonkaniDataset(args.train_manifest, args.vocab_file, split_name="train")
    val_ds   = KonkaniDataset(args.val_manifest,   args.vocab_file, split_name="val")

    if len(train_ds) == 0:
        print("\n  ❌ No training samples found!")
        print("     Check that KonkaniRawSpeechCorpus/Data exists and")
        print("     that paths inside the manifests resolve correctly.")
        return

    train_loader = DataLoader(train_ds, batch_size=SETTINGS['batch_size'],
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=SETTINGS['num_workers'],
                              pin_memory=SETTINGS['pin_memory'],
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=SETTINGS['batch_size'],
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=SETTINGS['num_workers'],
                              pin_memory=SETTINGS['pin_memory'])

    # ── optimizer ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01, eps=1e-6)
    accum = SETTINGS['gradient_accumulation']
    steps = max(1, len(train_loader) // accum)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        epochs=args.epochs, steps_per_epoch=steps)

    # No GradScaler — AMP disabled due to BatchNorm1d float16 incompatibility
    scaler = None

    # ── output directory ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'training_log.txt'

    print(f"\n{'='*60}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Batch size      : {SETTINGS['batch_size']}  (eff. {SETTINGS['batch_size']*accum})")
    print(f"  Learning rate   : {args.lr}")
    print(f"  Train samples   : {len(train_ds)}")
    print(f"  Val samples     : {len(val_ds)}")
    print(f"  Output dir      : {out_dir}")
    print(f"  Mixed precision : {SETTINGS['mixed_precision']}")
    print(f"{'='*60}\n")

    best_val = float('inf')
    cfg = checkpoint.get('config', {})
    print(f"  Starting from epoch {checkpoint.get('epoch', 0)} (previous best val_loss={checkpoint.get('val_loss', 'N/A')})")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}  [{datetime.now().strftime('%H:%M:%S')}]")

        tr_loss = train_epoch(model, train_loader, optimizer, device, scaler, accum)
        vl_loss = val_epoch(model, val_loader, device)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(f"  Train loss: {tr_loss:.4f}  |  Val loss: {vl_loss:.4f}  |  LR: {lr_now:.2e}")

        with open(log_path, 'a') as flog:
            flog.write(f"epoch={epoch+1}  train={tr_loss:.4f}  val={vl_loss:.4f}  lr={lr_now:.2e}\n")

        if vl_loss < best_val:
            best_val = vl_loss
            save_path = out_dir / 'best_model_rtx3060.pt'
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss':         vl_loss,
                'config':           cfg,
            }, save_path)
            print(f"  ✓ Saved best model → {save_path}  (val_loss={vl_loss:.4f})")

        clear_memory()

    print(f"\n{'='*60}")
    print(f"  Training complete!  Best val loss: {best_val:.4f}")
    print(f"  Model saved in: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
