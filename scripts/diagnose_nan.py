#!/usr/bin/env python3
"""Diagnose NaN source in training pipeline."""
import torch, json, os, numpy as np, librosa, sys
import torch.nn.functional as F
import torchaudio.transforms as T
sys.path.insert(0, '.')
from models.conformer_ctc_v2 import create_model_v2

# ── Tokenizer ──────────────────────────────────────────────
bpe_vocab = 'data/bpe_tokenizer/bpe_vocab.json'
with open(bpe_vocab, encoding='utf-8') as f:
    bpe_data = json.load(f)
vocab_size = bpe_data['vocab_size']
blank_id   = 0
print(f'Vocab size : {vocab_size}')

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('data/bpe_tokenizer/konkani_bpe.model')

# ── Load a few samples ─────────────────────────────────────
samples = []
with open('data/konkani-20gb/train.json', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 8: break
        samples.append(json.loads(line))

print(f'\nChecking {len(samples)} samples...')
for i, s in enumerate(samples):
    path = s['audio_filepath']
    exists = os.path.exists(path)
    text   = s.get('text', '')
    ids    = sp.encode(text, out_type=int)
    print(f'  [{i}] exists={exists} | text_len={len(text)} | tokens={len(ids)} | path={os.path.basename(path)}')

# ── Mel check ──────────────────────────────────────────────
print('\nMel feature check:')
mel_t = T.MelSpectrogram(sample_rate=16000, n_mels=80, n_fft=400, hop_length=160, win_length=400)
for i, s in enumerate(samples[:4]):
    if not os.path.exists(s['audio_filepath']):
        print(f'  [{i}] MISSING FILE')
        continue
    audio, _ = librosa.load(s['audio_filepath'], sr=16000, dtype=np.float32)
    audio_t  = torch.from_numpy(audio).unsqueeze(0)
    mel      = mel_t(audio_t)
    mel_log  = torch.log(mel.clamp(min=1e-9)).transpose(1, 2).float()
    ids      = sp.encode(s.get('text',''), out_type=int)
    mel_len  = mel_log.size(1)
    txt_len  = len(ids)
    ctc_ok   = mel_len >= txt_len
    print(f'  [{i}] mel_len={mel_len} txt_len={txt_len} ctc_ok={ctc_ok} '
          f'nan={torch.isnan(mel_log).any().item()} inf={torch.isinf(mel_log).any().item()} '
          f'range=[{mel_log.min():.2f},{mel_log.max():.2f}]')

# ── Model forward + CTC loss check ─────────────────────────
print('\nModel forward + CTC loss check:')
model = create_model_v2(vocab_size=vocab_size, d_model=256, num_layers=12)
model.eval()
criterion = torch.nn.CTCLoss(blank=blank_id, zero_infinity=True)

for i, s in enumerate(samples[:4]):
    if not os.path.exists(s['audio_filepath']):
        continue
    audio, _ = librosa.load(s['audio_filepath'], sr=16000, dtype=np.float32)
    audio_t  = torch.from_numpy(audio).unsqueeze(0)
    mel      = mel_t(audio_t)
    mel_log  = torch.log(mel.clamp(min=1e-9)).transpose(1, 2).float()
    mel_len  = torch.LongTensor([mel_log.size(1)])
    ids      = sp.encode(s.get('text',''), out_type=int)
    target   = torch.LongTensor(ids).unsqueeze(0)
    t_len    = torch.LongTensor([len(ids)])

    with torch.no_grad():
        logits, _ = model(mel_log, mel_len)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        loss      = criterion(log_probs, target, mel_len, t_len)

    print(f'  [{i}] logits nan={torch.isnan(logits).any().item()} '
          f'loss={loss.item():.4f} finite={torch.isfinite(loss).item()} '
          f'mel_len={mel_len.item()} txt_len={t_len.item()}')

print('\nDiagnosis complete.')
