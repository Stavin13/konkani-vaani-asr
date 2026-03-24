"""
Quick sanity check: load one real batch and compute CTC loss manually.
"""
import torch, torch.nn.functional as F, json, librosa, numpy as np, os, sys
sys.path.insert(0, 'E:/konkani')
os.chdir('E:/konkani')

import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('data/bpe_tokenizer/konkani_bpe.model')
blank_id = 0  # pad = blank

# Load 4 real samples
samples = [json.loads(l) for l in open('data/konkani-20gb/train.json', encoding='utf-8').readlines()[:8]]

audios, texts = [], []
for s in samples[:4]:
    path = s['audio_filepath']
    audio, _ = librosa.load(path, sr=16000, dtype=np.float32)
    audio = torch.from_numpy(audio[:16000*15])  # cap 15s
    audios.append(audio)
    ids = sp.encode(s['text'], out_type=int)
    texts.append(torch.LongTensor(ids))
    print(f"audio_len={len(audio)} | n_tokens={len(ids)} | mel_frames={len(audio)//160} | text={s['text']!r}")

import torchaudio
mel_tr = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, n_fft=400, hop_length=160)

audio_lens = torch.LongTensor([len(a) for a in audios])
padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)

mel = mel_tr(padded)
mel = torch.log(mel.clamp(min=1e-9)).transpose(1,2).float()
mel_lens = torch.clamp(audio_lens // 160 + 1, max=mel.size(1))
t_lens = torch.LongTensor([len(t) for t in texts])
padded_text = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)

print(f"\nmel shape: {mel.shape}, mel_lens: {mel_lens.tolist()}, t_lens: {t_lens.tolist()}")

# Load model
from models.conformer_ctc_v2 import create_model_v2
ckpt = torch.load('outputs/conformer_v2_200ep/latest_checkpoint.pt', map_location='cpu', weights_only=False)
model = create_model_v2(vocab_size=500, d_model=256, num_layers=12)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

with torch.no_grad():
    logits, _ = model(mel, mel_lens)

print(f"logits shape: {logits.shape}, logits stats: min={logits.min():.4f} max={logits.max():.4f} has_nan={logits.isnan().any()}")

log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0,1)
print(f"log_probs stats: min={log_probs.min():.4f} max={log_probs.max():.4f} has_nan={log_probs.isnan().any()} has_inf={log_probs.isinf().any()}")

criterion = torch.nn.CTCLoss(blank=blank_id, zero_infinity=False)  # NO zero_infinity to see real loss
try:
    loss = criterion(log_probs, padded_text, mel_lens, t_lens)
    print(f"\nCTCLoss (zero_infinity=False): {loss.item():.6f}")
except Exception as e:
    print(f"CTCLoss error: {e}")

criterion_zi = torch.nn.CTCLoss(blank=blank_id, zero_infinity=True)
loss_zi = criterion_zi(log_probs, padded_text, mel_lens, t_lens)
print(f"CTCLoss (zero_infinity=True):  {loss_zi.item():.6f}")
