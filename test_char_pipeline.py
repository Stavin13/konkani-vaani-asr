#!/usr/bin/env python3
"""
Character-based Konkani ASR Integrated Pipeline
Encoder: outputs/conformer_ctc_run1/best_conformer_ctc.pt
LM: KenLM 3-gram
Normalization: log(mel + 1e-9)
"""
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio

BASE = Path("/Volumes/data&proj/konkani")
sys.path.insert(0, str(BASE))

# Architecture is slightly different for run1? 
# Usually ConformerCTC class is consistent. 
# Looking at train_conformer_ctc.py: from models.conformer_ctc import create_model
from models.conformer_ctc import ConformerCTC

# Paths
CHECKPOINT   = BASE / "outputs/conformer_ctc_run1/best_conformer_ctc.pt"
LM_DIR       = BASE / "models/language_models"
VOCAB_FILE   = BASE / "data/konkani-10k/vocab.json"
TEST_MANIFEST= BASE / "data/konkani-10k/test_manifest.json"

# ── Loader ───────────────────────────────────────────────────────────────────
def process_audio(path, device):
    import librosa
    audio, s = librosa.load(path, sr=16000)
    wav = torch.FloatTensor(audio)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1: wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0).to(device)
    
    import torchaudio.transforms as T
    mel_fn = T.MelSpectrogram(16000, n_mels=80, n_fft=400, hop_length=160, win_length=400).to(device)
    mel = mel_fn(wav.unsqueeze(0))
    # TRANSPOSE matches train_conformer_ctc.py:1261
    mel = mel.transpose(1, 2)
    # Norm: log(mel + 1e-9)
    mel = torch.log(mel + 1e-9)
    
    # Calculate mel_lens exactly as in training: (a_lens // 160) + 1
    mel_len = torch.tensor([(wav.size(0) // 160) + 1], device=device)
    return mel.float(), mel_len

# ── Tokenizer ─────────────────────────────────────────────────────────────────
class CharTokenizer:
    def __init__(self):
        v = json.load(open(VOCAB_FILE, encoding='utf-8'))
        self.idx2char = {int(k): c for k, c in v['idx2char'].items()}
        self.vocab_size = v['vocab_size']
        self.blank_id = 0 # Match train_conformer_ctc.py:239

    def decode(self, ids):
        chars = []
        prev = -1
        for i in ids:
            if i != self.blank_id and i != prev:
                chars.append(self.idx2char.get(i, ''))
            prev = i
        return "".join(chars).strip()
        
    def labels(self):
        L = [self.idx2char.get(i, f"<id{i}>") for i in range(self.vocab_size)]
        L[self.blank_id] = ""
        # Handle special tokens for pyctcdecode
        for i, p in enumerate(L):
            if p in ["<pad>", "<blank>", "<sos>", "<eos>", "<unk>"]:
                if i != self.blank_id: L[i] = f"<{p[1:-1]}_{i}>"
        return L

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Loading Character Model on {device}...")
    
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    v_size = state['ctc_head.weight'].shape[0]
    d_model = state['ctc_head.weight'].shape[1]
    print(f"  Vocab Size: {v_size}, D_Model: {d_model}")
    
    # Use ConformerCTC class
    model = ConformerCTC(vocab_size=v_size, input_dim=80, d_model=d_model, num_layers=12)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    
    tok = CharTokenizer()
    
    # Load few samples
    samples = []
    with open(TEST_MANIFEST, 'r', encoding='utf-8') as f:
        for line in f: samples.append(json.loads(line))
    
    import warnings
    warnings.filterwarnings("ignore")
    
    decoder = None
    try:
        from pyctcdecode import build_ctcdecoder
        labels = tok.labels()
        lm_path = LM_DIR / "konkani_3gram.binary"
        if lm_path.exists():
            decoder = build_ctcdecoder(labels, kenlm_model_path=str(lm_path))
    except Exception as e:
        print(f"Warning: Could not load LM: {e}")

    print("\nInference Results (Beam Search + 3G LM):")
    for i in range(5):
        s = samples[i]
        ref = s['text'].strip()
        mel, mel_len = process_audio(s['audio_filepath'], device)
        
        with torch.no_grad():
            logits, _ = model(mel, mel_len)
        
        if decoder:
            lp = F.log_softmax(logits.float(), dim=-1).squeeze(0).cpu().numpy()
            pred = decoder.decode(lp, beam_width=20)
        else:
            ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
            pred = tok.decode(ids)
        
        print(f"\nSample {i+1}:")
        print(f"  REF : {ref}")
        print(f"  PRED: '{pred}'")

if __name__ == "__main__":
    main()
