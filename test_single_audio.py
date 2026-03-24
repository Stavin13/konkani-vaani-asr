#!/usr/bin/env python3
"""
Test INT8 Quantized 200ep ASR Pipeline
Encoder: outputs/conformer_v2_200ep/best_model_int8.pt
LM: KenLM 3-gram
"""
import json, os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

BASE = Path("/Volumes/data&proj/konkani")
sys.path.insert(0, str(BASE))

from models.conformer_ctc_v2 import ConformerCTCv2

# ── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINT   = BASE / "outputs/conformer_v2_200ep/best_model.pt"
LM_DIR       = BASE / "models/language_models"
BPE_VOCAB    = BASE / "data/bpe_tokenizer/bpe_vocab.json"
BPE_MODEL    = BASE / "data/bpe_tokenizer/konkani_bpe.model"
# Reference: शान हो एक नामनेचो गायक.
AUDIO_FILE   = "/Volumes/data&proj/konkani/KonkaniRawSpeechCorpus/Data/Phonetically Balanced-W4/Female/21To50/LDC-IL_Scheduled_Konkani_Female_21To50_Phonetically Balanced-W4_SP-0421_W4-0010.wav"
# Reference: अंतीं

# ── Loader ───────────────────────────────────────────────────────────────────
def process_audio(path, device):
    wav, s = torchaudio.load(path)
    if s != 16000:
        import torchaudio.transforms as T
        wav = T.Resample(s, 16000)(wav)
    if wav.size(0) > 1: wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0).to(device)
    
    import torchaudio.transforms as T
    mel_fn = T.MelSpectrogram(16000, n_mels=80, n_fft=400, hop_length=160, win_length=400).to(device)
    mel = torch.log(mel_fn(wav.unsqueeze(0)).clamp(min=1e-7))
    # Normalization
    mel = (mel - (-10.0)) / (4.0 + 1e-5)
    mel_len = torch.tensor([mel.size(2)], device=device)
    return mel.transpose(1, 2).float(), mel_len

# ── Tokenizer ─────────────────────────────────────────────────────────────────
class BPETokenizer:
    def __init__(self):
        import sentencepiece as spm
        v = json.load(open(BPE_VOCAB, encoding='utf-8'))
        self.vocab_size = v['vocab_size']
        self.blank_id = v.get('blank_id', 0)
        self.id2piece = v['id2piece']
        self.sp = spm.SentencePieceProcessor(model_file=str(BPE_MODEL))

    def decode(self, ids):
        clean = []
        prev = -1
        for i in ids:
            if i != self.blank_id and i != prev: clean.append(i)
            prev = i
        return self.sp.decode(clean)
        
    def labels(self):
        L = []
        for i in range(self.vocab_size):
            p = self.id2piece.get(str(i), f"<id{i}>")
            if p in ["<pad>", "<s>", "</s>", "<unk>", "<blank>"]:
                if i == self.blank_id: L.append("")
                else: L.append(f"<{p[1:-1]}_{i}>") # unique special tag
            elif p == "": L.append(f"<empty_{i}>")
            else: L.append(p)
        return L

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Loading FP32 Model on {device}...")
    
    # 1. Init FP32
    v_size = 500
    d_model = 256
    model = ConformerCTCv2(vocab_size=v_size, input_dim=80, d_model=d_model, num_layers=12)
    
    # 2. Load weights
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    
    tok = BPETokenizer()
    print(f"Inference on: {Path(AUDIO_FILE).name}")
    
    # Audio
    mel, mel_len = process_audio(AUDIO_FILE, device)
    
    # Forward
    with torch.no_grad():
        logits, _ = model(mel, mel_len)
    
    # Greedy
    ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    pred_greedy = tok.decode(ids)
    print(f"\nGreedy Decoding: '{pred_greedy}'")
    
    # LM
    try:
        from pyctcdecode import build_ctcdecoder
        labels = tok.labels()
        lm3 = LM_DIR / "konkani_3gram.binary"
        if lm3.exists():
            decoder = build_ctcdecoder(labels, kenlm_model_path=str(lm3))
            lp = F.log_softmax(logits.float(), dim=-1).squeeze(0).cpu().numpy()
            pred_lm = decoder.decode(lp, beam_width=20)
            print(f"3-gram LM Decoding: '{pred_lm}'")
    except Exception as e:
        print(f"LM Decoding failed: {e}")

if __name__ == "__main__":
    main()
