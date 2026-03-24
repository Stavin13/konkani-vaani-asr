#!/usr/bin/env python3
"""
Konkani ASR - Full Pipeline Accuracy Evaluation
Integrated: Encoder (Conformer) + LM (KenLM / Neural)
Fixing: Mel Normalization and pyctcdecode argument.
"""
import json, os, sys, time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

BASE = Path("/Volumes/data&proj/konkani")
sys.path.insert(0, str(BASE))

# Paths
CHECKPOINT   = BASE / "outputs/conformer_v2_200ep/best_model.pt"
LM_DIR       = BASE / "models/language_models"
BPE_VOCAB    = BASE / "data/bpe_tokenizer/bpe_vocab.json"
BPE_MODEL    = BASE / "data/bpe_tokenizer/konkani_bpe.model"
TEST_MANIFEST= BASE / "data/konkani-10k/test_manifest.json"

from models.conformer_ctc_v2 import ConformerCTCv2

# ── Robust Audio Loader ───────────────────────────────────────────────────────
def load_audio(path, sr=16000):
    try:
        if not os.path.exists(path):
            return None
        wav, s = torchaudio.load(path)
        if s != sr:
            import torchaudio.transforms as T
            wav = T.Resample(s, sr)(wav)
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        return wav.squeeze(0).float()
    except Exception as e:
        return None

# ── Mel Processor ─────────────────────────────────────────────────────────────
_mel_fn = None
def get_mel_fn(device):
    global _mel_fn
    if _mel_fn is None:
        import torchaudio.transforms as T
        _mel_fn = T.MelSpectrogram(
            sample_rate=16000, n_mels=80, n_fft=400,
            hop_length=160, win_length=400
        ).to(device)
    return _mel_fn

def compute_mel(audio, device):
    mel_fn = get_mel_fn(device)
    with torch.no_grad():
        m = mel_fn(audio.unsqueeze(0).to(device))
        m = torch.log(m.clamp(min=1e-7)) # (1, 80, T)
        m = m.squeeze(0) # (80, T)
        # Normalization used in train_conformer_v2.py
        m = (m - (-10.0)) / (4.0 + 1e-5)
        m_len = torch.tensor([m.size(1)], device=device)
        m = m.transpose(0, 1).unsqueeze(0).float() # (1, T, 80)
    return m, m_len

# ── Tokenizer ─────────────────────────────────────────────────────────────────
class BPETokenizer:
    def __init__(self):
        import sentencepiece as spm
        v = json.load(open(BPE_VOCAB, encoding='utf-8'))
        self.vocab_size = v['vocab_size']
        self.blank_id   = v.get('blank_id', 0) # usually 0
        self.id2piece   = v['id2piece']
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(BPE_MODEL))

    def decode_ids(self, ids):
        clean = []
        prev = -1
        for i in ids:
            if i != self.blank_id and i != prev:
                clean.append(i)
            prev = i
        return self.sp.decode(clean) if clean else ""

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

# ── Accuracy Helpers ──────────────────────────────────────────────────────────
def edit_dist(a, b):
    m, n = len(a), len(b)
    dp = list(range(n+1))
    for i in range(1, m+1):
        prev, dp[0] = dp[0], i
        for j in range(1, n+1):
            t = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = t
    return dp[n]

def wer_cer(ref, hyp):
    c_dist = edit_dist(list(ref), list(hyp))
    c_len = max(len(ref), 1)
    
    r_words, h_words = ref.split(), hyp.split()
    w_dist = edit_dist(r_words, h_words)
    w_len = max(len(r_words), 1)
    
    return w_dist / w_len, c_dist / c_len

# ── Pipeline Eval ─────────────────────────────────────────────────────────────
def run_eval(model, tok, device, samples, decoder=None, label="Greedy", beam=10):
    total_wer = 0.0
    total_cer = 0.0
    processed = 0
    t0 = time.time()
    
    print(f"\nRunning {label}...")
    
    for i, s in enumerate(samples[:args.count]):
        audio = load_audio(s['audio_filepath'])
        if audio is None: continue
        
        mel, mel_len = compute_mel(audio, device)
        with torch.no_grad():
            try:
                logits, _ = model(mel, mel_len)
            except: continue
        
        ref = s['text'].strip()
        if decoder:
            lp = F.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            hyp = decoder.decode(lp, beam_width=beam)
        else:
            ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
            hyp = tok.decode_ids(ids)
            
        w, c = wer_cer(ref, hyp)
        total_wer += w
        total_cer += c
        processed += 1
        
        if i < 3:
            print(f"  Sample {i+1}:")
            print(f"    REF: {ref}")
            print(f"    HYP: {hyp}")

    if processed == 0: return None
    
    avg_wer = total_wer / processed * 100
    avg_cer = total_cer / processed * 100
    print(f"  Result -> WER: {avg_wer:.2f}% | CER: {avg_cer:.2f}% ({time.time()-t0:.1f}s)")
    return {"label": label, "wer": avg_wer, "cer": avg_cer, "n": processed}

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global args
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=50)
    p.add_argument('--beam', type=int, default=10)
    args = p.parse_args()
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Konkani ASR Integration Test ({device})")
    
    # Model
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    v_size = state['ctc_head.weight'].shape[0]
    d_model = state['ctc_head.weight'].shape[1]
    
    model = ConformerCTCv2(vocab_size=v_size, input_dim=80, d_model=d_model, num_layers=12)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    
    tok = BPETokenizer()
    
    # Samples
    samples = []
    with open(TEST_MANIFEST, 'r', encoding='utf-8') as f:
        for line in f: samples.append(json.loads(line))
    
    results = []
    
    # Greedy
    rg = run_eval(model, tok, device, samples, label="Greedy (Baseline)")
    if rg: results.append(rg)
    
    # NGO KenLM
    try:
        from pyctcdecode import build_ctcdecoder
        labels = tok.labels()
        
        lm_fname = "konkani_3gram.binary"
        l_path = LM_DIR / lm_fname
        if l_path.exists():
            # Standard build_ctcdecoder (pyctcdecode 0.5.0)
            dec = build_ctcdecoder(labels, kenlm_model_path=str(l_path))
            rlm = run_eval(model, tok, device, samples, decoder=dec, label="ASR + 3-gram integrated", beam=args.beam)
            if rlm: results.append(rlm)
    except Exception as e:
        print(f"LM integration failed: {e}")

    # Final Report
    if results:
        print(f"\n{'INTEGRATED PIPELINE ACCURACY REPORT':^60}")
        print("-" * 60)
        for r in results:
            print(f"{r['label']:<40} WER: {r['wer']:>6.2f}% CER: {r['cer']:>6.2f}%")
        print("-" * 60)

if __name__ == "__main__":
    main()
