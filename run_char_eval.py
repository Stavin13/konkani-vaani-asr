#!/usr/bin/env python3
"""
Full Evaluation for Character-based Konkani ASR Pipeline
Model: outputs/conformer_ctc_run1/best_conformer_ctc.pt
LM: KenLM 3-gram & 4-gram
"""
import json, os, sys, time
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "mamba"))

from models.conformer_ctc import ConformerCTC

# Paths
CHECKPOINT   = BASE / "outputs/conformer_ctc_run1/best_conformer_ctc.pt"
LM_DIR       = BASE / "models/language_models"
VOCAB_FILE   = BASE / "data/konkani-10k/vocab.json"
TEST_MANIFEST= BASE / "data/konkani-ultimate/train.json"
MAMBA_CHECKPOINT = BASE / "mamba/best_model_mamba_test.pt"
MAMBA_VOCAB      = BASE / "data/konkani-10k/vocab.json"


# ── Loader ───────────────────────────────────────────────────────────────────
def process_audio(path, device):
    try:
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
        mel = mel.transpose(1, 2)
        mel = torch.log(mel + 1e-9)
        mel_len = torch.tensor([(wav.size(0) // 160) + 1], device=device)
        return mel.float(), mel_len
    except: return None, None

# ── Tokenizer ─────────────────────────────────────────────────────────────────
class CharTokenizer:
    def __init__(self):
        v = json.load(open(VOCAB_FILE, encoding='utf-8'))
        self.idx2char = {int(k): c for k, c in v['idx2char'].items()}
        self.vocab_size = v['vocab_size']
        self.blank_id = 0

    def decode(self, ids):
        chars = []
        prev = -1
        for i in ids:
            if i != self.blank_id and i != prev:
                chars.append(self.idx2char.get(i, ''))
            prev = i
        return "".join(chars).strip()
        
    def labels(self):
        L = []
        for i in range(self.vocab_size):
            p = self.idx2char.get(i, f"<id{i}>")
            if p in ["<pad>", "<blank>", "<sos>", "<eos>", "<unk>"]:
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

# ── Eval Core ─────────────────────────────────────────────────────────────────
def run_eval(model, tok, device, samples, decoder=None, label="Greedy", beam=10):
    total_wer, total_cer, processed = 0.0, 0.0, 0
    t0 = time.time()
    print(f"\nEvaluating {label}...")
    
    for s in tqdm(samples):
        mel, mel_len = process_audio(s['audio_filepath'], device)
        if mel is None: continue
        
        with torch.no_grad():
            logits, _ = model(mel, mel_len)
        
        ref = s['text'].strip()
        if decoder:
            lp = F.log_softmax(logits.float(), dim=-1).squeeze(0).cpu().numpy()
            hyp = decoder.decode(lp, beam_width=beam)
        else:
            ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
            hyp = tok.decode(ids)
            
        w, c = wer_cer(ref, hyp)
        total_wer += w
        total_cer += c
        processed += 1
        
    if processed == 0: return None
    
    res = {"label": label, "wer": (total_wer/processed)*100, "cer": (total_cer/processed)*100, "n": processed, "time": time.time()-t0}
    print(f"  {label} -> WER: {res['wer']:.2f}% | CER: {res['cer']:.2f}%")
    return res

def run_mamba_eval(model, tok, mamba_model, mamba_tok, device, samples, label="ASR + Mamba Corrector"):
    total_wer, total_cer, processed = 0.0, 0.0, 0
    t0 = time.time()
    print(f"\nEvaluating {label}...")
    
    for s in tqdm(samples):
        mel, mel_len = process_audio(s['audio_filepath'], device)
        if mel is None: continue
        
        with torch.no_grad():
            logits, _ = model(mel, mel_len)
        
        # Greedy decoding
        ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        hyp_greedy = tok.decode(ids)
        
        # Mamba correction
        src_ids = mamba_tok.encode(hyp_greedy) + [mamba_tok.sep_id]
        src_t   = torch.tensor([src_ids], dtype=torch.long, device=device)
        mask    = torch.ones_like(src_t)
        
        with torch.no_grad():
            out_ids = mamba_model.generate(src_t, attention_mask=mask, max_new=len(s['text'].strip())+20)
        hyp = mamba_tok.decode(out_ids)
        
        ref = s['text'].strip()
        w, c = wer_cer(ref, hyp)
        total_wer += w
        total_cer += c
        processed += 1
        
    if processed == 0: return None
    
    res = {"label": label, "wer": (total_wer/processed)*100, "cer": (total_cer/processed)*100, "n": processed, "time": time.time()-t0}
    print(f"  {label} -> WER: {res['wer']:.2f}% | CER: {res['cer']:.2f}%")
    return res


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Full Character Model Evaluation on {device}")
    
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    v_size = state['ctc_head.weight'].shape[0]
    model = ConformerCTC(vocab_size=v_size, input_dim=80, d_model=256, num_layers=12)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    
    tok = CharTokenizer()
    samples = [json.loads(l) for l in open(TEST_MANIFEST, encoding='utf-8')]
    print(f"Test Set Size: {len(samples)}")
    
    results = []
    
    # 1. Greedy
    rg = run_eval(model, tok, device, samples, label="Greedy (Baseline)")
    if rg: results.append(rg)
    
    # 2. LM
    try:
        from pyctcdecode import build_ctcdecoder
        labels = tok.labels()
        for ngram in ["3gram", "4gram"]:
            lmp = LM_DIR / f"konkani_{ngram}.binary"
            if lmp.exists():
                dec = build_ctcdecoder(labels, kenlm_model_path=str(lmp))
                rlm = run_eval(model, tok, device, samples, decoder=dec, label=f"ASR + {ngram} LM", beam=15)
                if rlm: results.append(rlm)
    except: pass

    # 3. Mamba Corrector
    try:
        from train_custom_mamba import TinyMambaCorrectorModel, KonkaniCharTokenizer
        mamba_tok = KonkaniCharTokenizer(str(MAMBA_VOCAB))
        
        state_dict = torch.load(MAMBA_CHECKPOINT, map_location=device)
        config = state_dict.get('config', {})
        
        mamba_model = TinyMambaCorrectorModel(
            vocab_size=83,
            d_model=config.get("d_model", 256),
            n_layers=config.get("n_layers", 6),
            d_state=config.get("d_state", 16),
            d_conv=config.get("d_conv", 4),
            expand=config.get("expand", 2),
            dropout=config.get("dropout", 0.1)
        )
        
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict['model_state'].items()}
        mamba_model.load_state_dict(clean_state_dict)
        mamba_model.eval().to(device)
        
        rmm = run_mamba_eval(model, tok, mamba_model, mamba_tok, device, samples, label="ASR + Mamba Corrector")
        if rmm: results.append(rmm)
    except Exception as e:
        print(f"Error running Mamba corrector evaluation: {e}")

    # Report
    print(f"\n{'='*70}")
    print(f"{'FINAL CHARACTER PIPELINE ACCURACY REPORT':^70}")
    print(f"{'='*70}")
    print(f"{'Configuration':<35} {'WER (%)':>12} {'CER (%)':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<35} {r['wer']:>12.2f} {r['cer']:>12.2f}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
