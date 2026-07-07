#!/usr/bin/env python3
"""
Clean WER Evaluation — Konkani ASR (Bucketed & Robust)
=====================================================
Runs the model on the test manifest and calculates WER/CER split by:
  - Word/Command Clips (LDC)
  - Sentence Clips (LDC Sentence-S)
  - Long-form Segments (segment_*.wav)
"""

import json
import re
import sys
import time
import argparse
import unicodedata
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).resolve().parent.parent
CHECKPOINT    = BASE / "outputs/conformer_ctc_run1/best_conformer_ctc.pt"
VOCAB_FILE    = BASE / "data/konkani-10k/vocab.json"
TEST_MANIFEST = BASE / "data/konkani-combined/manifests/test.json"
UNIGRAMS      = BASE / "models/language_models/unigrams.txt"

sys.path.insert(0, str(BASE))
from models.conformer_ctc import ConformerCTC

# ── Characters to strip in "clean" mode ───────────────────────────────────────
MISSING_DIGITS = set("०३४५६७८")

def normalise_text(text: str, clean: bool = False) -> str:
    """Normalize and clean Devanagari text for robust comparison."""
    if text is None: return ""
    
    # 1. Unicode NFC is essential for Devanagari
    text = unicodedata.normalize('NFC', text)
    
    # 2. Strip hidden control/formatting characters (the "Invisible Killers")
    text = text.replace('\u200c', '') # ZWNJ
    text = text.replace('\u200d', '') # ZWJ
    text = text.replace('\ufeff', '') # BOM
    
    # 3. Handle slash alternatives: पयलो/पयलें -> पयलो
    text = re.sub(r'(\w+)/(\w+)', r'\1', text)
    
    # 4. Remove system tokens
    text = re.sub(r'<unk(?:_\d+)?>', '', text)
    text = re.sub(r'_[0-9]+_', '', text) # Remove our dummy tags if any
    
    if clean:
        # Punctuation and specific char cleaning
        text = text.replace('‘', "'").replace('’', "'").replace('“', '"').replace('”', '"')
        text = text.replace('—', '-').replace('–', '-')
        
        cleaned = []
        for ch in text:
            cat = unicodedata.category(ch)
            # L=Letter, M=Mark (matras), N=Number, Z=Separator (space)
            if (cat.startswith('L') or cat.startswith('M') or cat.startswith('N') or ch.isspace()):
                if ch not in MISSING_DIGITS:
                    cleaned.append(ch)
        text = "".join(cleaned)
    
    # Standardize spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def has_mixed_script(text: str) -> bool:
    """Check if text contains both Devanagari and Latin script."""
    has_deva = bool(re.search(r'[\u0900-\u097F]', text))
    has_latin = bool(re.search(r'[a-zA-Z]', text))
    return has_deva and has_latin

def get_bucket(filepath: str, text: str) -> str:
    """Categorize sample into word/command, sentence, or long-form."""
    filepath_str = str(filepath)
    if "segment_" in filepath_str:
        return "long-form"
    elif "LDC-IL" in filepath_str:
        if "Sentence-S" in filepath_str:
            return "sentence"
        else:
            return "word/command"
    else:
        word_count = len(text.split())
        return "word/command" if word_count <= 2 else "sentence"

# ── VAD ───────────────────────────────────────────────────────────────────────
_vad_model = None
_utils = None

def _get_vad():
    global _vad_model, _utils
    if _vad_model is None:
        _vad_model, _utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True, verbose=False)
    return _vad_model, _utils

VAD_THRESHOLD_SEC = 8

def _wav_to_mel(wav_1d: torch.Tensor, device: torch.device):
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=80, n_fft=400, hop_length=160, win_length=400
    ).to(device)
    mel = mel_fn(wav_1d.unsqueeze(0).to(device))
    mel = mel.transpose(1, 2)
    mel = torch.log(mel + 1e-9)
    mel_len = torch.tensor([mel.size(1)], device=device)
    return mel.float(), mel_len

def process_audio(path: str, device: torch.device):
    try:
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0)

        duration_sec = wav.size(0) / 16000
        if duration_sec <= VAD_THRESHOLD_SEC:
            return [_wav_to_mel(wav, device)]

        vad_model, utils = _get_vad()
        get_speech_timestamps = utils[0]
        timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000, min_silence_duration_ms=300)
        
        if not timestamps:
            return [_wav_to_mel(wav, device)]

        chunks = []
        cur_start, cur_end = timestamps[0]['start'], timestamps[0]['end']
        for seg in timestamps[1:]:
            if (seg['end'] - cur_start) / 16000 <= VAD_THRESHOLD_SEC:
                cur_end = seg['end']
            else:
                chunks.append((cur_start, cur_end))
                cur_start, cur_end = seg['start'], seg['end']
        chunks.append((cur_start, cur_end))
        
        results = []
        for start, end in chunks:
            if (end - start) < 400: continue
            results.append(_wav_to_mel(wav[start:end], device))
        return results if results else [_wav_to_mel(wav, device)]
    except Exception:
        import traceback
        traceback.print_exc()
        return [(None, None)]

def compute_snr(path: str):
    """Calculate Signal-to-Noise Ratio using Silero VAD to segment speech/noise."""
    try:
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.mean(0)
        
        vad_model, utils = _get_vad()
        get_speech_timestamps = utils[0]
        speech_ts = get_speech_timestamps(wav, vad_model, sampling_rate=16000)
        
        if not speech_ts:
            return 0.0
            
        speech_mask = torch.zeros_like(wav, dtype=torch.bool)
        for ts in speech_ts:
            speech_mask[ts['start']:ts['end']] = True
            
        s_samples = wav[speech_mask]
        n_samples = wav[~speech_mask]
        
        if len(s_samples) == 0 or len(n_samples) == 0:
            return 0.0
            
        s_pow = (s_samples ** 2).mean().item()
        n_pow = (n_samples ** 2).mean().item()
        
        if n_pow <= 1e-10: return 50.0 # Clean
        return 10 * torch.log10(torch.tensor(s_pow / n_pow)).item()
    except:
        return 0.0

# ── Tokenizer ──────────────────────────────────────────────────────────────────
class CharTokenizer:
    def __init__(self):
        v = json.load(open(VOCAB_FILE, encoding="utf-8"))
        self.idx2char = {int(k): c for k, c in v["idx2char"].items()}
        self.vocab_size = v["vocab_size"]
        self.blank_id = 1 # Correct index for <blank> in this vocab

    def decode(self, ids):
        chars, prev = [], -1
        for i in ids:
            if i != self.blank_id and i != prev:
                ch = self.idx2char.get(i, "")
                if not (ch.startswith("<") and ch.endswith(">")):
                    chars.append(ch)
            prev = i
        return "".join(chars).strip()

    def labels(self):
        L = []
        for i in range(self.vocab_size):
            p = self.idx2char.get(i, f"<id{i}>")
            if p in ["<pad>", "<blank>", "<sos>", "<eos>", "<unk>"]:
                L.append("") if i == self.blank_id else L.append(f"<{p[1:-1]}_{i}>")
            else:
                L.append(p)
        return L

# ── Scoring ────────────────────────────────────────────────────────────────────
def edit_dist(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            t = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = t
    return dp[n]

def wer_cer(ref: str, hyp: str):
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    c_dist = edit_dist(ref_chars, hyp_chars)
    c_len = max(len(ref_chars), 1)
    
    r_words = ref.split()
    h_words = hyp.split()
    w_dist = edit_dist(r_words, h_words)
    w_len = max(len(r_words), 1)
    return w_dist / w_len, c_dist / c_len

def evaluate(model, tok, samples, device, args, decoder=None):
    buckets = ["word/command", "sentence", "long-form"]
    bucket_stats = {b: {"w_err": 0.0, "w_len": 0, "snr_sum": 0.0, "n": 0} for b in buckets + ["overall"]}
    audit_data = []

    for s in tqdm(samples, desc="Evaluating", unit="sample"):
        path = s["audio_filepath"]
        ref_orig = s["text"]
        bucket = get_bucket(path, ref_orig)
        snr_val = compute_snr(path)
        
        chunks = process_audio(path, device)
        if not chunks or chunks[0][0] is None: continue

        greedy_parts, lm_parts = [], []
        for mel, mel_len in chunks:
            with torch.no_grad():
                logits, _ = model(mel, mel_len)
            
            ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
            greedy_parts.append(tok.decode(ids))
            
            if decoder:
                lp = F.log_softmax(logits, dim=-1).squeeze(0).cpu().float().numpy()
                lm_parts.append(normalise_text(decoder.decode(lp, beam_width=args.beam), clean=False))

        hyp_greedy = normalise_text(" ".join(greedy_parts), clean=False)
        ref_raw = normalise_text(ref_orig, clean=False)
        ref_cln = normalise_text(ref_raw, clean=True)
        hyp_cln = normalise_text(hyp_greedy, clean=True)
        
        # We use CLEAN WER as the primary metric for comparison
        w_dist, _ = wer_cer(ref_cln, hyp_cln)
        
        clm_w_dist = 0.0
        hyp_lm_str = None
        if decoder:
            hyp_lm_str = " ".join(lm_parts)
            clm_w_dist, _ = wer_cer(ref_cln, normalise_text(hyp_lm_str, clean=True))

        for b in [bucket, "overall"]:
            bucket_stats[b]["w_err"] += w_dist
            bucket_stats[b]["w_len"] += (clm_w_dist if decoder else 0)
            bucket_stats[b]["snr_sum"] += snr_val
            bucket_stats[b]["n"] += 1

        audit_data.append({
            "bucket": bucket,
            "filename": Path(path).name,
            "snr_db": round(snr_val, 2),
            "ref": ref_raw,
            "hyp_greedy": hyp_greedy,
            "hyp_lm": hyp_lm_str,
            "wer_cln": round(w_dist * 100, 2),
            "wer_clm": round(clm_w_dist * 100, 2) if decoder else None,
            "mixed_script": has_mixed_script(ref_raw),
            "duration": s.get("duration", 0)
        })

    results = {}
    for b, d in bucket_stats.items():
        if d["n"] > 0:
            results[b] = {
                "n": d["n"],
                "clean_wer": d["w_err"] / d["n"] * 100,
                "clm_wer": (d["w_len"] / d["n"] * 100) if decoder else None,
                "avg_snr": d["snr_sum"] / d["n"]
            }
    return results, audit_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--lm", type=str, default=None)
    parser.add_argument("--beam", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--output_csv", default="outputs/eval_audit.csv")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available(): device = torch.device("cuda")
    print(f"Device: {device}")

    # Load Model
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    v_size = state["ctc_head.weight"].shape[0]
    model = ConformerCTC(vocab_size=v_size).to(device).eval()
    model.load_state_dict(state, strict=False)
    
    tok = CharTokenizer()
    samples = [json.loads(l) for l in open(TEST_MANIFEST)]
    if args.count: samples = samples[:args.count]

    decoder = None
    if args.lm:
        from pyctcdecode import build_ctcdecoder
        decoder = build_ctcdecoder(tok.labels(), kenlm_model_path=args.lm, alpha=args.alpha, beta=args.beta)

    results, audit_data = evaluate(model, tok, samples, device, args, decoder)

    print(f"\n{'BUCKETED EVALUATION (WER %)':^85}")
    print("="*85)
    print(f"{'Bucket':<20} {'N':>5} {'CLEAN WER':>15} {'LM WER':>15} {'AVG SNR':>15}")
    print("-"*85)
    for b in ["word/command", "sentence", "long-form", "overall"]:
        if b in results:
            r = results[b]
            lm_s = f"{r['clm_wer']:14.2f}%" if r["clm_wer"] is not None else "              -"
            print(f"{b:<20} {r['n']:>5} {r['clean_wer']:14.2f}% {lm_s} {r['avg_snr']:12.1f} dB")
    
    import csv
    audit_data.sort(key=lambda x: x['wer_cln'], reverse=True)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline='') as f:
        w = csv.DictWriter(f, fieldnames=audit_data[0].keys())
        w.writeheader()
        w.writerows(audit_data)
    print(f"\nAudit CSV: {out_path}")

if __name__ == "__main__":
    main()
