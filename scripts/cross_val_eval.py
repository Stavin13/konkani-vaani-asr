#!/usr/bin/env python3
import json
import random
import sys
import argparse
import traceback
from pathlib import Path
from collections import defaultdict

import torch
import torchaudio
import soundfile as sf
import numpy as np
from tqdm import tqdm
import openpyxl
from openpyxl.styles import PatternFill, Font

# Paths
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from scripts.clean_eval_wer import (
    CharTokenizer, ConformerCTC, normalise_text, wer_cer, _get_vad
)

# Configuration
CHECKPOINT = BASE / "outputs/conformer_ctc_run1/best_conformer_ctc.pt"
DATA_DIR   = BASE / "data/cross_val_newsonair/newsonair_konkani_external_aligned_lab_02-09-2021_06-55"
DATA_JSON  = DATA_DIR / "data.json"
LM_PATH    = BASE / "models/language_models/konkani_4gram_news.arpa"
UNIGRAMS   = BASE / "models/language_models/unigrams.txt"
OUT_EXCEL  = BASE / "outputs/cross_val_analysis.xlsx"

VAD_THRESHOLD_SEC = 8.0

def _wav_to_mel(wav_1d: torch.Tensor, device: torch.device):
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=80, n_fft=400, hop_length=160, win_length=400
    ).to(device)
    mel = mel_fn(wav_1d.unsqueeze(0).to(device))
    mel = mel.transpose(1, 2)
    mel = torch.log(mel + 1e-9)
    mel_len = torch.tensor([mel.size(1)], device=device)
    return mel.float(), mel_len

def process_audio_v4(path: str, device: torch.device):
    try:
        if not Path(path).exists(): return None
        wav, sr = sf.read(path)
        wav = torch.from_numpy(wav).float()
        if wav.ndim > 1: wav = wav.mean(-1)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
        
        duration_sec = wav.size(0) / 16000
        if duration_sec <= VAD_THRESHOLD_SEC:
            return [_wav_to_mel(wav, device)]

        vad_model, utils = _get_vad()
        timestamps = utils[0](wav, vad_model, sampling_rate=16000, min_silence_duration_ms=400)
        if not timestamps: return [_wav_to_mel(wav, device)]

        results = []
        for ts in timestamps:
            start, end = ts['start'], ts['end']
            if (end - start) < 400: continue
            results.append(_wav_to_mel(wav[start:end], device))
        return results if results else [_wav_to_mel(wav, device)]
    except: return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    tok = CharTokenizer()
    print(f"Vocab loaded. Blank index: {tok.blank_id}")

    model = ConformerCTC(vocab_size=tok.vocab_size).to(device).eval()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

    unigrams = None
    if UNIGRAMS.exists():
        with open(UNIGRAMS, encoding="utf-8") as f:
            unigrams = [l.strip() for l in f if l.strip()]

    labels = []
    for i in range(tok.vocab_size):
        c = tok.idx2char.get(i, f"_{i}_")
        if i == tok.blank_id: labels.append("")
        elif c in ["<pad>", "<sos>", "<eos>", "<unk>"] or c.startswith("<"):
            labels.append(f"_{i}_")
        else: labels.append(c)

    decoder = None
    if LM_PATH.exists():
        from pyctcdecode import build_ctcdecoder
        decoder = build_ctcdecoder(labels, kenlm_model_path=str(LM_PATH), unigrams=unigrams, alpha=0.6, beta=0.0)
        print("Decoder built with alpha=0.6, beta=0.0, beam=50")

    with open(DATA_JSON, 'r') as f: data = json.load(f)
    session_map = defaultdict(list)
    for i in data: session_map[i["audioFilename"].rsplit('_', 1)[0]].append(i)
    sessions = sorted(list(session_map.keys()))
    random.seed(42); random.shuffle(sessions)
    folds = [[] for _ in range(args.k)]
    for i, s in enumerate(sessions): folds[i % args.k].extend(session_map[s])

    results = []
    all_rows = []
    for f in range(args.k):
        print(f"\n--- Fold {f+1} ---")
        samples = folds[f][:args.max_samples]
        e, l = 0, 0
        for s in tqdm(samples):
            mels = process_audio_v4(str(DATA_DIR / s["audioFilename"]), device)
            if not mels: continue
            
            p_parts_lm = []
            p_parts_greedy = []
            for m, ml in mels:
                with torch.no_grad(): logits, _ = model(m, ml)
                
                # Merge index 0 (pad) and 1 (blank) into 1 for pyctcdecode
                if decoder:
                    lp = torch.nn.functional.log_softmax(logits, dim=-1)
                    lp_merged = lp.clone()
                    lp_merged[:, :, 1] = torch.logsumexp(lp[:, :, [0, 1]], dim=-1)
                    lp_merged[:, :, 0] = -1e9 # Mask pad
                    lp_merged_np = lp_merged.squeeze(0).cpu().numpy()
                    # Use higher beam and tuned alpha
                    p_parts_lm.append(decoder.decode(lp_merged_np, beam_width=50))
                
                ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
                p_parts_greedy.append(tok.decode(ids))
            
            hyp_lm = normalise_text(" ".join(p_parts_lm), clean=False) if decoder else ""
            hyp_greedy = normalise_text(" ".join(p_parts_greedy), clean=False)
            
            hyp = hyp_lm if hyp_lm else hyp_greedy
            ref = normalise_text(s["text"], clean=False)
            w, _ = wer_cer(normalise_text(ref, clean=True), normalise_text(hyp, clean=True))
            e += w; l += 1
            all_rows.append({"f":f+1, "a":s["audioFilename"], "r":ref, "h":hyp, "w":w})
            if l == 1: 
                print(f"\nREF:    {ref}")
                print(f"GREEDY: {hyp_greedy}")
                print(f"LM:     {hyp_lm}")
                print(f"WER:    {w:.1%}\n")
        results.append(e/l if l>0 else 1.0)

    print(f"\nMEAN WER: {np.mean(results):.2%}")
    wb = openpyxl.Workbook(); ws = wb.active
    ws.append(["Fold", "Audio", "Ref", "Hyp", "WER"])
    for r in all_rows: ws.append([r["f"], r["a"], r["r"], r["h"], r["w"]])
    wb.save(OUT_EXCEL)

if __name__ == "__main__": main()
