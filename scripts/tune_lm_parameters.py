#!/usr/bin/env python3
"""
Tuning CTC Decoder Parameters (Optimized & Fixed)
==============================================
"""

import sys
import json
import time
import argparse
import itertools
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from scripts.beam_search_decoder import BeamSearchDecoder, load_model, extract_features
from scripts.clean_eval_wer import normalise_text, wer_cer, get_bucket

def evaluate_config(cached_probs, decoder, config, desc=""):
    """Evaluate a specific decoder configuration on cached log_probs."""
    total_w_dist = 0.0
    for ref, log_probs in tqdm(cached_probs, desc=desc, leave=False):
        if config['type'] == 'greedy':
            hyp = decoder.greedy_decode(log_probs)
        else:
            hyp = decoder.beam_search_decode(
                log_probs, 
                beam_width=config.get('beam_width', 10),
                lm_weight=config.get('lm_weight', 1.0),
                length_norm=config.get('length_norm', 0.0),
                space_reward=config.get('space_reward', 0.0)
            )
        
        hyp = normalise_text(hyp, clean=True)
        w_dist, _ = wer_cer(ref, hyp)
        total_w_dist += w_dist
    
    return total_w_dist / len(cached_probs) * 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_manifest", default="data/konkani-combined/manifests/val.json")
    parser.add_argument("--lm4", default="models/language_models/konkani_4gram.binary")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available(): device = torch.device("cuda")
    
    # Load Model
    checkpoint_path = BASE / "outputs/conformer_ctc_run1/best_conformer_ctc.pt"
    vocab_path = BASE / "data/konkani-10k/vocab.json"
    model, _ = load_model(str(checkpoint_path), device)
    
    # 1. Cache log_probs (EXCLUDING LONG-FORM)
    print(f"Caching log_probs (Sentence/Word only) for up to {args.count} samples...")
    cached_probs = []
    with open(BASE / args.val_manifest) as f:
        for line in f:
            if len(cached_probs) >= args.count: break
            s = json.loads(line)
            bucket = get_bucket(s['audio_filepath'], s['text'])
            
            # Skip long-form to get honest tuning
            if bucket == "long-form": continue
            
            ref = normalise_text(s['text'], clean=True)
            features = extract_features(s['audio_filepath'], device)
            with torch.no_grad():
                encoder_out, _ = model.encoder(features)
                logits = model.ctc_head(encoder_out)
                lp = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0).cpu()
            cached_probs.append((ref, lp))
    
    print(f"Cached {len(cached_probs)} clean samples.")

    # 2. Results
    results = []

    # Greedy Baseline
    decoder_no_lm = BeamSearchDecoder(str(vocab_path))
    wer_greedy = evaluate_config(cached_probs, decoder_no_lm, {'type': 'greedy'}, desc="Greedy")
    results.append(("Greedy", wer_greedy))

    # Beam No-LM Tuning
    print("\nTuning Beam (no-LM)...")
    best_wer_no_lm = 100
    for lw in [0.0, 0.1, 0.2]: # Using length_norm as our tuning knob
        wer = evaluate_config(cached_probs, decoder_no_lm, 
                              {'type': 'beam', 'beam_width': 10, 'length_norm': lw}, 
                              desc=f"Beam LN={lw}")
        best_wer_no_lm = min(best_wer_no_lm, wer)
    results.append(("Beam (no-LM)", best_wer_no_lm))

    # Beam + 4gram Tuning
    if Path(args.lm4).exists():
        print("\nTuning Beam + 4-gram...")
        decoder_4g = BeamSearchDecoder(str(vocab_path), lm_path=args.lm4)
        best_wer_4g = 100
        for weight in [0.5, 1.0, 1.5]:
            wer = evaluate_config(cached_probs, decoder_4g, 
                                  {'type': 'beam', 'beam_width': 10, 'lm_weight': weight, 'length_norm': 0.1}, 
                                  desc=f"4G Weight={weight}")
            best_wer_4g = min(best_wer_4g, wer)
        results.append(("Beam + 4gram", best_wer_4g))

    # Final Table
    print("\n" + "="*40)
    print(f"{'Decoding Strategy':<25} {'WER %':>10}")
    print("-" * 40)
    for name, wer in results:
        print(f"{name:<25} {wer:10.2f}%")
    print("=" * 40)

if __name__ == "__main__":
    main()
