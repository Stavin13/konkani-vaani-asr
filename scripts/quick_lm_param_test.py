#!/usr/bin/env python3
"""
Quick LM Parameter Test
========================
Test different alpha (LM weight) and beta (word bonus) values quickly.
"""

import sys
sys.path.insert(0, '.')

from scripts.beam_search_decoder import BeamSearchDecoder, load_model, decode_audio
import json
from jiwer import cer
from tqdm import tqdm

def test_params(model, test_data, vocab_path, lm_path, alpha, beta, device='cpu'):
    """Test specific alpha/beta combination"""
    decoder = BeamSearchDecoder(vocab_path, lm_path, alpha, beta)
    
    predictions = []
    references = []
    
    for item in test_data:
        try:
            pred = decode_audio(model, item['audio_filepath'], decoder, beam_width=15, device=device)
            predictions.append(pred)
            references.append(item['text'])
        except:
            predictions.append("")
            references.append(item['text'])
    
    cer_score = cer(references, predictions) * 100
    return cer_score

def main():
    print("="*80)
    print("QUICK LM PARAMETER TEST")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model, _ = load_model('outputs/conformer_ctc_run1/best_conformer_ctc.pt')
    
    # Load test data (small subset for speed)
    print("Loading test data...")
    test_data = []
    with open('data/konkani-mega-dataset/manifests/test.json', 'r') as f:
        for i, line in enumerate(f):
            if i >= 20:  # Use 20 samples for quick test
                break
            test_data.append(json.loads(line))
    
    print(f"Testing on {len(test_data)} samples\n")
    
    # Test parameters
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6]
    betas = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    lm_path = 'models/language_models/konkani_4gram.binary'
    vocab_path = 'data/konkani-mega-dataset/vocab.json'
    
    results = []
    
    print("Testing parameter combinations...")
    print("-"*80)
    
    for alpha in alphas:
        for beta in betas:
            print(f"Testing alpha={alpha:.1f}, beta={beta:.1f}...", end=' ', flush=True)
            cer_score = test_params(model, test_data, vocab_path, lm_path, alpha, beta)
            results.append({
                'alpha': alpha,
                'beta': beta,
                'cer': cer_score
            })
            print(f"CER={cer_score:.2f}%")
    
    # Sort by CER
    results.sort(key=lambda x: x['cer'])
    
    print("\n" + "="*80)
    print("RESULTS (sorted by CER)")
    print("="*80)
    print(f"\n{'Rank':<6} {'Alpha':<8} {'Beta':<8} {'CER':<10}")
    print("-"*80)
    
    for i, r in enumerate(results[:10], 1):  # Show top 10
        print(f"{i:<6} {r['alpha']:<8.1f} {r['beta']:<8.1f} {r['cer']:>6.2f}%")
    
    print("\n" + "="*80)
    print("BEST PARAMETERS:")
    print("="*80)
    best = results[0]
    print(f"  alpha (LM weight): {best['alpha']}")
    print(f"  beta (word bonus): {best['beta']}")
    print(f"  CER: {best['cer']:.2f}%")
    
    # Compare with baseline (no LM)
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE")
    print("="*80)
    
    print("\nTesting beam search without LM...")
    decoder_no_lm = BeamSearchDecoder(vocab_path)
    predictions = []
    references = []
    for item in test_data:
        try:
            pred = decode_audio(model, item['audio_filepath'], decoder_no_lm, beam_width=15)
            predictions.append(pred)
            references.append(item['text'])
        except:
            predictions.append("")
            references.append(item['text'])
    
    baseline_cer = cer(references, predictions) * 100
    
    print(f"\nBeam (no LM):     {baseline_cer:.2f}% CER")
    print(f"Beam + LM (best): {best['cer']:.2f}% CER")
    
    improvement = ((baseline_cer - best['cer']) / baseline_cer) * 100
    if improvement > 0:
        print(f"\nImprovement: {improvement:.2f}% better with LM! ✓")
    else:
        print(f"\nWorse: {abs(improvement):.2f}% worse with LM ✗")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
