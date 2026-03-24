#!/usr/bin/env python3
"""
Tune Language Model Parameters
===============================
Grid search to find optimal beam_width, alpha (LM weight), and beta (word bonus).
"""

import torch
import json
import time
from pathlib import Path
from jiwer import cer
from tqdm import tqdm
import argparse
import itertools

from beam_search_decoder import BeamSearchDecoder, load_model, decode_audio


def load_validation_data(manifest_path, max_samples=None):
    """Load validation manifest"""
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
            if max_samples and len(data) >= max_samples:
                break
    return data


def evaluate_params(model, val_data, vocab_path, lm_path, beam_width, alpha, beta, device):
    """
    Evaluate a specific parameter combination
    
    Returns:
        cer_score: Character error rate
        time_taken: Inference time
    """
    # Create decoder with these parameters
    decoder = BeamSearchDecoder(vocab_path, lm_path, alpha, beta)
    
    predictions = []
    references = []
    start_time = time.time()
    
    for item in val_data:
        audio_path = item['audio_filepath']
        reference = item['text']
        
        try:
            prediction = decode_audio(model, audio_path, decoder, beam_width, device)
            predictions.append(prediction)
            references.append(reference)
        except Exception as e:
            predictions.append("")
            references.append(reference)
    
    elapsed_time = time.time() - start_time
    cer_score = cer(references, predictions) * 100
    
    return cer_score, elapsed_time


def grid_search(model, val_data, vocab_path, lm_path, device, 
                beam_widths, alphas, betas):
    """
    Perform grid search over parameters
    
    Returns:
        results: List of dicts with parameters and scores
    """
    results = []
    total_combinations = len(beam_widths) * len(alphas) * len(betas)
    
    print(f"\nGrid search over {total_combinations} parameter combinations...")
    print(f"  beam_width: {beam_widths}")
    print(f"  alpha (LM weight): {alphas}")
    print(f"  beta (word bonus): {betas}")
    print()
    
    with tqdm(total=total_combinations, desc="Grid search") as pbar:
        for beam_width, alpha, beta in itertools.product(beam_widths, alphas, betas):
            pbar.set_description(f"beam={beam_width}, α={alpha:.1f}, β={beta:.1f}")
            
            cer_score, time_taken = evaluate_params(
                model, val_data, vocab_path, lm_path,
                beam_width, alpha, beta, device
            )
            
            results.append({
                'beam_width': beam_width,
                'alpha': alpha,
                'beta': beta,
                'cer': cer_score,
                'time': time_taken
            })
            
            pbar.update(1)
    
    return results


def print_results_table(results):
    """Print formatted results table"""
    print("\n" + "="*90)
    print("PARAMETER TUNING RESULTS")
    print("="*90)
    print(f"\n{'Beam':<8} {'Alpha':<8} {'Beta':<8} {'CER':<10} {'Time (s)':<12} {'Rank':<6}")
    print("-"*90)
    
    # Sort by CER (best first)
    sorted_results = sorted(results, key=lambda x: x['cer'])
    
    for i, result in enumerate(sorted_results, 1):
        print(f"{result['beam_width']:<8} {result['alpha']:<8.1f} {result['beta']:<8.1f} "
              f"{result['cer']:>6.2f}%   {result['time']:>8.2f}     #{i}")
    
    print("\n" + "="*90)
    
    # Best parameters
    best = sorted_results[0]
    print("\nBEST PARAMETERS:")
    print(f"  beam_width: {best['beam_width']}")
    print(f"  alpha (LM weight): {best['alpha']}")
    print(f"  beta (word bonus): {best['beta']}")
    print(f"  CER: {best['cer']:.2f}%")
    print(f"  Time: {best['time']:.2f}s")
    
    return best


def analyze_parameter_effects(results):
    """Analyze individual parameter effects"""
    print("\n" + "="*90)
    print("PARAMETER EFFECT ANALYSIS")
    print("="*90)
    
    # Group by beam_width
    print("\nEffect of Beam Width (averaged over alpha, beta):")
    beam_groups = {}
    for r in results:
        bw = r['beam_width']
        if bw not in beam_groups:
            beam_groups[bw] = []
        beam_groups[bw].append(r['cer'])
    
    for bw in sorted(beam_groups.keys()):
        avg_cer = sum(beam_groups[bw]) / len(beam_groups[bw])
        print(f"  beam_width={bw:2d}: CER={avg_cer:.2f}%")
    
    # Group by alpha
    print("\nEffect of Alpha/LM Weight (averaged over beam_width, beta):")
    alpha_groups = {}
    for r in results:
        a = r['alpha']
        if a not in alpha_groups:
            alpha_groups[a] = []
        alpha_groups[a].append(r['cer'])
    
    for a in sorted(alpha_groups.keys()):
        avg_cer = sum(alpha_groups[a]) / len(alpha_groups[a])
        print(f"  alpha={a:.1f}: CER={avg_cer:.2f}%")
    
    # Group by beta
    print("\nEffect of Beta/Word Bonus (averaged over beam_width, alpha):")
    beta_groups = {}
    for r in results:
        b = r['beta']
        if b not in beta_groups:
            beta_groups[b] = []
        beta_groups[b].append(r['cer'])
    
    for b in sorted(beta_groups.keys()):
        avg_cer = sum(beta_groups[b]) / len(beta_groups[b])
        print(f"  beta={b:.1f}: CER={avg_cer:.2f}%")


def save_tuning_results(results, best_params, output_path):
    """Save tuning results to JSON"""
    output = {
        'best_parameters': best_params,
        'all_results': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Tune LM parameters')
    parser.add_argument('--model', type=str, default='outputs/conformer_ctc_run1/best_conformer_ctc.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, default='data/konkani-mega-dataset/vocab.json',
                        help='Path to vocab.json')
    parser.add_argument('--val-manifest', type=str, default='data/konkani-mega-dataset/manifests/val.json',
                        help='Path to validation manifest')
    parser.add_argument('--lm', type=str, default='models/language_models/konkani_4gram.binary',
                        help='Path to KenLM binary')
    parser.add_argument('--beam-widths', type=int, nargs='+', default=[10, 15, 20],
                        help='Beam widths to try')
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.5, 1.0, 1.5],
                        help='Alpha (LM weight) values to try')
    parser.add_argument('--betas', type=float, nargs='+', default=[0.0, 1.0, 2.0],
                        help='Beta (word bonus) values to try')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='Max validation samples (use subset for speed)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--output', type=str, default='outputs/lm_parameter_tuning.json',
                        help='Output JSON path')
    
    args = parser.parse_args()
    
    print("="*90)
    print("LANGUAGE MODEL PARAMETER TUNING")
    print("="*90)
    
    # Check files exist
    required_files = [args.model, args.vocab, args.val_manifest, args.lm]
    for path in required_files:
        if not Path(path).exists():
            print(f"ERROR: {path} not found!")
            return
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model, vocab_size = load_model(args.model, args.device)
    
    # Load validation data
    print(f"Loading validation data from {args.val_manifest}...")
    val_data = load_validation_data(args.val_manifest, args.max_samples)
    print(f"  Using {len(val_data)} validation samples")
    
    # Grid search
    results = grid_search(
        model, val_data, args.vocab, args.lm, args.device,
        args.beam_widths, args.alphas, args.betas
    )
    
    # Print results
    best_params = print_results_table(results)
    analyze_parameter_effects(results)
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_tuning_results(results, best_params, args.output)
    
    print("\n" + "="*90)
    print("TUNING COMPLETE")
    print("="*90)
    print("\nNext steps:")
    print(f"1. Use these parameters in your inference code:")
    print(f"   beam_width={best_params['beam_width']}")
    print(f"   alpha={best_params['alpha']}")
    print(f"   beta={best_params['beta']}")
    print(f"2. Test on full test set with: python scripts/test_beam_search_improvements.py")
    print(f"3. Update production inference to use beam search + LM")


if __name__ == "__main__":
    main()
