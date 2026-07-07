#!/usr/bin/env python3
"""
Test and Compare Beam Search Improvements
==========================================
Compare greedy vs beam search vs beam search + LM on test set.
"""

import torch
import json
import time
from pathlib import Path
from jiwer import wer, cer
from tqdm import tqdm
import argparse

from beam_search_decoder import BeamSearchDecoder, load_model, decode_audio


def load_test_manifest(manifest_path, max_samples=None):
    """Load test manifest"""
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
            if max_samples and len(data) >= max_samples:
                break
    return data


def evaluate_strategy(model, test_data, decoder, beam_width, device, strategy_name):
    """
    Evaluate a decoding strategy
    
    Args:
        model: ASR model
        test_data: List of test samples
        decoder: BeamSearchDecoder instance
        beam_width: Beam width (None for greedy)
        device: Device for computation
        strategy_name: Name for display
    
    Returns:
        results: Dict with predictions, time, cer, wer
    """
    print(f"\n{strategy_name}...")
    
    predictions = []
    references = []
    start_time = time.time()
    
    for item in tqdm(test_data, desc=strategy_name):
        audio_path = item['audio_filepath']
        reference = item['text']
        
        try:
            # Decode
            prediction = decode_audio(model, audio_path, decoder, beam_width, device)
            predictions.append(prediction)
            references.append(reference)
        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            predictions.append("")
            references.append(reference)
    
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    cer_score = cer(references, predictions) * 100
    wer_score = wer(references, predictions) * 100
    
    # Average WER (per-sample)
    individual_wers = [wer([r], [p]) for r, p in zip(references, predictions)]
    avg_wer_score = (sum(individual_wers) / len(individual_wers)) * 100
    
    return {
        'predictions': predictions,
        'references': references,
        'time': elapsed_time,
        'cer': cer_score,
        'wer': wer_score,
        'avg_wer': avg_wer_score
    }


def print_comparison_table(results_dict, num_samples):
    """Print formatted comparison table"""
    print("\n" + "="*80)
    print("BEAM SEARCH IMPROVEMENT COMPARISON")
    print("="*80)
    print(f"\nTested on {num_samples} samples\n")
    
    # Main results table
    print(f"{'Strategy':<30} {'CER':<10} {'WER':<10} {'Time (s)':<12} {'Speed':<10}")
    print("-"*80)
    
    baseline_time = results_dict['greedy']['time']
    baseline_cer = results_dict['greedy']['cer']
    
    for strategy, results in results_dict.items():
        cer_val = results['cer']
        wer_val = results['wer']
        time_val = results['time']
        speed = baseline_time / time_val if time_val > 0 else 0
        
        print(f"{strategy:<30} {cer_val:>6.2f}%   {wer_val:>6.2f}% ({results['avg_wer']:>6.2f}% avg)   "
              f"{time_val:>8.2f}     {speed:>6.2f}x")
    
    # Improvement table
    print("\n" + "="*80)
    print("RELATIVE IMPROVEMENTS (vs Greedy Baseline)")
    print("="*80)
    print(f"\n{'Strategy':<30} {'CER Improvement':<20} {'WER Improvement':<20} {'Avg WER Improv.'}")
    print("-"*80)
    
    baseline_wer = results_dict['greedy']['wer']
    
    for strategy, results in results_dict.items():
        if strategy == 'greedy':
            continue
        
        cer_val = results['cer']
        wer_val = results['wer']
        
        cer_improvement = ((baseline_cer - cer_val) / baseline_cer) * 100
        wer_improvement = ((baseline_wer - wer_val) / baseline_wer) * 100
        
        avg_wer_val = results['avg_wer']
        baseline_avg_wer = results_dict['greedy']['avg_wer']
        avg_wer_improvement = ((baseline_avg_wer - avg_wer_val) / baseline_avg_wer) * 100
        
        print(f"{strategy:<30} {cer_improvement:>6.2f}% better      {wer_improvement:>6.2f}% better      {avg_wer_improvement:>6.2f}% better")
    
    print("\n" + "="*80)


def print_sample_outputs(results_dict, num_samples=5):
    """Print sample predictions for comparison"""
    print("\n" + "="*80)
    print(f"SAMPLE PREDICTIONS (first {num_samples} samples)")
    print("="*80)
    
    references = results_dict['greedy']['references'][:num_samples]
    
    for i in range(min(num_samples, len(references))):
        print(f"\nSample {i+1}:")
        print(f"  Reference:  {references[i]}")
        
        for strategy, results in results_dict.items():
            pred = results['predictions'][i] if i < len(results['predictions']) else ""
            print(f"  {strategy:<12}: {pred}")


def save_results(results_dict, output_path):
    """Save detailed results to JSON"""
    output = {
        'summary': {},
        'detailed_predictions': {}
    }
    
    for strategy, results in results_dict.items():
        output['summary'][strategy] = {
            'cer': results['cer'],
            'wer': results['wer'],
            'avg_wer': results['avg_wer'],
            'time': results['time']
        }
        
        output['detailed_predictions'][strategy] = [
            {
                'reference': ref,
                'prediction': pred
            }
            for ref, pred in zip(results['references'], results['predictions'])
        ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")

def save_results_csv(results_dict, output_path):
    """Save side-by-side results to CSV for Excel comparison"""
    import csv
    strategies = list(results_dict.keys())
    references = results_dict['greedy']['references']
    
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        header = ['Reference'] + strategies
        writer.writerow(header)
        
        for i in range(len(references)):
            row = [references[i]]
            for s in strategies:
                row.append(results_dict[s]['predictions'][i])
            writer.writerow(row)
    
    print(f"Excel-ready CSV results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test beam search improvements')
    parser.add_argument('--model', type=str, default='outputs/conformer_ctc_run1/best_conformer_ctc.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, default='data/konkani-10k/vocab.json',
                        help='Path to vocab.json')
    parser.add_argument('--test-manifest', type=str, default='data/konkani-combined/manifests/test.json',
                        help='Path to test manifest')
    parser.add_argument('--lm-3gram', type=str, default='models/language_models/konkani_3gram.binary',
                        help='Path to 3-gram LM')
    parser.add_argument('--lm-4gram', type=str, default='models/language_models/konkani_4gram.binary',
                        help='Path to 4-gram LM')
    parser.add_argument('--unigrams', type=str, default='models/language_models/unigrams.txt',
                        help='Path to unigrams.txt')
    parser.add_argument('--beam_width', type=int, default=15, help='Beam width')
    parser.add_argument('--alpha', type=float, default=1.0, help='LM weight')
    parser.add_argument('--beta', type=float, default=1.0, help='Word bonus')
    parser.add_argument('--max-samples', type=int, default=None, help='Max test samples (for quick testing)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--output', type=str, default='outputs/beam_search_comparison.json',
                        help='Output JSON path')
    
    args = parser.parse_args()
    
    print("="*80)
    print("BEAM SEARCH + LANGUAGE MODEL EVALUATION")
    print("="*80)
    
    # Check files exist
    required_files = [args.model, args.vocab, args.test_manifest]
    for path in required_files:
        if not Path(path).exists():
            print(f"ERROR: {path} not found!")
            return
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model, vocab_size = load_model(args.model, args.device)
    
    # Load test data
    print(f"Loading test data from {args.test_manifest}...")
    test_data = load_test_manifest(args.test_manifest, args.max_samples)
    print(f"  Loaded {len(test_data)} test samples")
    
    # Results dictionary
    results = {}
    
    # 1. Greedy decoding (baseline)
    print("\n" + "="*80)
    print("STRATEGY 1: Greedy Decoding (Baseline)")
    print("="*80)
    decoder_greedy = BeamSearchDecoder(args.vocab)
    results['greedy'] = evaluate_strategy(
        model, test_data, decoder_greedy, None, args.device, "Greedy"
    )
    
    # 2. Beam search (no LM)
    print("\n" + "="*80)
    print(f"STRATEGY 2: Beam Search (width={args.beam_width}, no LM)")
    print("="*80)
    decoder_beam = BeamSearchDecoder(args.vocab)
    results['beam_no_lm'] = evaluate_strategy(
        model, test_data, decoder_beam, args.beam_width, args.device, "Beam (no LM)"
    )
    
    # 3. Beam search + 3-gram LM
    if Path(args.lm_3gram).exists():
        print("\n" + "="*80)
        print(f"STRATEGY 3: Beam Search + 3-gram LM")
        print("="*80)
        decoder_3gram = BeamSearchDecoder(args.vocab, args.lm_3gram, args.unigrams, args.alpha, args.beta)
        results['beam_3gram'] = evaluate_strategy(
            model, test_data, decoder_3gram, args.beam_width, args.device, "Beam + 3-gram"
        )
    else:
        print(f"\nSkipping 3-gram LM (not found: {args.lm_3gram})")
    
    # 4. Beam search + 4-gram LM
    if Path(args.lm_4gram).exists():
        print("\n" + "="*80)
        print(f"STRATEGY 4: Beam Search + 4-gram LM")
        print("="*80)
        decoder_4gram = BeamSearchDecoder(args.vocab, args.lm_4gram, args.unigrams, args.alpha, args.beta)
        results['beam_4gram'] = evaluate_strategy(
            model, test_data, decoder_4gram, args.beam_width, args.device, "Beam + 4-gram"
        )
    else:
        print(f"\nSkipping 4-gram LM (not found: {args.lm_4gram})")
    
    # Print comparison
    print_comparison_table(results, len(test_data))
    print_sample_outputs(results, num_samples=5)
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_results(results, args.output)
    
    # Also save CSV
    csv_path = args.output.replace('.json', '.csv')
    save_results_csv(results, csv_path)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
