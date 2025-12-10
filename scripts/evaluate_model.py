#!/usr/bin/env python3
"""
Evaluate ASR model with ground truth comparisons
"""
import torch
import json
import sys
from pathlib import Path
import argparse
from jiwer import wer, cer
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.test_best_model import ASRInference


def load_test_manifest(manifest_path, max_samples=None):
    """Load test manifest with ground truth"""
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line)
            samples.append(data)
    return samples


def evaluate_model(checkpoint_path, manifest_path, max_samples=10, vocab_path='data/vocab.json'):
    """Evaluate model on test set"""
    
    print("="*70)
    print("ASR MODEL EVALUATION")
    print("="*70)
    
    # Load model
    asr = ASRInference(checkpoint_path, vocab_path)
    
    # Load test samples
    print(f"\nLoading test samples from {manifest_path}...")
    test_samples = load_test_manifest(manifest_path, max_samples)
    print(f"Loaded {len(test_samples)} test samples\n")
    
    # Evaluate
    results = []
    predictions = []
    references = []
    
    print("="*70)
    print("EVALUATION RESULTS")
    print("="*70 + "\n")
    
    for i, sample in enumerate(test_samples, 1):
        audio_path = sample['audio_filepath']
        ground_truth = sample['text']
        
        try:
            # Transcribe
            transcription, tokens = asr.transcribe(audio_path)
            
            # Store results
            results.append({
                'audio': audio_path,
                'prediction': transcription,
                'ground_truth': ground_truth,
                'success': True
            })
            
            predictions.append(transcription if transcription else " ")
            references.append(ground_truth)
            
            # Print result
            print(f"Sample {i}/{len(test_samples)}")
            print(f"  Audio: {Path(audio_path).name}")
            print(f"  Ground Truth: {ground_truth[:100]}...")
            print(f"  Prediction:   {transcription if transcription else '(empty)'}...")
            print(f"  Raw Tokens:   {tokens[:5]}...")
            print()
            
        except Exception as e:
            print(f"✗ Error on {Path(audio_path).name}: {e}\n")
            results.append({
                'audio': audio_path,
                'error': str(e),
                'ground_truth': ground_truth,
                'success': False
            })
    
    # Calculate metrics
    print("="*70)
    print("METRICS")
    print("="*70)
    
    if predictions and references:
        try:
            wer_score = wer(references, predictions) * 100
            cer_score = cer(references, predictions) * 100
            
            print(f"\nWord Error Rate (WER): {wer_score:.2f}%")
            print(f"Character Error Rate (CER): {cer_score:.2f}%")
            
            # Success rate
            success_rate = sum(1 for r in results if r['success']) / len(results) * 100
            print(f"Success Rate: {success_rate:.1f}%")
            
            # Empty predictions
            empty_preds = sum(1 for p in predictions if not p.strip())
            print(f"Empty Predictions: {empty_preds}/{len(predictions)}")
            
        except Exception as e:
            print(f"\n⚠ Could not calculate metrics: {e}")
            print("This might be because all predictions are empty.")
    
    # Save results
    output_file = Path('outputs/evaluation_results.json')
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'checkpoint': checkpoint_path,
            'manifest': manifest_path,
            'num_samples': len(test_samples),
            'results': results,
            'metrics': {
                'wer': wer_score if 'wer_score' in locals() else None,
                'cer': cer_score if 'cer_score' in locals() else None
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate ASR model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='kaggle_asr_outputs/checkpoints/checkpoint_epoch_50.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default='data/konkani-asr-v0/splits/manifests/test.json',
        help='Path to test manifest'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        default='data/vocab.json',
        help='Path to vocabulary file'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=10,
        help='Maximum number of samples to evaluate'
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        args.checkpoint,
        args.manifest,
        args.max_samples,
        args.vocab
    )


if __name__ == '__main__':
    main()
