#!/usr/bin/env python3
"""
Enhanced training script with periodic transcription testing
Tests model every N epochs to monitor actual transcription quality
"""
import torch
import json
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkanivani_asr import KonkaniVaniASR
from data.audio_processing.audio_processor import AudioProcessor
from data.audio_processing.text_tokenizer import KonkaniTokenizer


def test_transcription_quality(model, tokenizer, test_samples, device, epoch):
    """
    Test actual transcription quality on sample audio files
    
    Returns:
        dict with metrics: blank_prob, unique_tokens, sample_transcriptions
    """
    model.eval()
    audio_processor = AudioProcessor(sample_rate=16000, n_mels=80)
    
    results = {
        'epoch': epoch,
        'timestamp': datetime.now().isoformat(),
        'samples': []
    }
    
    blank_probs = []
    unique_token_counts = []
    
    with torch.no_grad():
        for sample in test_samples:
            try:
                # Load and process audio
                audio_path = sample['audio_filepath']
                waveform = audio_processor.load_audio(audio_path)
                features = audio_processor.compute_features(waveform, apply_augment=False)
                features_batch = features.unsqueeze(0).to(device)
                
                # Forward pass
                encoder_out, _ = model.encoder(features_batch)
                ctc_logits = model.ctc_head(encoder_out)
                
                # Get predictions
                probs = torch.softmax(ctc_logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
                # Analyze
                blank_prob = probs[0, :, tokenizer.blank_id].mean().item()
                unique_tokens = len(torch.unique(preds[0]))
                
                # Decode
                pred_tokens = preds[0].cpu().numpy()
                transcription = decode_ctc(pred_tokens, tokenizer)
                
                blank_probs.append(blank_prob)
                unique_token_counts.append(unique_tokens)
                
                results['samples'].append({
                    'audio': Path(audio_path).name,
                    'ground_truth': sample['text'][:100],
                    'prediction': transcription[:100] if transcription else '(empty)',
                    'blank_prob': blank_prob,
                    'unique_tokens': unique_tokens
                })
                
            except Exception as e:
                results['samples'].append({
                    'audio': Path(audio_path).name,
                    'error': str(e)
                })
    
    # Summary metrics
    results['avg_blank_prob'] = sum(blank_probs) / len(blank_probs) if blank_probs else 1.0
    results['avg_unique_tokens'] = sum(unique_token_counts) / len(unique_token_counts) if unique_token_counts else 0
    results['is_working'] = results['avg_blank_prob'] < 0.85 and results['avg_unique_tokens'] > 10
    
    return results


def decode_ctc(tokens, tokenizer):
    """CTC decoding"""
    decoded = []
    prev_token = None
    
    special_tokens = {'<pad>', '<blank>', '<sos>', '<eos>', '<unk>'}
    
    for token in tokens:
        if token == tokenizer.blank_id:
            prev_token = None
            continue
        
        if token == prev_token:
            continue
        
        if token in tokenizer.idx_to_char:
            char = tokenizer.idx_to_char[token]
            if char not in special_tokens:
                decoded.append(char)
        
        prev_token = token
    
    return ''.join(decoded)


def print_test_results(results):
    """Pretty print test results"""
    print("\n" + "="*80)
    print(f"TRANSCRIPTION TEST - EPOCH {results['epoch']}")
    print("="*80)
    
    print(f"\nOverall Metrics:")
    print(f"  Avg blank probability: {results['avg_blank_prob']:.1%}")
    print(f"  Avg unique tokens: {results['avg_unique_tokens']:.1f}")
    print(f"  Status: {'✅ WORKING!' if results['is_working'] else '❌ Not working yet'}")
    
    print(f"\nSample Transcriptions:")
    for i, sample in enumerate(results['samples'][:3], 1):
        if 'error' in sample:
            print(f"\n  [{i}] {sample['audio']}: Error - {sample['error']}")
        else:
            print(f"\n  [{i}] {sample['audio']}")
            print(f"      Ground truth: {sample['ground_truth']}")
            print(f"      Prediction:   {sample['prediction']}")
            print(f"      Blank prob: {sample['blank_prob']:.1%}, Tokens: {sample['unique_tokens']}")


def add_testing_callback_to_training():
    """
    Modify training script to add periodic testing
    """
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  TRAINING WITH PERIODIC TRANSCRIPTION TESTING                        ║
╚══════════════════════════════════════════════════════════════════════╝

This will test transcription quality every N epochs during training.

Configuration:
  • Test every: 5 epochs (configurable)
  • Test samples: 5 audio files
  • Metrics tracked:
    - Blank token probability (should decrease)
    - Unique tokens predicted (should increase)
    - Sample transcriptions (visual inspection)

Expected Progress:
  Epoch 1-10:   High blank prob (>95%), few tokens
  Epoch 10-20:  Blank prob drops to 80-90%, more tokens
  Epoch 20-30:  Blank prob <80%, recognizable words
  Epoch 30-50:  Blank prob <50%, good transcriptions

To integrate into your training:

1. Add to training_scripts/train_konkanivani_asr.py:

   from scripts.train_with_periodic_testing import test_transcription_quality, print_test_results
   
   # In ASRTrainer class, after validate():
   def train(self, num_epochs):
       for epoch in range(1, num_epochs + 1):
           # ... existing training code ...
           
           # Periodic testing
           if epoch % 5 == 0:  # Test every 5 epochs
               test_results = test_transcription_quality(
                   self.model, 
                   self.tokenizer,
                   self.test_samples,  # Load 5 test samples
                   self.device,
                   epoch
               )
               print_test_results(test_results)
               
               # Save test results
               results_file = self.checkpoint_dir / f'test_results_epoch_{epoch}.json'
               with open(results_file, 'w') as f:
                   json.dump(test_results, f, indent=2)
               
               # Early stopping if working well
               if test_results['is_working']:
                   print(f"\\n🎉 Model is working! Consider stopping or reducing LR.")

2. Or use this standalone script to test any checkpoint:

   python scripts/train_with_periodic_testing.py --checkpoint checkpoints/checkpoint_epoch_20.pt

""")


def test_checkpoint_standalone(checkpoint_path, num_samples=5):
    """Test a checkpoint standalone"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = KonkaniTokenizer('data/vocab.json')
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = KonkaniVaniASR(
        vocab_size=tokenizer.vocab_size,
        input_dim=80,
        d_model=256,
        encoder_layers=12,
        decoder_layers=6,
        num_heads=4,
        dropout=0.1
    )
    
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Load test samples
    test_manifest = 'data/konkani-asr-v0/splits/manifests/test.json'
    with open(test_manifest, 'r') as f:
        test_samples = [json.loads(line) for line in f][:num_samples]
    
    # Test
    epoch = checkpoint.get('epoch', 0)
    results = test_transcription_quality(model, tokenizer, test_samples, device, epoch)
    print_test_results(results)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to test')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of test samples')
    
    args = parser.parse_args()
    
    if args.checkpoint:
        test_checkpoint_standalone(args.checkpoint, args.num_samples)
    else:
        add_testing_callback_to_training()
