#!/usr/bin/env python3
"""
Test best_model.pt and calculate accuracy metrics
"""
import torch
import json
import sys
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import jiwer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.konkanivani_asr import KonkaniVaniASR
from data.audio_processing.audio_processor import AudioProcessor
from data.audio_processing.text_tokenizer import KonkaniTokenizer


def load_model(checkpoint_path, vocab_path, device):
    """Load model from checkpoint"""
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Print checkpoint info
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    # Load tokenizer
    tokenizer = KonkaniTokenizer(vocab_path)
    print(f"   Vocab size: {tokenizer.vocab_size}")
    
    # Get vocab size from actual model weights (most reliable)
    vocab_size = checkpoint['model_state_dict']['ctc_head.weight'].shape[0]
    print(f"   Model vocab size (from weights): {vocab_size}")
    
    # Get model config from checkpoint or use defaults
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        model_config = checkpoint['config']['model']
    else:
        # Default config
        model_config = {
            'input_dim': 80,
            'd_model': 128,
            'encoder_layers': 8,
            'decoder_layers': 6,
            'num_heads': 4,
            'conv_kernel_size': 31,
            'dropout': 0.3
        }
    
    # Create model with checkpoint's vocab size
    model = KonkaniVaniASR(
        vocab_size=vocab_size,
        **{k: v for k, v in model_config.items() if k != 'vocab_size'}
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("   ✅ Model loaded successfully!")
    
    return model, tokenizer


def transcribe_audio(model, tokenizer, audio_processor, audio_path, device):
    """Transcribe a single audio file"""
    try:
        # Process audio
        audio_features, duration = audio_processor.process_audio_file(str(audio_path))
        
        # Get length
        audio_length = audio_features.size(0)
        
        # Move to device
        audio_features = audio_features.unsqueeze(0).to(device)
        audio_length = torch.tensor([audio_length], device=device)
        
        # Inference
        with torch.no_grad():
            ctc_logits, _ = model(audio_features, audio_length)
        
        # Decode CTC output
        ctc_probs = torch.nn.functional.log_softmax(ctc_logits, dim=-1)
        predicted_tokens = torch.argmax(ctc_probs, dim=-1)[0].cpu().numpy()
        
        # Collapse repeats and remove blanks
        decoded_tokens = []
        prev_token = None
        for token in predicted_tokens:
            if token != tokenizer.blank_id and token != prev_token:
                decoded_tokens.append(token)
            prev_token = token
        
        # Convert to text
        transcription = tokenizer.decode(decoded_tokens)
        
        return transcription
    
    except Exception as e:
        print(f"   ❌ Error transcribing {audio_path}: {e}")
        return ""


def calculate_metrics(predictions, references):
    """Calculate WER, CER, and accuracy"""
    
    # Word Error Rate (WER)
    wer = jiwer.wer(references, predictions)
    
    # Character Error Rate (CER)
    cer = jiwer.cer(references, predictions)
    
    # Word Accuracy (1 - WER)
    word_accuracy = (1 - wer) * 100
    
    # Character Accuracy (1 - CER)
    char_accuracy = (1 - cer) * 100
    
    return {
        'wer': wer * 100,  # Convert to percentage
        'cer': cer * 100,
        'word_accuracy': word_accuracy,
        'char_accuracy': char_accuracy
    }


def test_model(checkpoint_path, test_manifest, vocab_path, device, max_samples=None):
    """Test model on test set"""
    
    print("="*80)
    print("🧪 TESTING ASR MODEL")
    print("="*80)
    
    # Load model
    model, tokenizer = load_model(checkpoint_path, vocab_path, device)
    
    # Load audio processor
    audio_processor = AudioProcessor()
    
    # Load test manifest
    print(f"\n📊 Loading test data: {test_manifest}")
    test_data = []
    with open(test_manifest, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    print(f"   Test samples: {len(test_data)}")
    
    # Test on samples
    print(f"\n🔬 Running inference...")
    predictions = []
    references = []
    
    for i, item in enumerate(tqdm(test_data, desc="Testing")):
        audio_path = item['audio_filepath']
        reference_text = item['text']
        
        # Transcribe
        prediction = transcribe_audio(model, tokenizer, audio_processor, audio_path, device)
        
        predictions.append(prediction)
        references.append(reference_text)
        
        # Show first few examples
        if i < 5:
            print(f"\n   Sample {i+1}:")
            print(f"   Reference: {reference_text}")
            print(f"   Predicted: {prediction}")
    
    # Calculate metrics
    print(f"\n📈 Calculating metrics...")
    metrics = calculate_metrics(predictions, references)
    
    # Print results
    print("\n" + "="*80)
    print("📊 TEST RESULTS")
    print("="*80)
    print(f"Checkpoint: {Path(checkpoint_path).name}")
    print(f"Test samples: {len(test_data)}")
    print("-"*80)
    print(f"Word Error Rate (WER):      {metrics['wer']:.2f}%")
    print(f"Character Error Rate (CER): {metrics['cer']:.2f}%")
    print("-"*80)
    print(f"Word Accuracy:              {metrics['word_accuracy']:.2f}%")
    print(f"Character Accuracy:         {metrics['char_accuracy']:.2f}%")
    print("="*80)
    
    # Interpretation
    print("\n💡 Interpretation:")
    if metrics['word_accuracy'] >= 90:
        print("   ✅ Excellent! Production-ready quality")
    elif metrics['word_accuracy'] >= 80:
        print("   ✅ Very Good! Usable for most applications")
    elif metrics['word_accuracy'] >= 70:
        print("   ⚠️  Good, but could be improved")
    elif metrics['word_accuracy'] >= 60:
        print("   ⚠️  Fair, needs more training")
    else:
        print("   ❌ Poor, significant improvement needed")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Test ASR model')
    parser.add_argument('--checkpoint', type=str, 
                       default='best_model (1).pt',
                       help='Path to checkpoint file')
    parser.add_argument('--test_manifest', type=str,
                       default='data/konkani-10k/test_manifest.json',
                       help='Path to test manifest')
    parser.add_argument('--vocab', type=str,
                       default='data/konkani-10k/vocab.json',
                       help='Path to vocabulary file')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of test samples (for quick testing)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/mps/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Test model
    metrics = test_model(
        args.checkpoint,
        args.test_manifest,
        args.vocab,
        device,
        args.max_samples
    )
    
    # Save results
    results_file = Path(args.checkpoint).stem + '_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")


if __name__ == '__main__':
    main()
