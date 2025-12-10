#!/usr/bin/env python3
"""
Test better trained checkpoints
"""

import torch
import json
import sys
import librosa
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkanivani_asr import KonkaniVaniASR

def test_checkpoint(checkpoint_path, vocab_path):
    """Test a specific checkpoint"""
    print(f"\n🔍 Testing checkpoint: {checkpoint_path.name}")
    
    # Load vocabulary
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    char2idx = vocab_data['char2idx']
    idx2char = vocab_data['idx2char']
    vocab_size = len(char2idx)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"📊 Checkpoint info:")
        print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"  Train Loss: {checkpoint.get('train_loss', 'Unknown'):.4f}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'Unknown'):.4f}")
        
        # Get model config
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        # Create model
        model = KonkaniVaniASR(
            vocab_size=vocab_size,
            input_dim=model_config.get('input_dim', 80),
            d_model=model_config.get('d_model', 256),
            encoder_layers=model_config.get('encoder_layers', 12),
            decoder_layers=model_config.get('decoder_layers', 6),
            num_heads=model_config.get('num_heads', 4),
            conv_kernel_size=model_config.get('conv_kernel_size', 31),
            dropout=model_config.get('dropout', 0.1)
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Test with real audio
        audio_dir = Path("data/konkani-asr-v0/data/processed_segments_diarized/audio_segments")
        audio_files = list(audio_dir.glob("*.wav"))
        
        if not audio_files:
            print("❌ No audio files found")
            return None
        
        test_file = audio_files[0]
        
        # Load and preprocess audio
        audio, sr = librosa.load(str(test_file), sr=16000)
        
        # Limit to 5 seconds for faster testing
        max_samples = 16000 * 5
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Extract mel features
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=80, hop_length=160, win_length=400, n_fft=400
        )
        mel = librosa.power_to_db(mel).T
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0)
        
        # Forward pass
        audio_length = torch.LongTensor([mel_tensor.size(1)])
        
        with torch.no_grad():
            ctc_logits, _ = model(mel_tensor, audio_length)
        
        # Analyze predictions
        ctc_predictions = ctc_logits.argmax(dim=-1).squeeze(0)
        
        # Count non-blank tokens
        blank_token = char2idx.get('<blank>', 1)
        non_blank_tokens = ctc_predictions[ctc_predictions != blank_token]
        blank_percentage = 100 * (ctc_predictions == blank_token).sum().item() / len(ctc_predictions)
        
        print(f"  🎯 Predictions: {len(ctc_predictions)} total, {len(non_blank_tokens)} non-blank")
        print(f"  🔇 Blank percentage: {blank_percentage:.1f}%")
        
        # Try to decode
        if len(non_blank_tokens) > 0:
            # Simple CTC decoding
            decoded_tokens = []
            prev_token = None
            for token in ctc_predictions.tolist():
                if token != blank_token and token != prev_token:
                    decoded_tokens.append(token)
                prev_token = token
            
            # Convert to text
            decoded_chars = []
            for token in decoded_tokens[:50]:  # First 50 tokens
                char = idx2char.get(str(token), '<unk>')
                if char not in ['<pad>', '<blank>', '<sos>', '<eos>', '<unk>']:
                    decoded_chars.append(char)
            
            decoded_text = ''.join(decoded_chars)
            print(f"  🔤 Decoded text: '{decoded_text}' ({len(decoded_tokens)} tokens)")
            
            return {
                'checkpoint': checkpoint_path.name,
                'epoch': checkpoint.get('epoch', 0),
                'train_loss': checkpoint.get('train_loss', float('inf')),
                'val_loss': checkpoint.get('val_loss', float('inf')),
                'non_blank_tokens': len(non_blank_tokens),
                'blank_percentage': blank_percentage,
                'decoded_text': decoded_text,
                'text_length': len(decoded_text)
            }
        else:
            print(f"  ❌ No non-blank tokens found")
            return {
                'checkpoint': checkpoint_path.name,
                'epoch': checkpoint.get('epoch', 0),
                'train_loss': checkpoint.get('train_loss', float('inf')),
                'val_loss': checkpoint.get('val_loss', float('inf')),
                'non_blank_tokens': 0,
                'blank_percentage': blank_percentage,
                'decoded_text': '',
                'text_length': 0
            }
            
    except Exception as e:
        print(f"❌ Error testing {checkpoint_path.name}: {e}")
        return None

def main():
    """Test multiple checkpoints to find the best one"""
    print("🔍 Testing Multiple Checkpoints")
    print("=" * 50)
    
    vocab_path = Path("data/vocab.json")
    
    # List of checkpoints to test (in order of preference)
    checkpoints_to_test = [
        # Later epochs (more training)
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_50.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_45.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_40.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_35.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_30.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_27.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_25.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_20.pt",
        "kaggle_asr_outputs/checkpoints/best_model.pt",
        # Current checkpoint
        "checkpoints/best_model_scripts1_fixed.pt",
    ]
    
    results = []
    
    for checkpoint_path_str in checkpoints_to_test:
        checkpoint_path = Path(checkpoint_path_str)
        if checkpoint_path.exists():
            result = test_checkpoint(checkpoint_path, vocab_path)
            if result:
                results.append(result)
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 CHECKPOINT COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    if results:
        # Sort by number of non-blank tokens (descending) and then by validation loss (ascending)
        results.sort(key=lambda x: (-x['non_blank_tokens'], x['val_loss']))
        
        print(f"{'Checkpoint':<35} {'Epoch':<6} {'Val Loss':<8} {'Non-blank':<10} {'Text Len':<8} {'Sample Text'}")
        print(f"{'-'*35} {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*20}")
        
        for result in results:
            sample_text = result['decoded_text'][:20] + '...' if len(result['decoded_text']) > 20 else result['decoded_text']
            print(f"{result['checkpoint']:<35} {result['epoch']:<6} {result['val_loss']:<8.3f} {result['non_blank_tokens']:<10} {result['text_length']:<8} '{sample_text}'")
        
        # Find best checkpoint
        best_checkpoint = results[0]
        print(f"\n🏆 Best checkpoint: {best_checkpoint['checkpoint']}")
        print(f"  📊 Epoch: {best_checkpoint['epoch']}")
        print(f"  📉 Val Loss: {best_checkpoint['val_loss']:.3f}")
        print(f"  🎯 Non-blank tokens: {best_checkpoint['non_blank_tokens']}")
        print(f"  🔤 Sample text: '{best_checkpoint['decoded_text'][:100]}'")
        
        if best_checkpoint['non_blank_tokens'] > 0:
            print(f"\n✅ Found working checkpoint! Use: {best_checkpoint['checkpoint']}")
        else:
            print(f"\n⚠️  All checkpoints produce blank outputs. Model may need more training.")
    else:
        print("❌ No checkpoints could be tested successfully")

if __name__ == "__main__":
    main()