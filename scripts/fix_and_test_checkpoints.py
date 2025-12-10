#!/usr/bin/env python3
"""
Fix checkpoint loading issues and test properly
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

def safe_format_loss(loss_value):
    """Safely format loss value"""
    if isinstance(loss_value, (int, float)):
        return f"{loss_value:.4f}"
    else:
        return str(loss_value)

def test_checkpoint_safe(checkpoint_path, vocab_path):
    """Safely test a checkpoint with better error handling"""
    print(f"\n🔍 Testing: {checkpoint_path.name}")
    
    try:
        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        char2idx = vocab_data['char2idx']
        idx2char = vocab_data['idx2char']
        vocab_size = len(char2idx)
        
        # Load checkpoint with error handling
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Safely extract values
        epoch = checkpoint.get('epoch', 'Unknown')
        train_loss = checkpoint.get('train_loss', 'Unknown')
        val_loss = checkpoint.get('val_loss', 'Unknown')
        
        print(f"  📊 Epoch: {epoch}")
        print(f"  📈 Train Loss: {safe_format_loss(train_loss)}")
        print(f"  📉 Val Loss: {safe_format_loss(val_loss)}")
        
        # Get model config
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        # Create model with safe defaults
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
        
        # Quick test with dummy data first
        dummy_input = torch.randn(1, 100, 80)
        dummy_length = torch.LongTensor([100])
        
        with torch.no_grad():
            ctc_logits, _ = model(dummy_input, dummy_length)
        
        # Check if model produces varied outputs
        predictions = ctc_logits.argmax(dim=-1).squeeze(0)
        unique_predictions = torch.unique(predictions)
        
        print(f"  🎯 Unique predictions: {len(unique_predictions)} different tokens")
        
        # Test with real audio if model shows promise
        if len(unique_predictions) > 1:
            print(f"  ✅ Model shows varied predictions - testing with real audio")
            
            # Load real audio
            audio_dir = Path("data/konkani-asr-v0/data/processed_segments_diarized/audio_segments")
            audio_files = list(audio_dir.glob("*.wav"))
            
            if audio_files:
                test_file = audio_files[0]
                
                # Load and preprocess audio (shorter for speed)
                audio, sr = librosa.load(str(test_file), sr=16000, duration=3.0)  # 3 seconds
                
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
                blank_token = char2idx.get('<blank>', 1)
                non_blank_tokens = ctc_predictions[ctc_predictions != blank_token]
                
                print(f"  🎵 Real audio test: {len(non_blank_tokens)} non-blank tokens")
                
                if len(non_blank_tokens) > 0:
                    # Try to decode
                    decoded_tokens = []
                    prev_token = None
                    for token in ctc_predictions.tolist():
                        if token != blank_token and token != prev_token:
                            decoded_tokens.append(token)
                        prev_token = token
                    
                    # Convert to text
                    decoded_chars = []
                    for token in decoded_tokens[:30]:  # First 30 tokens
                        char = idx2char.get(str(token), '<unk>')
                        if char not in ['<pad>', '<blank>', '<sos>', '<eos>', '<unk>']:
                            decoded_chars.append(char)
                    
                    decoded_text = ''.join(decoded_chars)
                    print(f"  🔤 Sample output: '{decoded_text}'")
                    
                    return {
                        'checkpoint': checkpoint_path.name,
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'non_blank_tokens': len(non_blank_tokens),
                        'decoded_text': decoded_text,
                        'status': 'working'
                    }
                else:
                    print(f"  ❌ Still producing only blanks with real audio")
        else:
            print(f"  ❌ Model only predicts single token (likely blank)")
        
        return {
            'checkpoint': checkpoint_path.name,
            'epoch': epoch,
            'val_loss': val_loss,
            'non_blank_tokens': 0,
            'decoded_text': '',
            'status': 'blank_only'
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {
            'checkpoint': checkpoint_path.name,
            'epoch': 'Unknown',
            'val_loss': float('inf'),
            'non_blank_tokens': 0,
            'decoded_text': '',
            'status': 'error'
        }

def main():
    """Test checkpoints with better error handling"""
    print("🔍 Testing Checkpoints (Safe Mode)")
    print("=" * 50)
    
    vocab_path = Path("data/vocab.json")
    
    # Test available checkpoints
    checkpoints_to_test = [
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_50.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_45.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_40.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_35.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_30.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_27.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_25.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_20.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_15.pt",
        "checkpoints/best_model_scripts1_fixed.pt",
        "checkpoints/checkpoint_epoch_15.pt",
    ]
    
    results = []
    
    for checkpoint_path_str in checkpoints_to_test:
        checkpoint_path = Path(checkpoint_path_str)
        if checkpoint_path.exists():
            result = test_checkpoint_safe(checkpoint_path, vocab_path)
            results.append(result)
        else:
            print(f"\n⚠️  Not found: {checkpoint_path.name}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 CHECKPOINT TEST RESULTS")
    print(f"{'='*80}")
    
    working_checkpoints = [r for r in results if r['status'] == 'working']
    blank_checkpoints = [r for r in results if r['status'] == 'blank_only']
    error_checkpoints = [r for r in results if r['status'] == 'error']
    
    print(f"✅ Working checkpoints: {len(working_checkpoints)}")
    print(f"🔇 Blank-only checkpoints: {len(blank_checkpoints)}")
    print(f"❌ Error checkpoints: {len(error_checkpoints)}")
    
    if working_checkpoints:
        print(f"\n🏆 WORKING CHECKPOINTS:")
        for result in working_checkpoints:
            print(f"  📁 {result['checkpoint']} (Epoch {result['epoch']})")
            print(f"     🔤 Sample: '{result['decoded_text'][:50]}'")
    
    if blank_checkpoints:
        print(f"\n🔇 BLANK-ONLY CHECKPOINTS:")
        for result in blank_checkpoints:
            print(f"  📁 {result['checkpoint']} (Epoch {result['epoch']})")
    
    if error_checkpoints:
        print(f"\n❌ ERROR CHECKPOINTS:")
        for result in error_checkpoints:
            print(f"  📁 {result['checkpoint']}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if working_checkpoints:
        best = working_checkpoints[0]
        print(f"  🎯 Use checkpoint: {best['checkpoint']}")
        print(f"  🚀 This model produces actual text output!")
    else:
        print(f"  ⚠️  No working checkpoints found")
        print(f"  📚 Model needs more training or different approach")
        print(f"  🔧 Consider:")
        print(f"     - Training for more epochs (current best: 10 epochs)")
        print(f"     - Adjusting learning rate")
        print(f"     - Checking data preprocessing")
        print(f"     - Using a different model architecture")

if __name__ == "__main__":
    main()