#!/usr/bin/env python3
"""
Fix DataParallel checkpoint loading and test all 50-epoch models
"""

import torch
import json
import sys
import librosa
import numpy as np
from pathlib import Path
from collections import OrderedDict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkanivani_asr import KonkaniVaniASR

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from DataParallel checkpoints"""
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def test_checkpoint_with_dataparallel_fix(checkpoint_path, vocab_path):
    """Test checkpoint with DataParallel fix"""
    print(f"\n🔍 Testing: {checkpoint_path.name}")
    
    try:
        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        char2idx = vocab_data['char2idx']
        idx2char = vocab_data['idx2char']
        vocab_size = len(char2idx)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract info
        epoch = checkpoint.get('epoch', 'Unknown')
        train_loss = checkpoint.get('train_loss', 'Unknown')
        val_loss = checkpoint.get('val_loss', 'Unknown')
        
        print(f"  📊 Epoch: {epoch}")
        print(f"  📈 Train Loss: {train_loss}")
        print(f"  📉 Val Loss: {val_loss}")
        
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
        
        # Fix DataParallel state dict
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('module.') for key in state_dict.keys()):
            print(f"  🔧 Fixing DataParallel state dict...")
            state_dict = remove_module_prefix(state_dict)
        
        # Load state dict
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"  ✅ Model loaded successfully!")
        
        # Test with real audio
        audio_dir = Path("data/konkani-asr-v0/data/processed_segments_diarized/audio_segments")
        audio_files = list(audio_dir.glob("*.wav"))
        
        if not audio_files:
            print(f"  ❌ No audio files found")
            return None
        
        test_file = audio_files[0]
        print(f"  🎵 Testing with: {test_file.name}")
        
        # Load and preprocess audio (3 seconds for speed)
        audio, sr = librosa.load(str(test_file), sr=16000, duration=3.0)
        
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
        blank_percentage = 100 * (ctc_predictions == blank_token).sum().item() / len(ctc_predictions)
        
        print(f"  🎯 Total predictions: {len(ctc_predictions)}")
        print(f"  🔤 Non-blank tokens: {len(non_blank_tokens)}")
        print(f"  🔇 Blank percentage: {blank_percentage:.1f}%")
        
        # Try to decode if we have non-blank tokens
        decoded_text = ""
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
            print(f"  🔤 Decoded text: '{decoded_text[:100]}{'...' if len(decoded_text) > 100 else ''}'")
        
        return {
            'checkpoint': checkpoint_path.name,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'non_blank_tokens': len(non_blank_tokens),
            'blank_percentage': blank_percentage,
            'decoded_text': decoded_text,
            'text_length': len(decoded_text),
            'status': 'working' if len(non_blank_tokens) > 0 else 'blank_only'
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {
            'checkpoint': checkpoint_path.name,
            'epoch': 'Unknown',
            'train_loss': 'Unknown',
            'val_loss': 'Unknown',
            'non_blank_tokens': 0,
            'blank_percentage': 100.0,
            'decoded_text': '',
            'text_length': 0,
            'status': 'error'
        }

def main():
    """Test all 50-epoch checkpoints with DataParallel fix"""
    print("🔍 Testing All 50-Epoch Checkpoints (DataParallel Fixed)")
    print("=" * 60)
    
    vocab_path = Path("data/vocab.json")
    
    # All available checkpoints (50 epochs each)
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
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_12.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_10.pt",
        "kaggle_asr_outputs/checkpoints/checkpoint_epoch_5.pt",
        # Also test the current one for comparison
        "checkpoints/best_model_scripts1_fixed.pt",
    ]
    
    results = []
    
    for checkpoint_path_str in checkpoints_to_test:
        checkpoint_path = Path(checkpoint_path_str)
        if checkpoint_path.exists():
            result = test_checkpoint_with_dataparallel_fix(checkpoint_path, vocab_path)
            if result:
                results.append(result)
        else:
            print(f"\n⚠️  Not found: {checkpoint_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE CHECKPOINT TEST RESULTS")
    print(f"{'='*80}")
    
    working_checkpoints = [r for r in results if r['status'] == 'working']
    blank_checkpoints = [r for r in results if r['status'] == 'blank_only']
    error_checkpoints = [r for r in results if r['status'] == 'error']
    
    print(f"✅ Working checkpoints: {len(working_checkpoints)}")
    print(f"🔇 Blank-only checkpoints: {len(blank_checkpoints)}")
    print(f"❌ Error checkpoints: {len(error_checkpoints)}")
    
    if working_checkpoints:
        print(f"\n🏆 WORKING CHECKPOINTS (sorted by non-blank tokens):")
        working_checkpoints.sort(key=lambda x: x['non_blank_tokens'], reverse=True)
        
        print(f"{'Checkpoint':<35} {'Epoch':<6} {'Val Loss':<10} {'Non-blank':<10} {'Text Len':<8} {'Sample Text'}")
        print(f"{'-'*35} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*30}")
        
        for result in working_checkpoints:
            sample_text = result['decoded_text'][:30] + '...' if len(result['decoded_text']) > 30 else result['decoded_text']
            val_loss_str = f"{result['val_loss']:.3f}" if isinstance(result['val_loss'], (int, float)) else str(result['val_loss'])[:10]
            print(f"{result['checkpoint']:<35} {result['epoch']:<6} {val_loss_str:<10} {result['non_blank_tokens']:<10} {result['text_length']:<8} '{sample_text}'")
        
        # Best checkpoint
        best = working_checkpoints[0]
        print(f"\n🎯 BEST CHECKPOINT: {best['checkpoint']}")
        print(f"  📊 Epoch: {best['epoch']}")
        print(f"  📉 Val Loss: {best['val_loss']}")
        print(f"  🎯 Non-blank tokens: {best['non_blank_tokens']}")
        print(f"  📏 Text length: {best['text_length']}")
        print(f"  🔤 Sample text: '{best['decoded_text'][:200]}'")
        
        # Save best checkpoint path for easy access
        best_checkpoint_path = f"kaggle_asr_outputs/checkpoints/{best['checkpoint']}"
        print(f"\n💾 To use this checkpoint:")
        print(f"   python scripts/production_asr_inference.py --checkpoint {best_checkpoint_path}")
        
    else:
        print(f"\n⚠️  NO WORKING CHECKPOINTS FOUND")
        print(f"This suggests the models may need:")
        print(f"  - More training epochs")
        print(f"  - Different learning rate")
        print(f"  - Better data preprocessing")
        print(f"  - Architecture adjustments")
    
    if blank_checkpoints:
        print(f"\n🔇 BLANK-ONLY CHECKPOINTS:")
        for result in blank_checkpoints:
            print(f"  📁 {result['checkpoint']} (Epoch {result['epoch']})")
    
    # Performance analysis
    if len(results) > 1:
        print(f"\n📈 TRAINING PROGRESS ANALYSIS:")
        
        # Sort by epoch for progression analysis
        epoch_results = [r for r in results if isinstance(r['epoch'], int)]
        epoch_results.sort(key=lambda x: x['epoch'])
        
        if len(epoch_results) > 1:
            print(f"  📊 Epoch progression:")
            for result in epoch_results:
                status_icon = "✅" if result['status'] == 'working' else "🔇" if result['status'] == 'blank_only' else "❌"
                print(f"    {status_icon} Epoch {result['epoch']:2d}: {result['non_blank_tokens']:3d} non-blank tokens")
    
    print(f"\n🎉 Checkpoint analysis complete!")

if __name__ == "__main__":
    main()