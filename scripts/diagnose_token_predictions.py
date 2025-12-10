#!/usr/bin/env python3
"""
Diagnose what tokens the model is actually predicting
"""

import torch
import json
import sys
import librosa
import numpy as np
from pathlib import Path
from collections import OrderedDict, Counter

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

def diagnose_predictions():
    """Diagnose what the model is actually predicting"""
    
    # Paths
    checkpoint_path = Path("kaggle_asr_outputs/checkpoints/checkpoint_epoch_45.pt")
    vocab_path = Path("data/vocab.json")
    audio_dir = Path("data/konkani-asr-v0/data/processed_segments_diarized/audio_segments")
    
    print("🔍 Diagnosing Token Predictions")
    print("=" * 50)
    
    # Load vocabulary
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    char2idx = vocab_data['char2idx']
    idx2char = vocab_data['idx2char']
    vocab_size = len(char2idx)
    
    print(f"📚 Vocabulary size: {vocab_size}")
    
    # Show vocabulary structure
    print(f"\n🔤 Vocabulary analysis:")
    special_tokens = []
    devanagari_chars = []
    latin_chars = []
    other_chars = []
    
    for char, idx in char2idx.items():
        if char.startswith('<') and char.endswith('>'):
            special_tokens.append((char, idx))
        elif '\u0900' <= char <= '\u097F':  # Devanagari range
            devanagari_chars.append((char, idx))
        elif 'a' <= char.lower() <= 'z':
            latin_chars.append((char, idx))
        else:
            other_chars.append((char, idx))
    
    print(f"  Special tokens: {len(special_tokens)}")
    print(f"  Devanagari chars: {len(devanagari_chars)}")
    print(f"  Latin chars: {len(latin_chars)}")
    print(f"  Other chars: {len(other_chars)}")
    
    # Show some examples
    print(f"\n📝 Sample characters:")
    print(f"  Special: {[t[0] for t in special_tokens[:5]]}")
    print(f"  Devanagari: {[t[0] for t in devanagari_chars[:10]]}")
    print(f"  Latin: {[t[0] for t in latin_chars[:10]]}")
    print(f"  Other: {[t[0] for t in other_chars[:10]]}")
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
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
        state_dict = remove_module_prefix(state_dict)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"\n✅ Model loaded (Epoch {checkpoint.get('epoch')})")
    
    # Test with audio files
    audio_files = list(audio_dir.glob("*.wav"))[:3]
    
    all_predictions = []
    
    for audio_file in audio_files:
        print(f"\n🎵 Testing: {audio_file.name}")
        
        # Load and preprocess audio
        audio, sr = librosa.load(str(audio_file), sr=16000, duration=5.0)
        
        # Normalized mel-spectrogram (best performing method)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=80, hop_length=160, win_length=400, n_fft=400
        )
        mel = librosa.power_to_db(mel).T
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0)
        
        # Forward pass
        audio_length = torch.LongTensor([mel_tensor.size(1)])
        
        with torch.no_grad():
            ctc_logits, _ = model(mel_tensor, audio_length)
        
        # Get predictions
        predictions = ctc_logits.argmax(dim=-1).squeeze(0).tolist()
        all_predictions.extend(predictions)
        
        # Analyze this file's predictions
        pred_counter = Counter(predictions)
        
        print(f"  📊 Total predictions: {len(predictions)}")
        print(f"  🎯 Unique tokens: {len(pred_counter)}")
        
        # Show top predicted tokens
        print(f"  🔝 Top 10 predicted tokens:")
        for token, count in pred_counter.most_common(10):
            char = idx2char.get(str(token), f'<{token}>')
            percentage = 100 * count / len(predictions)
            print(f"    Token {token:3d} ('{char}'): {count:3d} times ({percentage:5.1f}%)")
        
        # Show non-blank predictions
        blank_token = char2idx.get('<blank>', 1)
        non_blank_preds = [p for p in predictions if p != blank_token]
        
        print(f"  🔤 Non-blank predictions: {len(non_blank_preds)}")
        if non_blank_preds:
            non_blank_counter = Counter(non_blank_preds)
            print(f"    Non-blank tokens:")
            for token, count in non_blank_counter.most_common(5):
                char = idx2char.get(str(token), f'<{token}>')
                print(f"      Token {token:3d} ('{char}'): {count} times")
    
    # Overall analysis
    print(f"\n{'='*60}")
    print(f"📊 OVERALL PREDICTION ANALYSIS")
    print(f"{'='*60}")
    
    overall_counter = Counter(all_predictions)
    
    print(f"🎯 Total predictions across all files: {len(all_predictions)}")
    print(f"🔢 Unique tokens predicted: {len(overall_counter)}")
    
    # Categorize predictions
    blank_token = char2idx.get('<blank>', 1)
    blank_count = overall_counter.get(blank_token, 0)
    blank_percentage = 100 * blank_count / len(all_predictions)
    
    print(f"\n🔇 Blank token analysis:")
    print(f"  Blank token ('{idx2char.get(str(blank_token), '<blank>')}', idx={blank_token}): {blank_count} ({blank_percentage:.1f}%)")
    
    # Non-blank analysis
    non_blank_total = len(all_predictions) - blank_count
    print(f"\n🔤 Non-blank token analysis:")
    print(f"  Total non-blank predictions: {non_blank_total} ({100-blank_percentage:.1f}%)")
    
    if non_blank_total > 0:
        print(f"\n🏆 Top 20 non-blank tokens:")
        print(f"{'Token':<6} {'Char':<10} {'Count':<8} {'%':<8} {'Type'}")
        print(f"{'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*15}")
        
        for token, count in overall_counter.most_common(25):
            if token == blank_token:
                continue
            
            char = idx2char.get(str(token), f'<{token}>')
            percentage = 100 * count / len(all_predictions)
            
            # Categorize character
            if char.startswith('<') and char.endswith('>'):
                char_type = "Special"
            elif '\u0900' <= char <= '\u097F':
                char_type = "Devanagari"
            elif 'a' <= char.lower() <= 'z':
                char_type = "Latin"
            elif char == ' ':
                char_type = "Space"
            else:
                char_type = "Other"
            
            print(f"{token:<6} {char:<10} {count:<8} {percentage:<8.2f} {char_type}")
            
            if len([t for t, c in overall_counter.most_common() if t != blank_token]) >= 20:
                break
    
    # Check if model is learning
    print(f"\n🧠 Learning Analysis:")
    
    # Check if predictions are diverse
    entropy = -sum((count/len(all_predictions)) * np.log2(count/len(all_predictions)) 
                   for count in overall_counter.values())
    max_entropy = np.log2(vocab_size)
    normalized_entropy = entropy / max_entropy
    
    print(f"  📈 Prediction entropy: {entropy:.2f} / {max_entropy:.2f} ({normalized_entropy:.1%})")
    
    if normalized_entropy < 0.1:
        print(f"  ⚠️  Very low entropy - model is not diverse")
    elif normalized_entropy < 0.3:
        print(f"  🔶 Low entropy - model is somewhat repetitive")
    else:
        print(f"  ✅ Good entropy - model shows diversity")
    
    # Check if model predicts meaningful characters
    meaningful_chars = 0
    for token, count in overall_counter.items():
        if token == blank_token:
            continue
        char = idx2char.get(str(token), '')
        if char and not (char.startswith('<') and char.endswith('>')):
            meaningful_chars += count
    
    meaningful_percentage = 100 * meaningful_chars / len(all_predictions)
    print(f"  🔤 Meaningful characters: {meaningful_percentage:.1f}%")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if blank_percentage > 95:
        print(f"  🔧 Model is predicting too many blanks - consider:")
        print(f"     - Lower CTC blank penalty during training")
        print(f"     - Different learning rate schedule")
        print(f"     - More training epochs")
    
    if meaningful_percentage < 5:
        print(f"  📚 Model is not learning meaningful characters - consider:")
        print(f"     - Check training data quality")
        print(f"     - Verify audio-text alignment")
        print(f"     - Review model architecture")
    
    if normalized_entropy < 0.2:
        print(f"  🎲 Model lacks diversity - consider:")
        print(f"     - Increase model capacity")
        print(f"     - Add regularization")
        print(f"     - Check for overfitting")

if __name__ == "__main__":
    diagnose_predictions()