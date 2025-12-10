#!/usr/bin/env python3
"""
Debug ASR inference to understand why transcriptions are empty
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

def debug_asr_inference():
    """Debug the ASR inference process"""
    
    checkpoint_path = Path("checkpoints/best_model_scripts1_fixed.pt")
    vocab_path = Path("data/vocab.json")
    audio_dir = Path("data/konkani-asr-v0/data/processed_segments_diarized/audio_segments")
    
    print("🔍 Debugging ASR Inference")
    print("=" * 50)
    
    # Load vocabulary
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    char2idx = vocab_data['char2idx']
    idx2char = vocab_data['idx2char']
    vocab_size = len(char2idx)
    
    print(f"📚 Vocabulary: {vocab_size} characters")
    print(f"🔤 Special tokens:")
    for token in ['<pad>', '<blank>', '<sos>', '<eos>', '<unk>']:
        if token in char2idx:
            print(f"  {token}: {char2idx[token]}")
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded")
    
    # Get a test audio file
    audio_files = list(audio_dir.glob("*.wav"))
    if not audio_files:
        print("❌ No audio files found")
        return
    
    test_file = audio_files[0]
    print(f"\n🎵 Testing with: {test_file.name}")
    
    # Load and preprocess audio
    audio, sr = librosa.load(str(test_file), sr=16000)
    print(f"📊 Audio shape: {audio.shape}, duration: {len(audio)/sr:.2f}s")
    
    # Extract mel features
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=80, hop_length=160, win_length=400, n_fft=400
    )
    mel = librosa.power_to_db(mel).T  # (time, n_mels)
    mel_tensor = torch.FloatTensor(mel).unsqueeze(0)  # Add batch dimension
    
    print(f"📈 Mel features shape: {mel_tensor.shape}")
    print(f"📊 Mel stats: min={mel_tensor.min():.2f}, max={mel_tensor.max():.2f}, mean={mel_tensor.mean():.2f}")
    
    # Forward pass
    audio_length = torch.LongTensor([mel_tensor.size(1)])
    
    with torch.no_grad():
        ctc_logits, attn_logits = model(mel_tensor, audio_length)
    
    print(f"\n🔍 Model outputs:")
    print(f"  CTC logits shape: {ctc_logits.shape}")
    print(f"  CTC logits stats: min={ctc_logits.min():.2f}, max={ctc_logits.max():.2f}")
    
    if attn_logits is not None:
        print(f"  Attention logits shape: {attn_logits.shape}")
    
    # Analyze CTC predictions
    ctc_probs = torch.softmax(ctc_logits, dim=-1)
    ctc_predictions = ctc_logits.argmax(dim=-1).squeeze(0)
    
    print(f"\n🎯 CTC Analysis:")
    print(f"  Predictions shape: {ctc_predictions.shape}")
    print(f"  Raw predictions (first 20): {ctc_predictions[:20].tolist()}")
    
    # Check what tokens are being predicted most
    unique_tokens, counts = torch.unique(ctc_predictions, return_counts=True)
    print(f"\n📊 Token frequency:")
    for token, count in zip(unique_tokens[:10], counts[:10]):
        char = idx2char.get(str(token.item()), f'<{token.item()}>')
        percentage = 100 * count.item() / len(ctc_predictions)
        print(f"  Token {token.item()} ('{char}'): {count.item()} times ({percentage:.1f}%)")
    
    # Check if blank token is dominating
    blank_token = char2idx.get('<blank>', 1)
    blank_count = (ctc_predictions == blank_token).sum().item()
    blank_percentage = 100 * blank_count / len(ctc_predictions)
    print(f"\n🔇 Blank token analysis:")
    print(f"  Blank token index: {blank_token}")
    print(f"  Blank predictions: {blank_count}/{len(ctc_predictions)} ({blank_percentage:.1f}%)")
    
    # Try different decoding approaches
    print(f"\n🔤 Decoding attempts:")
    
    # Method 1: Simple greedy with blank removal
    decoded_tokens = []
    prev_token = None
    for token in ctc_predictions.tolist():
        if token != blank_token and token != prev_token:
            decoded_tokens.append(token)
        prev_token = token
    
    decoded_chars = [idx2char.get(str(token), '<unk>') for token in decoded_tokens[:50]]
    decoded_text1 = ''.join([c for c in decoded_chars if c not in ['<pad>', '<blank>', '<sos>', '<eos>', '<unk>']])
    print(f"  Method 1 (greedy): '{decoded_text1}' ({len(decoded_tokens)} tokens)")
    
    # Method 2: Only non-blank tokens
    non_blank_tokens = ctc_predictions[ctc_predictions != blank_token]
    if len(non_blank_tokens) > 0:
        decoded_chars2 = [idx2char.get(str(token.item()), '<unk>') for token in non_blank_tokens[:50]]
        decoded_text2 = ''.join([c for c in decoded_chars2 if c not in ['<pad>', '<blank>', '<sos>', '<eos>', '<unk>']])
        print(f"  Method 2 (non-blank): '{decoded_text2}' ({len(non_blank_tokens)} tokens)")
    else:
        print(f"  Method 2 (non-blank): No non-blank tokens found!")
    
    # Method 3: Check confidence scores
    max_probs, _ = ctc_probs.max(dim=-1)
    confident_positions = max_probs.squeeze(0) > 0.5  # High confidence positions
    confident_predictions = ctc_predictions[confident_positions]
    
    if len(confident_predictions) > 0:
        decoded_chars3 = [idx2char.get(str(token.item()), '<unk>') for token in confident_predictions[:50]]
        decoded_text3 = ''.join([c for c in decoded_chars3 if c not in ['<pad>', '<blank>', '<sos>', '<eos>', '<unk>']])
        print(f"  Method 3 (confident): '{decoded_text3}' ({len(confident_predictions)} tokens)")
    else:
        print(f"  Method 3 (confident): No confident predictions!")
    
    # Check model training state
    print(f"\n🧠 Model analysis:")
    print(f"  Training epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"  Training loss: {checkpoint.get('train_loss', 'Unknown')}")
    print(f"  Validation loss: {checkpoint.get('val_loss', 'Unknown')}")
    
    # Check if model is actually trained
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Check parameter statistics
    param_stats = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_stats.append({
                'name': name,
                'shape': list(param.shape),
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item()
            })
    
    print(f"\n📊 Parameter statistics (first 5 layers):")
    for stat in param_stats[:5]:
        print(f"  {stat['name']}: mean={stat['mean']:.4f}, std={stat['std']:.4f}")
    
    # Test with different audio preprocessing
    print(f"\n🔧 Testing different preprocessing:")
    
    # Try without log transform
    mel_linear = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=80, hop_length=160, win_length=400, n_fft=400
    ).T
    mel_linear_tensor = torch.FloatTensor(mel_linear).unsqueeze(0)
    
    with torch.no_grad():
        ctc_logits_linear, _ = model(mel_linear_tensor, audio_length)
    
    ctc_predictions_linear = ctc_logits_linear.argmax(dim=-1).squeeze(0)
    non_blank_linear = ctc_predictions_linear[ctc_predictions_linear != blank_token]
    
    print(f"  Linear mel: {len(non_blank_linear)} non-blank tokens")
    
    if len(non_blank_linear) > 0:
        decoded_chars_linear = [idx2char.get(str(token.item()), '<unk>') for token in non_blank_linear[:20]]
        decoded_text_linear = ''.join([c for c in decoded_chars_linear if c not in ['<pad>', '<blank>', '<sos>', '<eos>', '<unk>']])
        print(f"  Linear mel result: '{decoded_text_linear}'")

if __name__ == "__main__":
    debug_asr_inference()