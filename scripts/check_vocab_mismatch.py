#!/usr/bin/env python3
"""
Check Vocabulary Mismatch - Root Cause of 6% Accuracy
"""
import torch
import json
from pathlib import Path
from collections import Counter

def check_vocabulary_mismatch():
    """Check if vocabulary mismatch is causing poor performance"""
    
    print("="*70)
    print("VOCABULARY MISMATCH ANALYSIS")
    print("="*70)
    
    # 1. Load checkpoint and check model vocab size
    checkpoint_path = 'best_model (1).pt'
    
    if not Path(checkpoint_path).exists():
        print("❌ No checkpoint found")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', {})
    
    # Get model vocabulary size
    model_vocab_size = state_dict['ctc_head.weight'].shape[0]
    print(f"Model CTC vocabulary size: {model_vocab_size}")
    
    # Check if vocabulary is stored in checkpoint
    stored_vocab = checkpoint.get('vocab', {})
    print(f"Stored vocabulary size: {len(stored_vocab) if stored_vocab else 'None'}")
    
    # 2. Generate vocabulary from actual training data
    print("\nAnalyzing training data vocabulary...")
    
    manifest_path = Path('data/konkani-asr-v0/splits/manifests/train.json')
    if not manifest_path.exists():
        print("❌ No training manifest found")
        return
    
    # Count characters in training data
    char_counter = Counter()
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            text = sample.get('text', '')
            for char in text:
                char_counter[char] += 1
    
    # Create proper vocabulary
    data_vocab_size = len(char_counter) + 2  # +2 for <blank> and <unk>
    print(f"Training data vocabulary size needed: {data_vocab_size}")
    print(f"Unique characters in training data: {len(char_counter)}")
    
    # 3. Identify the mismatch
    print(f"\n{'='*70}")
    print("MISMATCH ANALYSIS:")
    print(f"{'='*70}")
    
    if model_vocab_size != data_vocab_size:
        print(f"❌ CRITICAL MISMATCH FOUND!")
        print(f"   Model expects: {model_vocab_size} tokens")
        print(f"   Data contains: {data_vocab_size} tokens")
        print(f"   Difference: {abs(model_vocab_size - data_vocab_size)}")
        
        if model_vocab_size < data_vocab_size:
            print(f"   → Model is TOO SMALL for the data")
            print(f"   → {data_vocab_size - model_vocab_size} characters will be mapped to <unk>")
        else:
            print(f"   → Model is TOO LARGE for the data")
            print(f"   → {model_vocab_size - data_vocab_size} tokens will never be predicted")
        
        print(f"\n🔥 THIS IS WHY YOU GET 6% ACCURACY!")
        print(f"   The model can't properly predict characters it wasn't sized for.")
        
    else:
        print(f"✅ Vocabulary sizes match!")
        print(f"   The issue is likely elsewhere (training process, hyperparameters, etc.)")
    
    # 4. Show character distribution
    print(f"\nMost common characters in training data:")
    for i, (char, count) in enumerate(char_counter.most_common(20)):
        print(f"  {i+3:2d}. '{char}' → {count:,} times")  # +3 because 0=blank, 1=unk, 2=first char
    
    # 5. Provide solution
    print(f"\n{'='*70}")
    print("SOLUTION:")
    print(f"{'='*70}")
    
    if model_vocab_size != data_vocab_size:
        print("You need to RETRAIN the model with correct vocabulary size:")
        print(f"1. Update model config: vocab_size = {data_vocab_size}")
        print(f"2. Retrain from scratch (can't resize existing model)")
        print(f"3. Or use transfer learning with a pretrained model")
        
        print(f"\nQuick fix options:")
        print(f"A) Use Whisper (pretrained, works immediately)")
        print(f"B) Use Wav2Vec2 + fine-tuning (faster than training from scratch)")
        print(f"C) Retrain your model with vocab_size={data_vocab_size}")
        
    else:
        print("Vocabulary size is correct. Other possible issues:")
        print("1. Training hyperparameters (learning rate, batch size)")
        print("2. Model architecture too complex")
        print("3. Data quality issues")
        print("4. Need more training epochs")
    
    return model_vocab_size, data_vocab_size

if __name__ == '__main__':
    check_vocabulary_mismatch()