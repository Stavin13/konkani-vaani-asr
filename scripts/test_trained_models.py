#!/usr/bin/env python3
"""
Test the trained Translation and Emotion models
"""
import torch
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkani_custom_emotion import create_custom_emotion_model
from models.konkani_custom_translator import create_custom_translation_model


def test_emotion_model():
    """Test emotion model on test set"""
    print("\n" + "="*70)
    print("TESTING EMOTION MODEL")
    print("="*70)
    
    # Load checkpoint
    checkpoint_path = Path('checkpoints/emotion_model/emotion_model_mac.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    vocab_size = checkpoint['config']['vocab_size']
    model = create_custom_emotion_model(vocab_size=vocab_size, num_emotions=7)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    vocab = checkpoint['vocab']
    emotion_map = checkpoint['emotion_map']
    reverse_emotion_map = {v: k for k, v in emotion_map.items()}
    
    print(f"Model loaded: {checkpoint['config']['num_params']:,} parameters")
    
    # Load test data
    test_path = Path('data/emotion_data/splits/test.json')
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Test samples: {len(test_data)}")
    
    # Test on samples
    correct = 0
    total = 0
    
    print("\nSample predictions:")
    for i, item in enumerate(test_data[:10]):
        text = item['text']
        true_emotion = item['emotion']
        
        # Tokenize
        tokens = [vocab.get(char, vocab['<UNK>']) for char in text]
        if len(tokens) < 100:
            attention_mask = [1] * len(tokens) + [0] * (100 - len(tokens))
            tokens = tokens + [vocab['<PAD>']] * (100 - len(tokens))
        else:
            tokens = tokens[:100]
            attention_mask = [1] * 100
        
        input_ids = torch.tensor(tokens).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask)
            pred_idx = torch.argmax(logits, dim=-1).item()
            pred_emotion = reverse_emotion_map[pred_idx]
        
        correct += (pred_emotion == true_emotion)
        total += 1
        
        status = "✓" if pred_emotion == true_emotion else "✗"
        print(f"{status} [{i+1}] {text[:50]}...")
        print(f"    True: {true_emotion:10s} | Pred: {pred_emotion:10s}")
    
    # Full test accuracy
    print("\nCalculating full test accuracy...")
    for item in test_data:
        text = item['text']
        true_emotion = item['emotion']
        
        tokens = [vocab.get(char, vocab['<UNK>']) for char in text]
        if len(tokens) < 100:
            attention_mask = [1] * len(tokens) + [0] * (100 - len(tokens))
            tokens = tokens + [vocab['<PAD>']] * (100 - len(tokens))
        else:
            tokens = tokens[:100]
            attention_mask = [1] * 100
        
        input_ids = torch.tensor(tokens).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask)
            pred_idx = torch.argmax(logits, dim=-1).item()
            pred_emotion = reverse_emotion_map[pred_idx]
        
        correct += (pred_emotion == true_emotion)
        total += 1
    
    accuracy = 100 * correct / total
    print(f"\n✓ Test Accuracy: {accuracy:.2f}% ({correct}/{total})")


def test_translation_model():
    """Test translation model"""
    print("\n" + "="*70)
    print("TESTING TRANSLATION MODEL")
    print("="*70)
    
    # Load checkpoint
    checkpoint_path = Path('checkpoints/translation_model/translation_model_mac.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    src_vocab_size = checkpoint['config']['src_vocab_size']
    tgt_vocab_size = checkpoint['config']['tgt_vocab_size']
    model = create_custom_translation_model(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    reverse_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    
    print(f"Model loaded: {checkpoint['config']['num_params']:,} parameters")
    
    # Load test data
    data_path = Path('data/translation_data/konkani_english_translated.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    n_train = int(len(all_data) * 0.8)
    n_val = int(len(all_data) * 0.1)
    test_data = all_data[n_train+n_val:]
    
    print(f"Test samples: {len(test_data)}")
    
    # Test on samples
    print("\nSample translations:")
    for i, item in enumerate(test_data[:5]):
        konkani = item['konkani']
        english = item['english']
        
        # Tokenize source
        src_tokens = [src_vocab.get(char, src_vocab['<UNK>']) for char in konkani]
        if len(src_tokens) < 150:
            src_tokens = src_tokens + [src_vocab['<PAD>']] * (150 - len(src_tokens))
        else:
            src_tokens = src_tokens[:150]
        
        src = torch.tensor(src_tokens).unsqueeze(0)
        
        # Simple greedy decoding
        tgt_tokens = [tgt_vocab['<SOS>']]
        max_len = 150
        
        with torch.no_grad():
            for _ in range(max_len):
                tgt_input = torch.tensor(tgt_tokens).unsqueeze(0)
                
                # Pad to max_len
                if tgt_input.size(1) < max_len:
                    pad_len = max_len - tgt_input.size(1)
                    tgt_input = torch.cat([tgt_input, torch.zeros(1, pad_len, dtype=torch.long)], dim=1)
                
                tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1))
                output = model(src, tgt_input, tgt_mask=tgt_mask)
                
                next_token = torch.argmax(output[0, len(tgt_tokens)-1, :]).item()
                
                if next_token == tgt_vocab['<EOS>'] or next_token == tgt_vocab['<PAD>']:
                    break
                
                tgt_tokens.append(next_token)
        
        # Decode
        predicted = ''.join([reverse_tgt_vocab.get(t, '') for t in tgt_tokens[1:]])
        
        print(f"\n[{i+1}]")
        print(f"  Konkani: {konkani[:80]}...")
        print(f"  True:    {english[:80]}...")
        print(f"  Pred:    {predicted[:80]}...")


def main():
    print("\n" + "="*70)
    print("TEST TRAINED MODELS")
    print("="*70)
    
    # Test emotion model
    test_emotion_model()
    
    # Test translation model
    test_translation_model()
    
    print("\n" + "="*70)
    print("TESTING COMPLETE BOSS! 🎉")
    print("="*70)


if __name__ == '__main__':
    main()
