#!/usr/bin/env python3
"""
Test the best translation model with sample inputs
"""
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkani_custom_translator import create_custom_translation_model


def test_translation_model():
    """Test translation model with sample inputs"""
    print("\n" + "="*70)
    print("TESTING BEST TRANSLATION MODEL")
    print("="*70)
    
    # Load best checkpoint
    checkpoint_path = Path('checkpoints/translation_model/translation_model_best.pt')
    
    if not checkpoint_path.exists():
        print(f"\n❌ Model not found at: {checkpoint_path}")
        return
    
    print(f"\nLoading model from: {checkpoint_path}")
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
    
    print(f"✓ Model loaded: {checkpoint['config']['num_params']:,} parameters")
    print(f"✓ Trained for: {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"✓ Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    # Test samples
    test_samples = [
        "नमस्कार",
        "हांव बरो आसां",
        "तुका कसें आसा?",
        "धन्यवाद",
        "हें बरें आसा",
        "मका भूक लागली",
        "हांव वता",
        "माझें नांव स्टेविन",
        "हें सुंदर आसा",
        "आयज",
    ]
    
    print("\n" + "="*70)
    print("SAMPLE TRANSLATIONS")
    print("="*70)
    
    for i, konkani_text in enumerate(test_samples, 1):
        # Tokenize source
        src_tokens = [src_vocab.get(char, src_vocab['<UNK>']) for char in konkani_text]
        max_len = 150
        
        if len(src_tokens) < max_len:
            src_tokens = src_tokens + [src_vocab['<PAD>']] * (max_len - len(src_tokens))
        else:
            src_tokens = src_tokens[:max_len]
        
        src = torch.tensor(src_tokens).unsqueeze(0)
        
        # Greedy decoding
        tgt_tokens = [tgt_vocab['<SOS>']]
        
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
                
                # Stop if too long
                if len(tgt_tokens) > 100:
                    break
        
        # Decode
        predicted = ''.join([reverse_tgt_vocab.get(t, '') for t in tgt_tokens[1:]])
        
        print(f"\n[{i}]")
        print(f"  Konkani:  {konkani_text}")
        print(f"  English:  {predicted if predicted else '(empty)'}")
    
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("\nEnter Konkani text to translate (or 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\nKonkani: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Bye boss! 👋")
                break
            
            if not user_input:
                continue
            
            # Tokenize
            src_tokens = [src_vocab.get(char, src_vocab['<UNK>']) for char in user_input]
            max_len = 150
            
            if len(src_tokens) < max_len:
                src_tokens = src_tokens + [src_vocab['<PAD>']] * (max_len - len(src_tokens))
            else:
                src_tokens = src_tokens[:max_len]
            
            src = torch.tensor(src_tokens).unsqueeze(0)
            
            # Decode
            tgt_tokens = [tgt_vocab['<SOS>']]
            
            with torch.no_grad():
                for _ in range(max_len):
                    tgt_input = torch.tensor(tgt_tokens).unsqueeze(0)
                    
                    if tgt_input.size(1) < max_len:
                        pad_len = max_len - tgt_input.size(1)
                        tgt_input = torch.cat([tgt_input, torch.zeros(1, pad_len, dtype=torch.long)], dim=1)
                    
                    tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1))
                    output = model(src, tgt_input, tgt_mask=tgt_mask)
                    
                    next_token = torch.argmax(output[0, len(tgt_tokens)-1, :]).item()
                    
                    if next_token == tgt_vocab['<EOS>'] or next_token == tgt_vocab['<PAD>']:
                        break
                    
                    tgt_tokens.append(next_token)
                    
                    if len(tgt_tokens) > 100:
                        break
            
            predicted = ''.join([reverse_tgt_vocab.get(t, '') for t in tgt_tokens[1:]])
            print(f"English:  {predicted if predicted else '(empty)'}")
            
        except KeyboardInterrupt:
            print("\n\nBye boss! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    test_translation_model()
