#!/usr/bin/env python3
"""
Test the trained translation model with interactive examples
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkani_custom_translator import create_custom_translation_model


def load_model(checkpoint_path='checkpoints/translation_model/translation_model_best.pt'):
    """Load trained translation model"""
    print("="*70)
    print("LOADING TRANSLATION MODEL")
    print("="*70)
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"\n❌ Model not found: {checkpoint_path}")
        return None, None, None, None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get vocabularies
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    
    # Create reverse vocab for decoding
    idx_to_tgt = {idx: char for char, idx in tgt_vocab.items()}
    
    # Get config
    config = checkpoint.get('config', {})
    src_vocab_size = config.get('src_vocab_size', len(src_vocab))
    tgt_vocab_size = config.get('tgt_vocab_size', len(tgt_vocab))
    
    # Create model
    model = create_custom_translation_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n✓ Model loaded from: {checkpoint_path}")
    print(f"  Konkani vocab: {len(src_vocab)} chars")
    print(f"  English vocab: {len(tgt_vocab)} chars")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Show training stats if available
    if 'best_val_loss' in checkpoint:
        print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    history = checkpoint.get('history', {})
    if history and 'val_acc' in history and len(history['val_acc']) > 0:
        print(f"  Best val acc: {max(history['val_acc']):.2f}%")
    
    return model, src_vocab, tgt_vocab, idx_to_tgt


def translate(model, text, src_vocab, tgt_vocab, idx_to_tgt, max_len=150, device='cpu'):
    """Translate Konkani text to English"""
    model.eval()
    
    # Tokenize input
    src_tokens = [src_vocab.get(char, src_vocab['<UNK>']) for char in text]
    
    # Pad
    if len(src_tokens) < max_len:
        src_tokens = src_tokens + [src_vocab['<PAD>']] * (max_len - len(src_tokens))
    else:
        src_tokens = src_tokens[:max_len]
    
    src = torch.tensor([src_tokens]).to(device)
    
    # Start with <SOS> token
    tgt_tokens = [tgt_vocab['<SOS>']]
    
    with torch.no_grad():
        for _ in range(max_len):
            # Prepare target input
            tgt_input = torch.tensor([tgt_tokens]).to(device)
            
            # Pad target to max_len
            if tgt_input.size(1) < max_len:
                padding = torch.full((1, max_len - tgt_input.size(1)), 
                                    tgt_vocab['<PAD>'], dtype=torch.long).to(device)
                tgt_input = torch.cat([tgt_input, padding], dim=1)
            
            # Generate mask
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            # Forward pass
            output = model(src, tgt_input, tgt_mask=tgt_mask)
            
            # Get next token (at current position)
            next_token_logits = output[0, len(tgt_tokens) - 1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            # Stop if <EOS> or <PAD>
            if next_token == tgt_vocab['<EOS>'] or next_token == tgt_vocab['<PAD>']:
                break
            
            tgt_tokens.append(next_token)
    
    # Decode
    translation = ''.join([idx_to_tgt.get(tok, '') for tok in tgt_tokens[1:]])  # Skip <SOS>
    
    return translation


def test_examples(model, src_vocab, tgt_vocab, idx_to_tgt):
    """Test with predefined examples"""
    print("\n" + "="*70)
    print("TESTING WITH EXAMPLES")
    print("="*70)
    
    test_cases = [
        # Letters
        ('अ', 'a'),
        ('क', 'ka'),
        ('म', 'ma'),
        
        # Words
        ('घर', 'house'),
        ('पाणी', 'water'),
        ('खाणे', 'food'),
        ('मनीस', 'person'),
        ('बायल', 'woman'),
        ('भुरगे', 'child'),
        
        # Phrases
        ('बरे दिस', 'good day'),
        ('घरा वच', 'go home'),
        ('पाणी पी', 'drink water'),
        
        # Sentences
        ('हांव घरा वचता', 'I am going home'),
        ('तूं पाणी पी', 'you drink water'),
        ('तो खाणे खाता', 'he eats food'),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for konkani, expected in test_cases:
        translation = translate(model, konkani, src_vocab, tgt_vocab, idx_to_tgt)
        
        # Check if translation matches (flexible matching)
        is_correct = expected.lower() in translation.lower() or translation.lower() in expected.lower()
        
        status = "✓" if is_correct else "✗"
        if is_correct:
            correct += 1
        
        print(f"\n{status} Konkani:  {konkani}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {translation}")
    
    accuracy = 100 * correct / total
    print(f"\n{'='*70}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"{'='*70}")


def interactive_mode(model, src_vocab, tgt_vocab, idx_to_tgt):
    """Interactive translation mode"""
    print("\n" + "="*70)
    print("INTERACTIVE TRANSLATION MODE")
    print("="*70)
    print("\nEnter Konkani text to translate (or 'quit' to exit)")
    print("Examples: घर, पाणी, हांव घरा वचता")
    
    while True:
        print("\n" + "-"*70)
        konkani_text = input("\nKonkani: ").strip()
        
        if konkani_text.lower() in ['quit', 'exit', 'q']:
            print("\nBye! 👋")
            break
        
        if not konkani_text:
            continue
        
        translation = translate(model, konkani_text, src_vocab, tgt_vocab, idx_to_tgt)
        print(f"English: {translation}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test translation model')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/translation_model/translation_model_best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['test', 'interactive', 'both'],
                       help='Test mode: test examples, interactive, or both')
    parser.add_argument('--text', type=str, default=None,
                       help='Single text to translate')
    args = parser.parse_args()
    
    # Load model
    model, src_vocab, tgt_vocab, idx_to_tgt = load_model(args.checkpoint)
    
    if model is None:
        return
    
    # Single translation
    if args.text:
        print("\n" + "="*70)
        print("SINGLE TRANSLATION")
        print("="*70)
        translation = translate(model, args.text, src_vocab, tgt_vocab, idx_to_tgt)
        print(f"\nKonkani: {args.text}")
        print(f"English: {translation}")
        return
    
    # Test examples
    if args.mode in ['test', 'both']:
        test_examples(model, src_vocab, tgt_vocab, idx_to_tgt)
    
    # Interactive mode
    if args.mode in ['interactive', 'both']:
        interactive_mode(model, src_vocab, tgt_vocab, idx_to_tgt)


if __name__ == '__main__':
    main()
