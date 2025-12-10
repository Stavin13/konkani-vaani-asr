#!/usr/bin/env python3
"""
Test word-level translation model
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.word_level_translator import create_word_level_translator


def load_model(checkpoint_path='checkpoints/translation_model/translation_word_level_best.pt'):
    """Load trained word-level model"""
    print("="*70)
    print("LOADING WORD-LEVEL TRANSLATION MODEL")
    print("="*70)
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"\n❌ Model not found: {checkpoint_path}")
        return None, None, None, None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    idx_to_tgt = {idx: word for word, idx in tgt_vocab.items()}
    
    config = checkpoint.get('config', {})
    
    model = create_word_level_translator(
        src_vocab_size=config.get('src_vocab_size', len(src_vocab)),
        tgt_vocab_size=config.get('tgt_vocab_size', len(tgt_vocab)),
        d_model=config.get('d_model', 256)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n✓ Model loaded from: {checkpoint_path}")
    print(f"  Konkani vocab: {len(src_vocab)} words")
    print(f"  English vocab: {len(tgt_vocab)} words")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if 'val_acc' in checkpoint:
        print(f"  Val accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model, src_vocab, tgt_vocab, idx_to_tgt


def translate(model, text, src_vocab, tgt_vocab, idx_to_tgt, max_len=50, device='cpu'):
    """Translate Konkani text to English"""
    model.eval()
    
    # Tokenize input (word-level)
    words = text.replace(',', ' ,').replace('.', ' .').replace('!', ' !').replace('?', ' ?').split()
    words = [w.strip() for w in words if w.strip()]
    
    src_tokens = [src_vocab.get(word, src_vocab['<UNK>']) for word in words]
    
    # Pad
    if len(src_tokens) < max_len:
        src_tokens = src_tokens + [src_vocab['<PAD>']] * (max_len - len(src_tokens))
    else:
        src_tokens = src_tokens[:max_len]
    
    src = torch.tensor([src_tokens]).to(device)
    
    # Start with <SOS>
    tgt_tokens = [tgt_vocab['<SOS>']]
    
    with torch.no_grad():
        for _ in range(max_len):
            tgt_input = torch.tensor([tgt_tokens]).to(device)
            
            # Pad target
            if tgt_input.size(1) < max_len:
                padding = torch.full((1, max_len - tgt_input.size(1)), 
                                    tgt_vocab['<PAD>'], dtype=torch.long).to(device)
                tgt_input = torch.cat([tgt_input, padding], dim=1)
            
            # Generate masks
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            src_padding_mask = (src == src_vocab['<PAD>'])
            tgt_padding_mask = (tgt_input == tgt_vocab['<PAD>'])
            
            # Forward pass
            output = model(src, tgt_input, tgt_mask=tgt_mask,
                          src_padding_mask=src_padding_mask,
                          tgt_padding_mask=tgt_padding_mask)
            
            # Get next token
            next_token_logits = output[0, len(tgt_tokens) - 1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            # Stop if <EOS> or <PAD>
            if next_token == tgt_vocab['<EOS>'] or next_token == tgt_vocab['<PAD>']:
                break
            
            tgt_tokens.append(next_token)
    
    # Decode (skip <SOS>)
    translation_words = [idx_to_tgt.get(tok, '<UNK>') for tok in tgt_tokens[1:]]
    translation = ' '.join(translation_words)
    
    return translation


def test_examples(model, src_vocab, tgt_vocab, idx_to_tgt):
    """Test with examples"""
    print("\n" + "="*70)
    print("TESTING WITH EXAMPLES")
    print("="*70)
    
    test_cases = [
        ('घर', 'house'),
        ('पाणी', 'water'),
        ('खाणे', 'food'),
        ('हांव', 'I'),
        ('तूं', 'you'),
        ('बरे दिस', 'good day'),
        ('घरा वच', 'go home'),
        ('हांव घरा वचता', 'I am going home'),
        ('तूं पाणी पी', 'you drink water'),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for konkani, expected in test_cases:
        translation = translate(model, konkani, src_vocab, tgt_vocab, idx_to_tgt)
        
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
    """Interactive translation"""
    print("\n" + "="*70)
    print("INTERACTIVE TRANSLATION MODE")
    print("="*70)
    print("\nEnter Konkani text (or 'quit' to exit)")
    
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/translation_model/translation_word_level_best.pt')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['test', 'interactive', 'both'])
    parser.add_argument('--text', type=str, default=None)
    args = parser.parse_args()
    
    model, src_vocab, tgt_vocab, idx_to_tgt = load_model(args.checkpoint)
    
    if model is None:
        return
    
    if args.text:
        print("\n" + "="*70)
        print("SINGLE TRANSLATION")
        print("="*70)
        translation = translate(model, args.text, src_vocab, tgt_vocab, idx_to_tgt)
        print(f"\nKonkani: {args.text}")
        print(f"English: {translation}")
        return
    
    if args.mode in ['test', 'both']:
        test_examples(model, src_vocab, tgt_vocab, idx_to_tgt)
    
    if args.mode in ['interactive', 'both']:
        interactive_mode(model, src_vocab, tgt_vocab, idx_to_tgt)


if __name__ == '__main__':
    main()
