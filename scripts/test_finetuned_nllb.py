#!/usr/bin/env python3
"""
Test fine-tuned NLLB model
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from pathlib import Path
import sys


def load_finetuned_model(model_path):
    """Load fine-tuned NLLB model"""
    print("="*70)
    print("LOADING FINE-TUNED NLLB MODEL")
    print("="*70)
    
    print(f"\nLoading from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Auto-detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Mac GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using NVIDIA GPU")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return tokenizer, model, device


def translate(tokenizer, model, device, text, src_lang="kok_Deva", tgt_lang="eng_Latn"):
    """Translate using fine-tuned model"""
    tokenizer.src_lang = src_lang
    
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=200,
            num_beams=5,
            early_stopping=True
        )
    
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation


def compare_models(base_model_path, finetuned_model_path, test_cases):
    """Compare base vs fine-tuned model"""
    print("\n" + "="*70)
    print("COMPARING BASE vs FINE-TUNED MODEL")
    print("="*70)
    
    # Load base model
    print("\nLoading base model...")
    base_tokenizer, base_model, device = load_finetuned_model(base_model_path)
    
    # Load fine-tuned model
    print("\nLoading fine-tuned model...")
    ft_tokenizer, ft_model, _ = load_finetuned_model(finetuned_model_path)
    
    print("\n" + "="*70)
    print("TRANSLATION COMPARISON")
    print("="*70)
    
    for konkani, expected in test_cases:
        base_trans = translate(base_tokenizer, base_model, device, konkani)
        ft_trans = translate(ft_tokenizer, ft_model, device, konkani)
        
        print(f"\nKonkani:    {konkani}")
        print(f"Expected:   {expected}")
        print(f"Base:       {base_trans}")
        print(f"Fine-tuned: {ft_trans}")
        
        # Check if fine-tuned is better
        if expected.lower() in ft_trans.lower():
            print("✓ Fine-tuned matches expected!")
        elif expected.lower() in base_trans.lower():
            print("⚠️  Base model was already good")
        else:
            print("⚠️  Neither matches perfectly")


def test_finetuned_model(model_path):
    """Test fine-tuned model with examples"""
    tokenizer, model, device = load_finetuned_model(model_path)
    
    print("\n" + "="*70)
    print("TESTING FINE-TUNED MODEL")
    print("="*70)
    
    test_cases = [
        ("घर", "house"),
        ("पाणी", "water"),
        ("खाणे", "food"),
        ("हांव", "I"),
        ("तूं", "you"),
        ("बरे दिस", "good day"),
        ("घरा वच", "go home"),
        ("हांव घरा वचता", "I am going home"),
        ("तूं पाणी पी", "you drink water"),
        ("तो खाणे खाता", "he eats food"),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for konkani, expected in test_cases:
        translation = translate(tokenizer, model, device, konkani)
        
        is_correct = expected.lower() in translation.lower()
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


def interactive_mode(model_path):
    """Interactive translation with fine-tuned model"""
    tokenizer, model, device = load_finetuned_model(model_path)
    
    print("\n" + "="*70)
    print("INTERACTIVE MODE (Fine-tuned Model)")
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
        
        english = translate(tokenizer, model, device, konkani_text)
        print(f"English: {english}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test fine-tuned NLLB model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to fine-tuned model')
    parser.add_argument('--mode', type=str, default='test',
                       choices=['test', 'interactive', 'compare'],
                       help='Test mode')
    parser.add_argument('--base_model', type=str, default='facebook/nllb-200-distilled-600M',
                       help='Base model for comparison')
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"\n❌ Model not found: {args.model}")
        print("\nRun fine-tuning first: python scripts/finetune_nllb.py")
        return
    
    if args.mode == 'test':
        test_finetuned_model(args.model)
    
    elif args.mode == 'interactive':
        interactive_mode(args.model)
    
    elif args.mode == 'compare':
        test_cases = [
            ("घर", "house"),
            ("पाणी", "water"),
            ("हांव घरा वचता", "I am going home"),
            ("तूं पाणी पी", "you drink water"),
        ]
        compare_models(args.base_model, args.model, test_cases)


if __name__ == '__main__':
    main()
