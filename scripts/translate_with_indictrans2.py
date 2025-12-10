#!/usr/bin/env python3
"""
Konkani-English Translation using IndicTrans2
Best model for Indian languages - supports Konkani natively!
"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import sys


class IndicTrans2Translator:
    def __init__(self, device='cpu'):
        """Initialize IndicTrans2 model"""
        print("Loading IndicTrans2 model...")
        print("(First time will download ~2GB model)")
        
        # IndicTrans2 for Indic to English
        model_name = "ai4bharat/indictrans2-indic-en-1B"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        print(f"✓ Model loaded on {device}")
    
    def translate(self, text, src_lang='kok_Deva', tgt_lang='eng_Latn'):
        """
        Translate Konkani to English
        
        Args:
            text: Konkani text in Devanagari script
            src_lang: Source language code (kok_Deva = Konkani Devanagari)
            tgt_lang: Target language code (eng_Latn = English Latin)
        """
        # Prepare input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=False
            )
        
        # Decode
        translation = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        
        return translation


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Translate Konkani to English using IndicTrans2')
    parser.add_argument('--text', type=str, help='Konkani text to translate')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                       help='Device to use')
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print("="*70)
    print("KONKANI → ENGLISH TRANSLATOR (IndicTrans2)")
    print("="*70)
    print(f"Device: {device}")
    
    # Initialize translator
    translator = IndicTrans2Translator(device=device)
    
    if args.text:
        # Single translation
        print(f"\nKonkani:  {args.text}")
        translation = translator.translate(args.text)
        print(f"English:  {translation}")
    
    elif args.interactive:
        # Interactive mode
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
            
            translation = translator.translate(konkani_text)
            print(f"English: {translation}")
    
    else:
        # Test examples
        print("\nTesting with examples...")
        
        test_cases = [
            'घर',
            'पाणी',
            'खाणे',
            'हांव',
            'तूं',
            'बरे दिस',
            'हांव घरा वचता',
            'तूं पाणी पी',
            'तो खाणे खाता'
        ]
        
        for konkani in test_cases:
            translation = translator.translate(konkani)
            print(f"\nKonkani:  {konkani}")
            print(f"English:  {translation}")


if __name__ == '__main__':
    main()
