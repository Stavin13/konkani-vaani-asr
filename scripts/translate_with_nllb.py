#!/usr/bin/env python3
"""
Offline Konkani-English Translation using NLLB
Works 100% offline after first download (~2.4GB)
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sys
from pathlib import Path


class NLLBTranslator:
    """NLLB offline translator for Konkani"""
    
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", device=None):
        """
        Initialize NLLB translator
        
        Args:
            model_name: NLLB model to use
            device: 'mps' for Mac GPU, 'cuda' for NVIDIA, 'cpu' for CPU
        """
        print("="*70)
        print("NLLB OFFLINE TRANSLATOR")
        print("="*70)
        
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
                print("✓ Using Mac GPU (MPS)")
            elif torch.cuda.is_available():
                device = "cuda"
                print("✓ Using NVIDIA GPU")
            else:
                device = "cpu"
                print("✓ Using CPU")
        
        self.device = torch.device(device)
        
        # Download model (only first time, then cached)
        print(f"\nLoading NLLB model: {model_name}")
        print("(First time: ~2.4GB download, then cached locally)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Language codes
        self.konkani_code = "kok_Deva"  # Konkani in Devanagari script
        self.english_code = "eng_Latn"  # English in Latin script
        
        print(f"✓ Model loaded successfully!")
        print(f"✓ Device: {self.device}")
        print(f"✓ Model size: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print("\n🌐 Ready for offline translation!")
    
    def translate(self, text, src_lang="kok_Deva", tgt_lang="eng_Latn", max_length=200):
        """
        Translate text
        
        Args:
            text: Text to translate
            src_lang: Source language code (default: kok_Deva for Konkani)
            tgt_lang: Target language code (default: eng_Latn for English)
            max_length: Maximum translation length
        
        Returns:
            Translated text
        """
        # Set source language
        self.tokenizer.src_lang = src_lang
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get target language token ID
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=max_length,
                num_beams=5,  # Beam search for better quality
                early_stopping=True
            )
        
        # Decode
        translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        return translation
    
    def translate_konkani_to_english(self, konkani_text):
        """Convenience method for Konkani → English"""
        return self.translate(konkani_text, src_lang=self.konkani_code, tgt_lang=self.english_code)
    
    def translate_english_to_konkani(self, english_text):
        """Convenience method for English → Konkani"""
        return self.translate(english_text, src_lang=self.english_code, tgt_lang=self.konkani_code)
    
    def translate_batch(self, texts, src_lang="kok_Deva", tgt_lang="eng_Latn", batch_size=8):
        """Translate multiple texts efficiently"""
        translations = []
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize batch
            self.tokenizer.src_lang = src_lang
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=200,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode
            batch_translations = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            translations.extend(batch_translations)
        
        return translations


def test_examples(translator):
    """Test with example translations"""
    print("\n" + "="*70)
    print("TESTING KONKANI → ENGLISH")
    print("="*70)
    
    test_cases = [
        "घर",
        "पाणी",
        "खाणे",
        "हांव",
        "तूं",
        "बरे दिस",
        "घरा वच",
        "हांव घरा वचता",
        "तूं पाणी पी",
        "तो खाणे खाता",
    ]
    
    for konkani in test_cases:
        english = translator.translate_konkani_to_english(konkani)
        print(f"\nKonkani:  {konkani}")
        print(f"English:  {english}")
    
    print("\n" + "="*70)
    print("TESTING ENGLISH → KONKANI")
    print("="*70)
    
    english_tests = [
        "house",
        "water",
        "food",
        "I am going home",
        "good morning",
    ]
    
    for english in english_tests:
        konkani = translator.translate_english_to_konkani(english)
        print(f"\nEnglish:  {english}")
        print(f"Konkani:  {konkani}")


def interactive_mode(translator):
    """Interactive translation mode"""
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("\nCommands:")
    print("  k2e <text>  - Translate Konkani to English")
    print("  e2k <text>  - Translate English to Konkani")
    print("  quit        - Exit")
    
    while True:
        print("\n" + "-"*70)
        user_input = input("\n> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nBye! 👋")
            break
        
        if not user_input:
            continue
        
        # Parse command
        if user_input.startswith("k2e "):
            konkani_text = user_input[4:].strip()
            if konkani_text:
                english = translator.translate_konkani_to_english(konkani_text)
                print(f"English: {english}")
        
        elif user_input.startswith("e2k "):
            english_text = user_input[4:].strip()
            if english_text:
                konkani = translator.translate_english_to_konkani(english_text)
                print(f"Konkani: {konkani}")
        
        else:
            # Default: assume Konkani input
            english = translator.translate_konkani_to_english(user_input)
            print(f"English: {english}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='NLLB Offline Translator for Konkani')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['test', 'interactive', 'translate'],
                       help='Mode: test examples, interactive, or single translation')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to translate (for translate mode)')
    parser.add_argument('--direction', type=str, default='k2e',
                       choices=['k2e', 'e2k'],
                       help='Translation direction: k2e (Konkani→English) or e2k (English→Konkani)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['mps', 'cuda', 'cpu'],
                       help='Device to use (auto-detected if not specified)')
    args = parser.parse_args()
    
    # Initialize translator (downloads model on first run)
    translator = NLLBTranslator(device=args.device)
    
    if args.mode == 'test':
        test_examples(translator)
    
    elif args.mode == 'translate' and args.text:
        if args.direction == 'k2e':
            result = translator.translate_konkani_to_english(args.text)
            print(f"\nKonkani: {args.text}")
            print(f"English: {result}")
        else:
            result = translator.translate_english_to_konkani(args.text)
            print(f"\nEnglish: {args.text}")
            print(f"Konkani: {result}")
    
    else:
        interactive_mode(translator)


if __name__ == '__main__':
    main()
