#!/usr/bin/env python3
"""
Simple Google Translate wrapper for Konkani → English
Uses the fixed googletrans library
"""
from googletrans import Translator
import sys


def translate_konkani_to_english(text):
    """Translate Konkani text to English using Google Translate"""
    try:
        translator = Translator()
        # Google Translate treats Konkani as Hindi (similar script)
        result = translator.translate(text, src='hi', dest='en')
        return result.text
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    if len(sys.argv) > 1:
        # Command line argument
        konkani_text = ' '.join(sys.argv[1:])
        english = translate_konkani_to_english(konkani_text)
        if english:
            print(f"Konkani:  {konkani_text}")
            print(f"English:  {english}")
    else:
        # Interactive mode
        print("="*70)
        print("KONKANI → ENGLISH TRANSLATOR (Google Translate)")
        print("="*70)
        print("\nEnter Konkani text to translate (or 'quit' to exit)")
        print("Note: Uses Google Translate API (treats Konkani as Hindi)")
        
        while True:
            print("\n" + "-"*70)
            konkani_text = input("\nKonkani: ").strip()
            
            if konkani_text.lower() in ['quit', 'exit', 'q']:
                print("\nBye! 👋")
                break
            
            if not konkani_text:
                continue
            
            english = translate_konkani_to_english(konkani_text)
            if english:
                print(f"English: {english}")


if __name__ == '__main__':
    main()
