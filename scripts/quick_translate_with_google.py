#!/usr/bin/env python3
"""
Quick translation using Google Translate (free, no API key needed)
"""
import json
from pathlib import Path
from tqdm import tqdm
import time


def load_konkani_texts():
    """Load unique Konkani texts from all sources"""
    texts = set()
    
    # From emotion dataset
    emotion_file = Path('konkani_emotion_synthetic_10k.jsonl')
    if emotion_file.exists():
        print(f"✓ Loading from {emotion_file.name}...")
        with open(emotion_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'text' in data and data['text'].strip():
                        texts.add(data['text'].strip())
                except:
                    pass
    
    # Convert to list and sort
    texts = sorted(list(texts))
    texts = [t for t in texts if len(t) > 5]  # Filter very short texts
    
    print(f"✓ Found {len(texts)} unique Konkani texts\n")
    return texts


def translate_batch_google(texts, batch_size=50):
    """Translate using googletrans (free Google Translate)"""
    try:
        from googletrans import Translator
    except ImportError:
        print("❌ googletrans not installed!")
        print("\nInstall with:")
        print("  pip install googletrans==4.0.0-rc1")
        return []
    
    translator = Translator()
    translations = []
    
    print("🌐 Translating with Google Translate...")
    print("(Using Hindi as proxy since Konkani not directly supported)\n")
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        
        for text in batch:
            try:
                # Translate from Hindi to English (Konkani uses Devanagari like Hindi)
                result = translator.translate(text, src='hi', dest='en')
                
                translations.append({
                    'konkani': text,
                    'english': result.text,
                    'method': 'google_translate',
                    'source_lang': 'hi',
                    'confidence': 0.8
                })
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\n⚠️  Error translating: {text[:50]}...")
                print(f"   Error: {e}")
                # Add with empty translation
                translations.append({
                    'konkani': text,
                    'english': '',
                    'method': 'google_translate_failed',
                    'confidence': 0.0
                })
        
        # Longer delay between batches
        if i + batch_size < len(texts):
            time.sleep(1)
    
    return translations


def main():
    print("="*70)
    print("QUICK TRANSLATION WITH GOOGLE TRANSLATE")
    print("="*70)
    print()
    
    # Load texts
    texts = load_konkani_texts()
    
    if not texts:
        print("❌ No Konkani texts found!")
        return
    
    # Show sample
    print("Sample texts to translate:")
    for i, text in enumerate(texts[:5], 1):
        print(f"  {i}. {text[:60]}...")
    print()
    
    # Ask for confirmation
    response = input(f"Translate {len(texts)} texts? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Translate
    translations = translate_batch_google(texts)
    
    if not translations:
        print("\n❌ Translation failed!")
        return
    
    # Filter out failed translations
    successful = [t for t in translations if t['confidence'] > 0]
    failed = [t for t in translations if t['confidence'] == 0]
    
    print(f"\n✓ Successfully translated: {len(successful)}")
    print(f"✗ Failed: {len(failed)}")
    
    # Save results
    output_path = Path('data/translation_data/konkani_english_google.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(successful, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved to: {output_path}")
    
    # Show samples
    print("\n" + "="*70)
    print("SAMPLE TRANSLATIONS")
    print("="*70)
    for i, item in enumerate(successful[:10], 1):
        print(f"\n[{i}]")
        print(f"  Konkani: {item['konkani'][:60]}")
        print(f"  English: {item['english'][:60]}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Review the translations in:")
    print(f"   {output_path}")
    print("\n2. Manually fix any bad translations")
    print("\n3. Train your model with:")
    print("   python scripts/train_translation_only.py")


if __name__ == '__main__':
    main()
