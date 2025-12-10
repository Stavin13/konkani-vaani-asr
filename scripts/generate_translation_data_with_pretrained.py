#!/usr/bin/env python3
"""
Generate high-quality Konkani-English translation data using pre-trained models
Supports: Google Translate API, IndicTrans2, mBART, M2M100
"""
import json
from pathlib import Path
from tqdm import tqdm
import argparse


def load_konkani_corpus():
    """Load Konkani text from various sources"""
    texts = []
    
    # From emotion dataset
    emotion_file = Path('konkani_emotion_synthetic_10k.jsonl')
    if emotion_file.exists():
        print(f"Loading from {emotion_file}...")
        with open(emotion_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'text' in data:
                    texts.append(data['text'])
    
    # From ASR corpus
    corpus_manifest = Path('KonkaniRawSpeechCorpus/Metadata/konkani_corpus_manifest.json')
    if corpus_manifest.exists():
        print(f"Loading from {corpus_manifest}...")
        with open(corpus_manifest, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
            for item in corpus_data:
                if 'text' in item:
                    texts.append(item['text'])
    
    # From ASR test data
    asr_test = Path('data/konkani-asr-v0/splits/manifests/test.json')
    if asr_test.exists():
        print(f"Loading from {asr_test}...")
        with open(asr_test, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'text' in data:
                    texts.append(data['text'])
    
    # Deduplicate and clean
    texts = list(set(texts))
    texts = [t.strip() for t in texts if t.strip() and len(t.strip()) > 3]
    
    print(f"\n✓ Loaded {len(texts)} unique Konkani texts")
    return texts


def translate_with_google(texts, source='hi', target='en'):
    """
    Translate using Google Translate (free tier via googletrans)
    Note: Konkani not directly supported, using Hindi as proxy
    """
    try:
        from googletrans import Translator
        translator = Translator()
        
        print("\n🌐 Translating with Google Translate...")
        translations = []
        
        for text in tqdm(texts):
            try:
                result = translator.translate(text, src=source, dest=target)
                translations.append({
                    'konkani': text,
                    'english': result.text,
                    'method': 'google_translate',
                    'confidence': 0.8
                })
            except Exception as e:
                print(f"Error translating '{text[:30]}...': {e}")
                translations.append({
                    'konkani': text,
                    'english': text,  # Fallback
                    'method': 'google_translate_failed',
                    'confidence': 0.0
                })
        
        return translations
    
    except ImportError:
        print("❌ googletrans not installed. Install with: pip install googletrans==4.0.0-rc1")
        return []


def translate_with_indictrans2(texts):
    """
    Translate using IndicTrans2 (AI4Bharat's model)
    Best for Indian languages including Konkani
    """
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        print("\n🇮🇳 Loading IndicTrans2 model...")
        model_name = "ai4bharat/indictrans2-en-indic-1B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        print("Translating with IndicTrans2...")
        translations = []
        
        for text in tqdm(texts):
            try:
                inputs = tokenizer(text, return_tensors="pt", padding=True)
                outputs = model.generate(**inputs, max_length=256)
                translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                translations.append({
                    'konkani': text,
                    'english': translation,
                    'method': 'indictrans2',
                    'confidence': 0.9
                })
            except Exception as e:
                print(f"Error: {e}")
                translations.append({
                    'konkani': text,
                    'english': text,
                    'method': 'indictrans2_failed',
                    'confidence': 0.0
                })
        
        return translations
    
    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers")
        return []


def translate_with_m2m100(texts):
    """
    Translate using Meta's M2M100 (multilingual model)
    """
    try:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        
        print("\n🌍 Loading M2M100 model...")
        model_name = "facebook/m2m100_418M"
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        
        # Set source language (using Hindi as proxy for Konkani)
        tokenizer.src_lang = "hi"
        
        print("Translating with M2M100...")
        translations = []
        
        for text in tqdm(texts):
            try:
                encoded = tokenizer(text, return_tensors="pt")
                generated_tokens = model.generate(
                    **encoded,
                    forced_bos_token_id=tokenizer.get_lang_id("en"),
                    max_length=256
                )
                translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                
                translations.append({
                    'konkani': text,
                    'english': translation,
                    'method': 'm2m100',
                    'confidence': 0.85
                })
            except Exception as e:
                print(f"Error: {e}")
                translations.append({
                    'konkani': text,
                    'english': text,
                    'method': 'm2m100_failed',
                    'confidence': 0.0
                })
        
        return translations
    
    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers")
        return []


def translate_with_mbart(texts):
    """
    Translate using mBART (multilingual BART)
    """
    try:
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        
        print("\n📚 Loading mBART model...")
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        
        # Set source language (using Hindi as proxy)
        tokenizer.src_lang = "hi_IN"
        
        print("Translating with mBART...")
        translations = []
        
        for text in tqdm(texts):
            try:
                encoded = tokenizer(text, return_tensors="pt")
                generated_tokens = model.generate(
                    **encoded,
                    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
                    max_length=256
                )
                translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                
                translations.append({
                    'konkani': text,
                    'english': translation,
                    'method': 'mbart',
                    'confidence': 0.85
                })
            except Exception as e:
                print(f"Error: {e}")
                translations.append({
                    'konkani': text,
                    'english': text,
                    'method': 'mbart_failed',
                    'confidence': 0.0
                })
        
        return translations
    
    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers")
        return []


def main():
    parser = argparse.ArgumentParser(description='Generate translation data using pre-trained models')
    parser.add_argument('--method', type=str, default='google',
                       choices=['google', 'indictrans2', 'm2m100', 'mbart', 'all'],
                       help='Translation method to use')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to translate')
    parser.add_argument('--output', type=str, default='data/translation_data/konkani_english_pretrained.json',
                       help='Output file path')
    args = parser.parse_args()
    
    print("="*70)
    print("GENERATE TRANSLATION DATA WITH PRE-TRAINED MODELS")
    print("="*70)
    
    # Load Konkani texts
    texts = load_konkani_corpus()
    
    if args.max_samples:
        texts = texts[:args.max_samples]
        print(f"Using first {args.max_samples} samples")
    
    # Translate
    translations = []
    
    if args.method == 'google' or args.method == 'all':
        translations.extend(translate_with_google(texts))
    
    if args.method == 'indictrans2' or args.method == 'all':
        translations.extend(translate_with_indictrans2(texts))
    
    if args.method == 'm2m100' or args.method == 'all':
        translations.extend(translate_with_m2m100(texts))
    
    if args.method == 'mbart' or args.method == 'all':
        translations.extend(translate_with_mbart(texts))
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved {len(translations)} translations to: {output_path}")
    
    # Statistics
    methods = {}
    for t in translations:
        method = t['method']
        methods[method] = methods.get(method, 0) + 1
    
    print("\nTranslation methods used:")
    for method, count in methods.items():
        print(f"  - {method}: {count}")
    
    avg_confidence = sum(t['confidence'] for t in translations) / len(translations)
    print(f"\nAverage confidence: {avg_confidence:.2f}")


if __name__ == '__main__':
    main()
