#!/usr/bin/env python3
"""
Generate 10,000 Konkani-English translation pairs using:
1. Curriculum data (letters, words, phrases)
2. Existing corpus data
3. Synthetic variations
4. Google Translate for real sentences
"""
import json
from pathlib import Path
from tqdm import tqdm
import random
from googletrans import Translator
import time


def load_curriculum_data():
    """Load existing curriculum data"""
    path = Path('data/translation_data/konkani_english_curriculum_sorted.json')
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Exclude complex to avoid duplicates
        return [d for d in data if d['level'] != 'complex']
    return []


def load_corpus_texts():
    """Load all Konkani texts from corpus"""
    texts = set()
    
    # From emotion dataset
    emotion_file = Path('konkani_emotion_synthetic_10k.jsonl')
    if emotion_file.exists():
        with open(emotion_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'text' in data and len(data['text'].strip()) > 3:
                        texts.add(data['text'].strip())
                except:
                    pass
    
    # From ASR train data
    train_file = Path('data/konkani-asr-v0/splits/manifests/train.json')
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'text' in data and len(data['text'].strip()) > 3:
                        texts.add(data['text'].strip())
                except:
                    pass
    
    # From ASR test data
    test_file = Path('data/konkani-asr-v0/splits/manifests/test.json')
    if test_file.exists():
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'text' in data and len(data['text'].strip()) > 3:
                        texts.add(data['text'].strip())
                except:
                    pass
    
    return list(texts)


def translate_batch_with_google(texts, batch_size=50):
    """Translate texts in batches with rate limiting"""
    translator = Translator()
    translations = []
    
    print(f"\n🌐 Translating {len(texts)} texts with Google Translate...")
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        
        for text in batch:
            try:
                result = translator.translate(text, src='hi', dest='en')
                translations.append({
                    'konkani': text,
                    'english': result.text,
                    'source': 'google_translate'
                })
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                # Skip failed translations
                continue
        
        # Longer pause between batches
        if i + batch_size < len(texts):
            time.sleep(2)
    
    return translations


def generate_synthetic_variations(base_data, target_count=5000):
    """Generate synthetic variations of existing data"""
    print(f"\n🔄 Generating synthetic variations...")
    
    variations = []
    words = [d for d in base_data if d.get('level') == 'word']
    phrases = [d for d in base_data if d.get('level') == 'phrase']
    
    # Generate word combinations
    for _ in range(target_count // 2):
        if len(words) >= 2:
            w1, w2 = random.sample(words, 2)
            variations.append({
                'konkani': f"{w1['konkani']} {w2['konkani']}",
                'english': f"{w1['english']} {w2['english']}",
                'source': 'synthetic_combination'
            })
    
    # Generate phrase variations with different articles
    articles = ['a', 'the', 'this', 'that', 'my', 'your']
    for phrase in phrases:
        for article in articles[:3]:  # Use first 3 articles
            variations.append({
                'konkani': phrase['konkani'],
                'english': f"{article} {phrase['english']}",
                'source': 'synthetic_article'
            })
    
    return variations[:target_count]


def main():
    print("="*70)
    print("GENERATE 10,000 TRANSLATION PAIRS")
    print("="*70)
    
    all_translations = []
    
    # 1. Load curriculum data (letters, words, phrases, sentences)
    print("\n📚 Loading curriculum data...")
    curriculum = load_curriculum_data()
    all_translations.extend(curriculum)
    print(f"  ✓ Loaded {len(curriculum)} curriculum examples")
    
    # 2. Load corpus texts
    print("\n📖 Loading corpus texts...")
    corpus_texts = load_corpus_texts()
    print(f"  ✓ Found {len(corpus_texts)} unique Konkani texts")
    
    # 3. Translate corpus with Google (limit to avoid rate limits)
    max_google_translations = 3000
    if len(corpus_texts) > max_google_translations:
        print(f"  ⚠️  Limiting to {max_google_translations} texts to avoid rate limits")
        corpus_texts = random.sample(corpus_texts, max_google_translations)
    
    google_translations = translate_batch_with_google(corpus_texts)
    all_translations.extend(google_translations)
    print(f"  ✓ Translated {len(google_translations)} texts")
    
    # 4. Generate synthetic variations
    remaining = 10000 - len(all_translations)
    if remaining > 0:
        synthetic = generate_synthetic_variations(curriculum, remaining)
        all_translations.extend(synthetic)
        print(f"  ✓ Generated {len(synthetic)} synthetic variations")
    
    # 5. Deduplicate
    seen = set()
    unique_translations = []
    for item in all_translations:
        key = (item['konkani'], item['english'])
        if key not in seen:
            seen.add(key)
            unique_translations.append(item)
    
    print(f"\n📊 Total unique translations: {len(unique_translations)}")
    
    # 6. Save
    output_path = Path('data/translation_data/konkani_english_10k.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_translations, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved to: {output_path}")
    
    # Statistics
    by_source = {}
    for item in unique_translations:
        source = item.get('source', item.get('level', 'unknown'))
        by_source[source] = by_source.get(source, 0) + 1
    
    print("\nBreakdown by source:")
    for source, count in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {source:30s}: {count:5d}")
    
    print("\n" + "="*70)
    print("NEXT STEP: Train the model")
    print("  python scripts/train_translation_word_level.py")
    print("="*70)


if __name__ == '__main__':
    main()
