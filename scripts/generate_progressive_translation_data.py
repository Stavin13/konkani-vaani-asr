#!/usr/bin/env python3
"""
Generate progressive translation training data using curriculum learning:
1. Individual letters (Devanagari → English transliteration)
2. Common words (10 translations per word)
3. Simple phrases (2-3 words)
4. Short sentences (4-6 words)
5. Complex sentences (7+ words)

This helps the model learn building blocks before tackling full sentences.
"""
import json
from pathlib import Path
from tqdm import tqdm
import random

# Konkani Devanagari to English transliteration mapping
LETTER_MAPPING = {
    # Vowels
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ऋ': 'ru', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    
    # Consonants
    'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'nga',
    'च': 'cha', 'छ': 'chha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'nya',
    'ट': 'ta', 'ठ': 'tha', 'ड': 'da', 'ढ': 'dha', 'ण': 'na',
    'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
    'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
    'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va', 'श': 'sha',
    'ष': 'sha', 'स': 'sa', 'ह': 'ha', 'ळ': 'la', 'क्ष': 'ksha',
    'ज्ञ': 'gnya', 'ं': 'n', 'ः': 'h', '्': '',
    
    # Matras (vowel signs)
    'ा': 'aa', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'ृ': 'ru', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
}

# Common Konkani words with multiple English translations
COMMON_WORDS = {
    'घर': ['house', 'home', 'residence', 'dwelling', 'abode'],
    'पाणी': ['water', 'aqua', 'liquid water', 'drinking water', 'H2O'],
    'खाणे': ['food', 'meal', 'eating', 'to eat', 'nourishment'],
    'मनीस': ['person', 'man', 'human', 'individual', 'people'],
    'बायल': ['woman', 'lady', 'female', 'girl', 'women'],
    'भुरगे': ['child', 'kid', 'children', 'youngster', 'little one'],
    'दिस': ['day', 'daytime', 'daily', 'daylight', 'date'],
    'रात': ['night', 'nighttime', 'evening', 'darkness', 'nocturnal'],
    'सकाळ': ['morning', 'dawn', 'early morning', 'sunrise', 'AM'],
    'संजे': ['evening', 'dusk', 'sunset', 'twilight', 'PM'],
    'येवप': ['to come', 'come', 'coming', 'arrive', 'approach'],
    'वचप': ['to go', 'go', 'going', 'leave', 'depart'],
    'खावप': ['to eat', 'eat', 'eating', 'consume', 'dine'],
    'पिवप': ['to drink', 'drink', 'drinking', 'sip', 'beverage'],
    'बसप': ['to sit', 'sit', 'sitting', 'seated', 'sit down'],
    'उबो': ['to stand', 'stand', 'standing', 'upright', 'stand up'],
    'निदप': ['to sleep', 'sleep', 'sleeping', 'rest', 'slumber'],
    'जागो': ['to wake', 'wake', 'awake', 'wake up', 'awaken'],
    'बरे': ['good', 'well', 'fine', 'okay', 'nice'],
    'वायट': ['bad', 'poor', 'wrong', 'evil', 'negative'],
    'व्हड': ['big', 'large', 'huge', 'great', 'enormous'],
    'ल्हान': ['small', 'little', 'tiny', 'minor', 'petite'],
    'नवो': ['new', 'fresh', 'novel', 'recent', 'modern'],
    'जुनो': ['old', 'ancient', 'aged', 'vintage', 'elderly'],
    'गरम': ['hot', 'warm', 'heated', 'spicy', 'temperature'],
    'थंड': ['cold', 'cool', 'chilly', 'freezing', 'icy'],
    'सुंदर': ['beautiful', 'pretty', 'lovely', 'gorgeous', 'attractive'],
    'काळो': ['black', 'dark', 'negro', 'ebony', 'jet black'],
    'धवो': ['white', 'fair', 'pale', 'bright', 'snow white'],
    'लाल': ['red', 'crimson', 'scarlet', 'ruby', 'vermillion'],
    'निळो': ['blue', 'azure', 'navy', 'sapphire', 'cerulean'],
    'हिरवो': ['green', 'verdant', 'emerald', 'lime', 'olive'],
    'पिवळो': ['yellow', 'golden', 'amber', 'blonde', 'lemon'],
    'हांव': ['I', 'me', 'myself', 'I am', 'my'],
    'तूं': ['you', 'yourself', 'thou', 'you are', 'your'],
    'तो': ['he', 'him', 'that man', 'himself', 'his'],
    'ती': ['she', 'her', 'that woman', 'herself', 'hers'],
    'आमी': ['we', 'us', 'ourselves', 'we are', 'our'],
    'तुमी': ['you all', 'you people', 'yourselves', 'you guys', 'your group'],
    'ते': ['they', 'them', 'those people', 'themselves', 'their'],
}

# Simple phrases (2-3 words)
SIMPLE_PHRASES = [
    ('बरे दिस', ['good day', 'good morning', 'nice day', 'fine day', 'pleasant day']),
    ('काय चालला', ['what happened', 'what\'s up', 'how are you', 'what\'s going on', 'what\'s new']),
    ('घरा वच', ['go home', 'go to house', 'return home', 'head home', 'get home']),
    ('पाणी पी', ['drink water', 'have water', 'take water', 'sip water', 'consume water']),
    ('खाणे खा', ['eat food', 'have meal', 'consume food', 'take food', 'dine']),
    ('येवप येता', ['is coming', 'will come', 'comes', 'arriving', 'on the way']),
    ('व्हड घर', ['big house', 'large home', 'huge house', 'great house', 'spacious home']),
    ('ल्हान भुरगे', ['small child', 'little kid', 'young child', 'tiny kid', 'small baby']),
    ('सुंदर बायल', ['beautiful woman', 'pretty lady', 'lovely woman', 'gorgeous girl', 'attractive female']),
    ('गरम पाणी', ['hot water', 'warm water', 'heated water', 'boiling water', 'hot liquid']),
]

# Short sentences (4-6 words)
SHORT_SENTENCES = [
    ('हांव घरा वचता', ['I am going home', 'I go home', 'I\'m heading home', 'I will go home', 'I return home']),
    ('तूं पाणी पी', ['you drink water', 'you have water', 'drink water', 'you should drink water', 'have some water']),
    ('तो खाणे खाता', ['he eats food', 'he is eating', 'he has food', 'he consumes food', 'he dines']),
    ('ती बरी आसा', ['she is good', 'she is well', 'she is fine', 'she is okay', 'she feels good']),
    ('आमी येवपाक येतात', ['we are coming', 'we will come', 'we come', 'we are arriving', 'we approach']),
    ('व्हड घर आसा', ['there is big house', 'big house exists', 'the house is big', 'it\'s a big house', 'large home is there']),
    ('ल्हान भुरगे खेळता', ['small child plays', 'little kid is playing', 'young child plays', 'the small child plays', 'tiny kid plays']),
    ('सुंदर दिस आसा', ['it is beautiful day', 'beautiful day today', 'the day is beautiful', 'lovely day', 'nice day it is']),
]


def generate_letter_data():
    """Generate letter-level training data"""
    print("\n📝 Generating letter-level data...")
    data = []
    
    for konkani_letter, english_trans in LETTER_MAPPING.items():
        if konkani_letter and english_trans:  # Skip empty
            # Add the letter itself
            data.append({
                'konkani': konkani_letter,
                'english': english_trans,
                'level': 'letter',
                'difficulty': 1
            })
            
            # Add variations with context
            data.append({
                'konkani': f'{konkani_letter} अक्षर',
                'english': f'{english_trans} letter',
                'level': 'letter_context',
                'difficulty': 1
            })
    
    print(f"  ✓ Generated {len(data)} letter examples")
    return data


def generate_word_data():
    """Generate word-level training data with multiple translations"""
    print("\n📚 Generating word-level data...")
    data = []
    
    for konkani_word, english_translations in COMMON_WORDS.items():
        # Add all translation variations
        for eng_trans in english_translations:
            data.append({
                'konkani': konkani_word,
                'english': eng_trans,
                'level': 'word',
                'difficulty': 2
            })
        
        # Add with article variations
        if english_translations[0] in ['house', 'person', 'woman', 'child', 'day', 'night', 'morning', 'evening']:
            data.append({
                'konkani': konkani_word,
                'english': f'a {english_translations[0]}',
                'level': 'word_article',
                'difficulty': 2
            })
            data.append({
                'konkani': konkani_word,
                'english': f'the {english_translations[0]}',
                'level': 'word_article',
                'difficulty': 2
            })
    
    print(f"  ✓ Generated {len(data)} word examples")
    return data


def generate_phrase_data():
    """Generate simple phrase data"""
    print("\n💬 Generating phrase-level data...")
    data = []
    
    for konkani_phrase, english_translations in SIMPLE_PHRASES:
        for eng_trans in english_translations:
            data.append({
                'konkani': konkani_phrase,
                'english': eng_trans,
                'level': 'phrase',
                'difficulty': 3
            })
    
    print(f"  ✓ Generated {len(data)} phrase examples")
    return data


def generate_sentence_data():
    """Generate short sentence data"""
    print("\n📄 Generating sentence-level data...")
    data = []
    
    for konkani_sent, english_translations in SHORT_SENTENCES:
        for eng_trans in english_translations:
            data.append({
                'konkani': konkani_sent,
                'english': eng_trans,
                'level': 'sentence',
                'difficulty': 4
            })
    
    print(f"  ✓ Generated {len(data)} sentence examples")
    return data


def load_existing_complex_data():
    """Load existing complex sentence data from other sources"""
    print("\n📦 Loading existing complex data...")
    data = []
    
    # From pretrained translations
    pretrained_path = Path('data/translation_data/konkani_english_pretrained.json')
    if pretrained_path.exists():
        with open(pretrained_path, 'r', encoding='utf-8') as f:
            pretrained = json.load(f)
            for item in pretrained:
                if item.get('english') and item['english'] != item['konkani']:
                    data.append({
                        'konkani': item['konkani'],
                        'english': item['english'],
                        'level': 'complex',
                        'difficulty': 5
                    })
        print(f"  ✓ Loaded {len(data)} from pretrained")
    
    # From augmented data
    augmented_path = Path('data/translation_data/konkani_english_augmented.json')
    if augmented_path.exists():
        with open(augmented_path, 'r', encoding='utf-8') as f:
            augmented = json.load(f)
            for item in augmented:
                data.append({
                    'konkani': item['konkani'],
                    'english': item['english'],
                    'level': 'complex',
                    'difficulty': 5
                })
        print(f"  ✓ Loaded {len(augmented)} from augmented")
    
    return data


def create_curriculum_dataset(output_path='data/translation_data/konkani_english_curriculum.json'):
    """Create complete curriculum learning dataset"""
    print("="*70)
    print("GENERATE PROGRESSIVE TRANSLATION DATA")
    print("="*70)
    
    all_data = []
    
    # Level 1: Letters
    all_data.extend(generate_letter_data())
    
    # Level 2: Words
    all_data.extend(generate_word_data())
    
    # Level 3: Phrases
    all_data.extend(generate_phrase_data())
    
    # Level 4: Sentences
    all_data.extend(generate_sentence_data())
    
    # Level 5: Complex (existing data)
    all_data.extend(load_existing_complex_data())
    
    # Statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    by_level = {}
    for item in all_data:
        level = item['level']
        by_level[level] = by_level.get(level, 0) + 1
    
    print(f"\nTotal examples: {len(all_data)}")
    print("\nBy difficulty level:")
    for level in ['letter', 'letter_context', 'word', 'word_article', 'phrase', 'sentence', 'complex']:
        count = by_level.get(level, 0)
        if count > 0:
            print(f"  {level:20s}: {count:5d} examples")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved curriculum dataset to: {output_path}")
    
    # Create sorted version (by difficulty)
    sorted_data = sorted(all_data, key=lambda x: x['difficulty'])
    sorted_path = output_path.parent / 'konkani_english_curriculum_sorted.json'
    
    with open(sorted_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved sorted curriculum dataset to: {sorted_path}")
    
    return all_data


def main():
    create_curriculum_dataset()
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Train with curriculum learning:")
    print("   python scripts/train_translation_curriculum.py")
    print("\n2. Or train with all data combined:")
    print("   python scripts/train_translation_combined.py")
    print("\n3. The model will learn progressively:")
    print("   - First: letters and their sounds")
    print("   - Then: common words with variations")
    print("   - Then: simple phrases")
    print("   - Then: short sentences")
    print("   - Finally: complex sentences")


if __name__ == '__main__':
    main()
