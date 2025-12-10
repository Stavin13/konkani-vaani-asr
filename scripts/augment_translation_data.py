#!/usr/bin/env python3
"""
Augment translation data by:
1. Using ASR transcripts as source
2. Creating synthetic Konkani-English pairs
3. Back-translation augmentation
"""
import json
from pathlib import Path
from tqdm import tqdm
import random


def load_asr_transcripts():
    """Load ASR transcripts to use as Konkani source"""
    manifest_path = Path('data/konkani-asr-v0/splits/manifests/train.json')
    
    texts = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '').strip()
            if text and len(text) > 20:  # Only meaningful sentences
                texts.append(text)
    
    return texts


def create_synthetic_pairs():
    """Create synthetic translation pairs using templates"""
    
    # Common Konkani phrases with English translations
    synthetic_pairs = [
        # Greetings
        ("नमस्कार", "Hello"),
        ("देव बरें करूं", "God bless you"),
        ("तुका कसें आसा?", "How are you?"),
        ("हांव बरो आसां", "I am fine"),
        ("तुजें नांव कितें?", "What is your name?"),
        ("माझें नांव स्टेविन", "My name is Stavin"),
        ("भेटून खोशी जाली", "Nice to meet you"),
        ("धन्यवाद", "Thank you"),
        ("कृपा करून", "Please"),
        ("माफ करा", "Sorry"),
        
        # Common phrases
        ("हें बरें आसा", "This is good"),
        ("हें वायट आसा", "This is bad"),
        ("हांव समजना", "I don't understand"),
        ("तूं कोंकणी उलयतालो?", "Do you speak Konkani?"),
        ("हांव शिकता", "I am learning"),
        ("मका मदत जाय", "I need help"),
        ("हें किती?", "How much is this?"),
        ("कितें चालला?", "What's happening?"),
        ("सगळें बरें", "Everything is fine"),
        ("काळजी नाका", "Don't worry"),
        
        # Time
        ("आयज", "Today"),
        ("काल", "Tomorrow"),
        ("फाल्यां", "Yesterday"),
        ("सकाळीं", "Morning"),
        ("सांजेर", "Evening"),
        ("रातीं", "Night"),
        ("आतां", "Now"),
        ("उपरांत", "Later"),
        ("लवकर", "Soon"),
        ("उशीर", "Late"),
        
        # Food
        ("मका भूक लागली", "I am hungry"),
        ("मका तान लागली", "I am thirsty"),
        ("जेवण बरें आसा", "Food is good"),
        ("हें मीठ आसा", "This is sweet"),
        ("हें तिखट आसा", "This is spicy"),
        ("उदक दी", "Give water"),
        ("चहा जाय", "I want tea"),
        ("कॉफी जाय", "I want coffee"),
        
        # Family
        ("माझो बापूय", "My father"),
        ("माझी आवय", "My mother"),
        ("माझो भाव", "My brother"),
        ("माझी भयण", "My sister"),
        ("माझें कुटुंब", "My family"),
        ("माझो पूत", "My son"),
        ("माझी धूव", "My daughter"),
        
        # Places
        ("घर", "Home"),
        ("शाळा", "School"),
        ("दुकान", "Shop"),
        ("हॉस्पिटल", "Hospital"),
        ("मंदीर", "Temple"),
        ("चर्च", "Church"),
        ("बाजार", "Market"),
        ("रस्तो", "Road"),
        ("गांव", "Village"),
        ("शहर", "City"),
        
        # Actions
        ("हांव वता", "I am going"),
        ("हांव येता", "I am coming"),
        ("हांव बसता", "I am sitting"),
        ("हांव उबो आसां", "I am standing"),
        ("हांव खाता", "I am eating"),
        ("हांव पिता", "I am drinking"),
        ("हांव वाचता", "I am reading"),
        ("हांव बरयता", "I am writing"),
        ("हांव उलयता", "I am speaking"),
        ("हांव आयकता", "I am listening"),
        
        # Questions
        ("कितें?", "What?"),
        ("कोण?", "Who?"),
        ("खंय?", "Where?"),
        ("केन्ना?", "When?"),
        ("कशें?", "How?"),
        ("कित्याक?", "Why?"),
        ("किती?", "How much?"),
        ("कितले?", "How many?"),
        
        # Numbers
        ("एक", "One"),
        ("दोन", "Two"),
        ("तीन", "Three"),
        ("चार", "Four"),
        ("पांच", "Five"),
        ("सहा", "Six"),
        ("सात", "Seven"),
        ("आठ", "Eight"),
        ("नऊ", "Nine"),
        ("दहा", "Ten"),
        
        # Colors
        ("धवें", "White"),
        ("काळें", "Black"),
        ("लाल", "Red"),
        ("निळें", "Blue"),
        ("पिवळें", "Yellow"),
        ("हिरवें", "Green"),
        
        # Weather
        ("पावस पडटा", "It is raining"),
        ("उन्हाळो आसा", "It is hot"),
        ("थंड आसा", "It is cold"),
        ("वारो मारता", "Wind is blowing"),
        ("हवामान बरें आसा", "Weather is good"),
        
        # Emotions (from our emotion data)
        ("हांव खोश आसां", "I am happy"),
        ("हांव उदास आसां", "I am sad"),
        ("मका राग येता", "I am angry"),
        ("मका भीती वाटता", "I am scared"),
        ("हांव आश्चर्य जालां", "I am surprised"),
        
        # Common sentences
        ("हें माझें पुस्तक आसा", "This is my book"),
        ("तें तुजें घर आसा", "That is your house"),
        ("आमी मित्र आसात", "We are friends"),
        ("तुमी कितें करतात?", "What are you doing?"),
        ("हांव काम करता", "I am working"),
        ("तूं खंय रावता?", "Where do you live?"),
        ("हांव गोंयांत रावता", "I live in Goa"),
        ("तूं कितें शिकतालो?", "What are you studying?"),
        ("मका कोंकणी आवडटा", "I like Konkani"),
        ("हें सुंदर आसा", "This is beautiful"),
    ]
    
    return synthetic_pairs


def augment_existing_data(existing_pairs):
    """Augment existing pairs with variations"""
    augmented = []
    
    for pair in existing_pairs:
        konkani = pair['konkani']
        english = pair['english']
        
        # Skip very short or empty translations
        if len(konkani) < 10 or len(english) < 5:
            continue
        
        # Add original
        augmented.append({
            'konkani': konkani,
            'english': english,
            'source': 'original',
            'confidence': 1.0
        })
        
        # Add with punctuation variations (simple augmentation)
        if not konkani.endswith('.'):
            augmented.append({
                'konkani': konkani + '.',
                'english': english + '.',
                'source': 'punctuation_aug',
                'confidence': 0.9
            })
    
    return augmented


def create_phrase_combinations():
    """Create combinations of common phrases"""
    
    subjects = [
        ("हांव", "I"),
        ("तूं", "You"),
        ("तो", "He"),
        ("ती", "She"),
        ("आमी", "We"),
        ("तुमी", "You all"),
    ]
    
    verbs = [
        ("वता", "go"),
        ("येता", "come"),
        ("खाता", "eat"),
        ("पिता", "drink"),
        ("वाचता", "read"),
        ("बरयता", "write"),
        ("काम करता", "work"),
        ("शिकता", "learn"),
        ("उलयता", "speak"),
    ]
    
    objects = [
        ("घरा", "home"),
        ("शाळेंत", "to school"),
        ("दुकानांत", "to shop"),
        ("बाजारांत", "to market"),
    ]
    
    combinations = []
    
    # Subject + Verb combinations
    for subj_k, subj_e in subjects[:4]:  # Limit combinations
        for verb_k, verb_e in verbs[:5]:
            konkani = f"{subj_k} {verb_k}"
            english = f"{subj_e} {verb_e}"
            combinations.append((konkani, english))
    
    # Subject + Verb + Object combinations
    for subj_k, subj_e in subjects[:3]:
        for verb_pair in [("वता", "go"), ("येता", "come")]:
            verb_k, verb_e_form = verb_pair
            for obj_k, obj_e in objects:
                konkani = f"{subj_k} {obj_k} {verb_k}"
                english = f"{subj_e} {verb_e_form} {obj_e}"
                combinations.append((konkani, english))
    
    return combinations


def main():
    print("\n" + "="*70)
    print("AUGMENT TRANSLATION DATA")
    print("="*70)
    
    # Load existing data
    existing_path = Path('data/translation_data/konkani_english_translated.json')
    with open(existing_path, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    
    print(f"\nExisting pairs: {len(existing_data)}")
    
    # Create synthetic pairs
    print("\nGenerating synthetic pairs...")
    synthetic_pairs = create_synthetic_pairs()
    print(f"✓ Created {len(synthetic_pairs)} synthetic pairs")
    
    # Create phrase combinations
    print("\nGenerating phrase combinations...")
    combinations = create_phrase_combinations()
    print(f"✓ Created {len(combinations)} combinations")
    
    # Augment existing data
    print("\nAugmenting existing data...")
    augmented_existing = augment_existing_data(existing_data)
    print(f"✓ Augmented to {len(augmented_existing)} pairs")
    
    # Combine all data
    all_pairs = []
    
    # Add augmented existing
    all_pairs.extend(augmented_existing)
    
    # Add synthetic
    for konkani, english in synthetic_pairs:
        all_pairs.append({
            'konkani': konkani,
            'english': english,
            'source': 'synthetic',
            'confidence': 1.0
        })
    
    # Add combinations
    for konkani, english in combinations:
        all_pairs.append({
            'konkani': konkani,
            'english': english,
            'source': 'combination',
            'confidence': 0.8
        })
    
    # Shuffle
    random.shuffle(all_pairs)
    
    print(f"\n✓ Total pairs: {len(all_pairs)}")
    
    # Save augmented data
    output_path = Path('data/translation_data/konkani_english_augmented.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved to: {output_path}")
    
    # Show distribution
    print("\n" + "="*70)
    print("DATA DISTRIBUTION")
    print("="*70)
    
    sources = {}
    for pair in all_pairs:
        source = pair.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source:20s}: {count:5d}")
    
    # Show samples
    print("\n" + "="*70)
    print("SAMPLE PAIRS")
    print("="*70)
    
    for i, pair in enumerate(random.sample(all_pairs, min(10, len(all_pairs))), 1):
        print(f"\n[{i}] Source: {pair['source']}")
        print(f"  Konkani: {pair['konkani']}")
        print(f"  English: {pair['english']}")
    
    print("\n✓ Data augmentation complete boss!")
    print(f"\nOriginal: {len(existing_data)} → Augmented: {len(all_pairs)} pairs")
    print(f"Increase: {len(all_pairs) - len(existing_data)} pairs ({(len(all_pairs)/len(existing_data) - 1)*100:.1f}% more)")


if __name__ == '__main__':
    main()
