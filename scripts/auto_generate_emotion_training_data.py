#!/usr/bin/env python3
"""
Auto-generate emotion training data with balanced classes
Uses sentiment dictionary + text augmentation + synthetic generation
"""
import json
from pathlib import Path
from tqdm import tqdm
import random
from collections import defaultdict


def load_sentiment_dict():
    """Load sentiment dictionary"""
    dict_path = Path('data/generated/konkani_sentiment_words.json')
    with open(dict_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_synthetic_emotional_texts(sentiment_dict, num_per_emotion=500):
    """
    Generate synthetic emotional texts by combining sentiment words
    """
    print("\n" + "="*70)
    print("GENERATING SYNTHETIC EMOTIONAL TEXTS")
    print("="*70)
    
    # Emotion templates with placeholders
    templates = {
        'joy': [
            "हें {positive1} आसा, खूप {positive2}!",
            "{positive1} आणी {positive2} आसा",
            "मका {positive1} वाटता, {positive2} आसा",
            "हें {context} {positive1} आसा",
            "खूप {positive1}, {intensifier} {positive2}",
            "{positive1} जालें, {positive2} दिसता",
            "हें {context} {intensifier} {positive1} आसा",
            "{positive1} आणी {positive2}, खूप {positive3}",
        ],
        'sadness': [
            "हें {negative1} आसा, खूप {negative2}",
            "{negative1} आणी {negative2} आसा",
            "मका {negative1} वाटता, {negative2} आसा",
            "हें {context} {negative1} आसा",
            "खूप {negative1}, {intensifier} {negative2}",
            "{negative1} जालें, {negative2} दिसता",
            "हें {context} {intensifier} {negative1} आसा",
            "{negative1} आणी {negative2}, खूप {negative3}",
        ],
        'anger': [
            "हें {negative1} आसा! खूप {negative2}!",
            "{negative1}! {negative2} जालें!",
            "मका {negative1} येता, {intensifier} {negative2}!",
            "हें {context} {intensifier} {negative1} आसा!",
            "{negative1} आणी {negative2}! खूप राग येता!",
        ],
        'fear': [
            "हें {negative1} आसा, भयंकर {negative2}",
            "{negative1} दिसता, भीती वाटता",
            "मका {negative1} वाटता, डर लागता",
            "हें {context} {negative1} आसा, घोर {negative2}",
            "{intensifier} {negative1}, भयंकर आसा",
        ],
        'surprise': [
            "हें {positive1} आसा! आश्चर्य!",
            "{positive1}! विशेष आसा!",
            "मका {positive1} दिसलें, आश्चर्य जालें",
            "हें {context} {intensifier} {positive1} आसा!",
            "{positive1} आणी {positive2}! विशेष!",
        ],
        'disgust': [
            "हें {negative1} आसा, घृणा येता",
            "{negative1} आणी {negative2}, दुर्गंधी आसा",
            "मका {negative1} वाटता, घाण आसा",
            "हें {context} {negative1} आसा, घृणास्पद",
            "{intensifier} {negative1}, अपवित्र आसा",
        ],
        'neutral': [
            "हें {neutral1} आसा",
            "{neutral1} आणी {neutral2} आसा",
            "हें {context} {neutral1} आसा",
            "{neutral1} दिसता",
            "मका {neutral1} वाटता",
        ]
    }
    
    # Get word lists
    positive_words = list(sentiment_dict['positive'].keys())
    negative_words = list(sentiment_dict['negative'].keys())
    neutral_words = list(sentiment_dict['neutral'].keys())
    intensifiers = list(sentiment_dict.get('intensifiers', {}).keys())
    contexts = list(sentiment_dict.get('context_phrases', {}).keys())
    
    # Specific words for emotions
    anger_words = ['क्रूर', 'कठोर', 'राक्षसी', 'निठुर', 'क्रोध']
    fear_words = ['भयंकर', 'भीतीदायक', 'डरावन', 'घोर', 'भयाण']
    disgust_words = ['दुर्गंधी', 'कुजिल्लें', 'सडिल्लें', 'घाण', 'मळ', 'अपवित्र']
    
    synthetic_data = []
    
    for emotion, emotion_templates in templates.items():
        print(f"\nGenerating {emotion}...")
        
        for _ in tqdm(range(num_per_emotion)):
            template = random.choice(emotion_templates)
            
            # Fill template
            text = template
            
            # Replace positive words
            for i in range(1, 4):
                if f'{{positive{i}}}' in text:
                    text = text.replace(f'{{positive{i}}}', random.choice(positive_words))
            
            # Replace negative words (with emotion-specific words)
            for i in range(1, 4):
                if f'{{negative{i}}}' in text:
                    if emotion == 'anger' and anger_words:
                        word = random.choice(anger_words + negative_words)
                    elif emotion == 'fear' and fear_words:
                        word = random.choice(fear_words + negative_words)
                    elif emotion == 'disgust' and disgust_words:
                        word = random.choice(disgust_words + negative_words)
                    else:
                        word = random.choice(negative_words)
                    text = text.replace(f'{{negative{i}}}', word)
            
            # Replace neutral words
            for i in range(1, 3):
                if f'{{neutral{i}}}' in text:
                    text = text.replace(f'{{neutral{i}}}', random.choice(neutral_words))
            
            # Replace intensifiers
            if '{intensifier}' in text and intensifiers:
                text = text.replace('{intensifier}', random.choice(intensifiers))
            
            # Replace context
            if '{context}' in text and contexts:
                text = text.replace('{context}', random.choice(contexts))
            
            synthetic_data.append({
                'text': text,
                'emotion': emotion,
                'confidence': 1.0,
                'source': 'synthetic_generated',
                'needs_review': False
            })
    
    print(f"\n✓ Generated {len(synthetic_data)} synthetic texts")
    
    return synthetic_data


def augment_real_data(sentiment_dict):
    """Augment real ASR data with emotion labels"""
    print("\n" + "="*70)
    print("AUGMENTING REAL ASR DATA")
    print("="*70)
    
    # Load ASR data
    asr_manifest = Path('data/konkani-asr-v0/splits/manifests/train.json')
    
    texts = []
    with open(asr_manifest, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '').strip()
            if text and len(text) > 10:
                texts.append(text)
    
    print(f"Loaded {len(texts)} real texts")
    
    # Auto-label with improved logic
    augmented_data = []
    
    for text in tqdm(texts):
        emotion, confidence = detect_emotion_improved(text, sentiment_dict)
        
        # Only keep high-confidence or non-neutral
        if confidence >= 0.5 or emotion != 'neutral':
            augmented_data.append({
                'text': text,
                'emotion': emotion,
                'confidence': confidence,
                'source': 'asr_auto_labeled',
                'needs_review': confidence < 0.7
            })
    
    print(f"✓ Augmented {len(augmented_data)} real texts")
    
    return augmented_data


def detect_emotion_improved(text, sentiment_dict):
    """Improved emotion detection"""
    text_lower = text.lower()
    
    # Emotion-specific keywords
    emotion_keywords = {
        'joy': ['आनंद', 'खोश', 'शाबास', 'गोड', 'मीठ', 'सोबीत', 'प्रिय', 'प्रेमळ'],
        'sadness': ['उदास', 'दुःखी', 'खिन्न', 'निराश', 'हताश', 'नाउमेद'],
        'anger': ['राग', 'क्रोध', 'क्रूर', 'कठोर', 'राक्षसी', 'निठुर'],
        'fear': ['भयंकर', 'भीती', 'डर', 'घोर', 'भयाण', 'भीतीदायक'],
        'surprise': ['आश्चर्य', 'विशेष', 'अनोखा', 'दुर्मिळ'],
        'disgust': ['घृणा', 'दुर्गंधी', 'घाण', 'मळ', 'कुजिल्लें', 'सडिल्लें']
    }
    
    # Check for emotion-specific keywords first
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in text:
                return emotion, 0.8
    
    # Check sentiment words
    positive_count = sum(1 for word in sentiment_dict['positive'].keys() if word in text)
    negative_count = sum(1 for word in sentiment_dict['negative'].keys() if word in text)
    
    if positive_count > negative_count and positive_count > 0:
        return 'joy', min(0.6 + positive_count * 0.1, 0.9)
    elif negative_count > positive_count and negative_count > 0:
        return 'sadness', min(0.6 + negative_count * 0.1, 0.9)
    else:
        return 'neutral', 0.4


def create_final_training_dataset():
    """Create final balanced training dataset"""
    print("\n" + "="*70)
    print("CREATING FINAL TRAINING DATASET")
    print("="*70)
    
    # Load sentiment dictionary
    sentiment_dict = load_sentiment_dict()
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_emotional_texts(sentiment_dict, num_per_emotion=500)
    
    # Augment real data
    real_data = augment_real_data(sentiment_dict)
    
    # Combine
    all_data = synthetic_data + real_data
    
    print(f"\nTotal data: {len(all_data)}")
    
    # Group by emotion
    by_emotion = defaultdict(list)
    for item in all_data:
        by_emotion[item['emotion']].append(item)
    
    print(f"\nDistribution before balancing:")
    for emotion, items in sorted(by_emotion.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {emotion:10s}: {len(items):5d}")
    
    # Balance dataset
    target_per_class = 500
    balanced_data = []
    
    for emotion, items in by_emotion.items():
        # Sort by confidence
        items_sorted = sorted(items, key=lambda x: x['confidence'], reverse=True)
        
        # Take up to target
        selected = items_sorted[:target_per_class]
        
        # If not enough, duplicate high-confidence ones
        while len(selected) < target_per_class and len(items_sorted) > 0:
            selected.append(random.choice(items_sorted[:min(50, len(items_sorted))]))
        
        balanced_data.extend(selected[:target_per_class])
        print(f"  {emotion:10s}: {len(selected[:target_per_class]):5d} (avg conf: {sum(x['confidence'] for x in selected[:target_per_class])/len(selected[:target_per_class]):.2f})")
    
    # Shuffle
    random.shuffle(balanced_data)
    
    # Split
    n_train = int(len(balanced_data) * 0.8)
    n_val = int(len(balanced_data) * 0.1)
    
    splits = {
        'train': balanced_data[:n_train],
        'val': balanced_data[n_train:n_train+n_val],
        'test': balanced_data[n_train+n_val:]
    }
    
    # Save
    output_dir = Path('data/emotion_data/splits')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in splits.items():
        output_file = output_dir / f'{split_name}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        # Show emotion distribution
        split_emotions = defaultdict(int)
        for item in split_data:
            split_emotions[item['emotion']] += 1
        
        print(f"\n{split_name.upper()} ({len(split_data)} samples):")
        for emotion, count in sorted(split_emotions.items()):
            print(f"  {emotion:10s}: {count:4d}")
        
        print(f"✓ Saved to: {output_file}")
    
    print(f"\n✓ Final dataset ready for training!")
    print(f"  Total: {len(balanced_data)} samples")
    print(f"  ~{target_per_class} per emotion class")
    
    return balanced_data


def main():
    print("\n" + "="*70)
    print("AUTO-GENERATE EMOTION TRAINING DATA")
    print("="*70)
    
    # Create final dataset
    dataset = create_final_training_dataset()
    
    # Show samples
    print("\n" + "="*70)
    print("SAMPLE GENERATED DATA")
    print("="*70)
    
    by_emotion = defaultdict(list)
    for item in dataset:
        by_emotion[item['emotion']].append(item)
    
    for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']:
        if emotion not in by_emotion:
            continue
        
        print(f"\n{emotion.upper()}:")
        samples = random.sample(by_emotion[emotion], min(3, len(by_emotion[emotion])))
        
        for i, sample in enumerate(samples, 1):
            print(f"  [{i}] {sample['text']}")
            print(f"      Source: {sample['source']}, Confidence: {sample['confidence']:.2f}")
    
    print("\n✓ Ready to train!")
    print("\nNext step:")
    print("  python scripts/train_on_mac_gpu.py")


if __name__ == '__main__':
    main()
