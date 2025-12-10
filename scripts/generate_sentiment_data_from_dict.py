#!/usr/bin/env python3
"""
Generate sentiment/emotion training data using the sentiment words dictionary
Uses rule-based approach with sentiment words to auto-label emotions
"""
import json
from pathlib import Path
from tqdm import tqdm
import random
from collections import defaultdict


# Load sentiment dictionary
def load_sentiment_dict():
    """Load the Konkani sentiment words dictionary"""
    dict_path = Path('data/generated/konkani_sentiment_words.json')
    
    if not dict_path.exists():
        print(f"❌ Sentiment dictionary not found: {dict_path}")
        return None
    
    with open(dict_path, 'r', encoding='utf-8') as f:
        sentiment_dict = json.load(f)
    
    print(f"✓ Loaded sentiment dictionary")
    print(f"  Positive words: {len(sentiment_dict['positive'])}")
    print(f"  Negative words: {len(sentiment_dict['negative'])}")
    print(f"  Neutral words: {len(sentiment_dict['neutral'])}")
    
    return sentiment_dict


# Map sentiment to emotion
SENTIMENT_TO_EMOTION = {
    'positive': {
        'default': 'joy',
        'keywords': {
            'आनंद': 'joy',
            'खोश': 'joy',
            'शाबास': 'joy',
            'गोड': 'joy',
            'मीठ': 'joy',
            'भयंकर': 'fear',  # Can be negative context
            'आश्चर्य': 'surprise',
            'विशेष': 'surprise'
        }
    },
    'negative': {
        'default': 'sadness',
        'keywords': {
            'राग': 'anger',
            'क्रोध': 'anger',
            'क्रूर': 'anger',
            'कठोर': 'anger',
            'भयंकर': 'fear',
            'भीती': 'fear',
            'डर': 'fear',
            'घोर': 'fear',
            'उदास': 'sadness',
            'दुःखी': 'sadness',
            'निराश': 'sadness',
            'घृणा': 'disgust',
            'दुर्गंधी': 'disgust',
            'घाण': 'disgust'
        }
    },
    'neutral': {
        'default': 'neutral'
    }
}


def detect_emotion_from_text(text, sentiment_dict):
    """
    Detect emotion from Konkani text using sentiment words
    
    Returns:
        emotion: str (joy/sadness/anger/fear/surprise/disgust/neutral)
        confidence: float (0-1)
        matched_words: list of matched sentiment words
    """
    text_lower = text.lower()
    
    # Count sentiment words
    sentiment_scores = {
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }
    
    matched_words = []
    
    # Check each sentiment category
    for sentiment, words_dict in sentiment_dict.items():
        if sentiment in ['verbs', 'intensifiers', 'context_phrases']:
            continue
            
        for word, variants in words_dict.items():
            # Check main word
            if word in text:
                sentiment_scores[sentiment] += 1
                matched_words.append((word, sentiment))
            
            # Check variants
            for variant in variants:
                if variant in text_lower:
                    sentiment_scores[sentiment] += 0.5
                    matched_words.append((variant, sentiment))
    
    # Check intensifiers
    intensifier_boost = 1.0
    if 'intensifiers' in sentiment_dict:
        for intensifier, variants in sentiment_dict['intensifiers'].items():
            if intensifier in text or any(v in text_lower for v in variants):
                intensifier_boost = 1.5
                break
    
    # Apply intensifier
    for key in sentiment_scores:
        sentiment_scores[key] *= intensifier_boost
    
    # Determine dominant sentiment
    max_score = max(sentiment_scores.values())
    
    if max_score == 0:
        return 'neutral', 0.3, []
    
    dominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
    
    # Map to emotion
    emotion_map = SENTIMENT_TO_EMOTION[dominant_sentiment]
    
    # Check for specific keywords
    emotion = emotion_map['default']
    if 'keywords' in emotion_map:
        for keyword, specific_emotion in emotion_map['keywords'].items():
            if keyword in text:
                emotion = specific_emotion
                break
    
    # Calculate confidence
    total_score = sum(sentiment_scores.values())
    confidence = min(max_score / (total_score + 1), 1.0)
    
    return emotion, confidence, matched_words


def generate_emotion_data_from_asr():
    """Generate emotion-labeled data from ASR transcriptions"""
    print("\n" + "="*70)
    print("GENERATING EMOTION DATA USING SENTIMENT DICTIONARY")
    print("="*70)
    
    # Load sentiment dictionary
    sentiment_dict = load_sentiment_dict()
    if not sentiment_dict:
        return
    
    # Load ASR data
    asr_manifest = Path('data/konkani-asr-v0/splits/manifests/train.json')
    
    if not asr_manifest.exists():
        print(f"❌ ASR manifest not found: {asr_manifest}")
        return
    
    print(f"\nLoading Konkani texts from: {asr_manifest}")
    
    texts = []
    with open(asr_manifest, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '').strip()
            if text and len(text) > 10:
                texts.append(text)
    
    print(f"Found {len(texts)} texts")
    
    # Auto-label emotions
    print("\nAuto-labeling emotions...")
    
    emotion_data = []
    emotion_counts = defaultdict(int)
    high_confidence_count = 0
    
    for text in tqdm(texts):
        emotion, confidence, matched_words = detect_emotion_from_text(text, sentiment_dict)
        
        emotion_data.append({
            'text': text,
            'emotion': emotion,
            'confidence': confidence,
            'matched_words': [w[0] for w in matched_words[:5]],  # Top 5
            'source': 'auto_labeled_sentiment_dict',
            'needs_review': confidence < 0.6
        })
        
        emotion_counts[emotion] += 1
        if confidence >= 0.6:
            high_confidence_count += 1
    
    # Statistics
    print(f"\n✓ Labeled {len(emotion_data)} texts")
    print(f"\nEmotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = 100 * count / len(emotion_data)
        print(f"  {emotion:10s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\nHigh confidence (≥60%): {high_confidence_count} ({100*high_confidence_count/len(emotion_data):.1f}%)")
    print(f"Needs review (<60%): {len(emotion_data)-high_confidence_count} ({100*(len(emotion_data)-high_confidence_count)/len(emotion_data):.1f}%)")
    
    # Save data
    output_dir = Path('data/emotion_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all data
    output_file = output_dir / 'konkani_emotion_auto_labeled.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(emotion_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved to: {output_file}")
    
    # Save high-confidence subset
    high_conf_data = [d for d in emotion_data if d['confidence'] >= 0.6]
    high_conf_file = output_dir / 'konkani_emotion_high_confidence.json'
    with open(high_conf_file, 'w', encoding='utf-8') as f:
        json.dump(high_conf_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ High confidence subset: {high_conf_file}")
    
    # Save samples for manual review
    review_data = [d for d in emotion_data if d['needs_review']][:100]
    review_file = output_dir / 'konkani_emotion_needs_review.json'
    with open(review_file, 'w', encoding='utf-8') as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Needs review (100 samples): {review_file}")
    
    return emotion_data


def create_balanced_dataset():
    """Create balanced training dataset"""
    print("\n" + "="*70)
    print("CREATING BALANCED DATASET")
    print("="*70)
    
    # Load auto-labeled data
    input_file = Path('data/emotion_data/konkani_emotion_auto_labeled.json')
    
    if not input_file.exists():
        print(f"❌ Auto-labeled data not found: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # Group by emotion
    by_emotion = defaultdict(list)
    for item in all_data:
        by_emotion[item['emotion']].append(item)
    
    print(f"\nOriginal distribution:")
    for emotion, items in sorted(by_emotion.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {emotion:10s}: {len(items):4d}")
    
    # Balance dataset (undersample majority classes)
    min_count = min(len(items) for items in by_emotion.values())
    target_count = min(min_count * 2, 500)  # At most 500 per class
    
    print(f"\nBalancing to {target_count} samples per class...")
    
    balanced_data = []
    for emotion, items in by_emotion.items():
        # Prioritize high confidence
        items_sorted = sorted(items, key=lambda x: x['confidence'], reverse=True)
        selected = items_sorted[:target_count]
        balanced_data.extend(selected)
        print(f"  {emotion:10s}: {len(selected):4d} (avg conf: {sum(x['confidence'] for x in selected)/len(selected):.2f})")
    
    # Shuffle
    random.shuffle(balanced_data)
    
    # Split into train/val/test
    n_train = int(len(balanced_data) * 0.8)
    n_val = int(len(balanced_data) * 0.1)
    
    splits = {
        'train': balanced_data[:n_train],
        'val': balanced_data[n_train:n_train+n_val],
        'test': balanced_data[n_train+n_val:]
    }
    
    # Save splits
    output_dir = Path('data/emotion_data/splits')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in splits.items():
        output_file = output_dir / f'{split_name}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ {split_name:5s}: {len(split_data):4d} samples → {output_file}")
    
    print(f"\n✓ Balanced dataset created!")
    print(f"  Total: {len(balanced_data)} samples")
    print(f"  Ready for training!")


def show_samples():
    """Show sample labeled data"""
    print("\n" + "="*70)
    print("SAMPLE LABELED DATA")
    print("="*70)
    
    input_file = Path('data/emotion_data/konkani_emotion_auto_labeled.json')
    
    if not input_file.exists():
        print(f"❌ Data not found: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Show samples from each emotion
    by_emotion = defaultdict(list)
    for item in data:
        by_emotion[item['emotion']].append(item)
    
    for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']:
        if emotion not in by_emotion:
            continue
        
        print(f"\n{emotion.upper()}:")
        samples = sorted(by_emotion[emotion], key=lambda x: x['confidence'], reverse=True)[:3]
        
        for i, sample in enumerate(samples, 1):
            print(f"\n  [{i}] Confidence: {sample['confidence']:.2f}")
            print(f"      Text: {sample['text'][:80]}...")
            if sample['matched_words']:
                print(f"      Keywords: {', '.join(sample['matched_words'])}")


def main():
    print("\n" + "="*70)
    print("SENTIMENT-BASED EMOTION DATA GENERATION")
    print("="*70)
    
    print("\nOptions:")
    print("  1. Generate emotion labels from ASR data")
    print("  2. Create balanced training dataset")
    print("  3. Show sample labeled data")
    print("  4. Run all")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == '1':
        generate_emotion_data_from_asr()
    elif choice == '2':
        create_balanced_dataset()
    elif choice == '3':
        show_samples()
    elif choice == '4':
        print("\nRunning full pipeline...")
        emotion_data = generate_emotion_data_from_asr()
        if emotion_data:
            create_balanced_dataset()
            show_samples()
        print("\n✓ Complete!")
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
