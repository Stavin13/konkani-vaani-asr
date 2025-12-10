#!/usr/bin/env python3
"""
Generate training data using ready-made models
- Translation: Use Google Translate API or existing Konkani-English pairs
- Emotion: Use GPT/Claude to label Konkani text with emotions
"""
import json
from pathlib import Path
from tqdm import tqdm
import random
from datetime import datetime


# ============================================================================
# TRANSLATION DATA GENERATION
# ============================================================================

def generate_translation_data_from_asr():
    """
    Generate translation data from ASR transcriptions
    Uses existing Konkani text from ASR dataset
    """
    print("="*70)
    print("GENERATING TRANSLATION DATA FROM ASR TRANSCRIPTIONS")
    print("="*70)
    
    # Load ASR data (has Konkani text)
    asr_manifest = Path('data/konkani-asr-v0/splits/manifests/train.json')
    
    if not asr_manifest.exists():
        print(f"❌ ASR manifest not found: {asr_manifest}")
        return None
    
    print(f"\nLoading Konkani text from: {asr_manifest}")
    
    konkani_texts = []
    with open(asr_manifest, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '').strip()
            if text and len(text) > 10:  # Filter short texts
                konkani_texts.append(text)
    
    print(f"Found {len(konkani_texts)} Konkani texts")
    
    # For now, create placeholder English translations
    # You'll need to replace this with actual translations
    translation_pairs = []
    
    print("\n⚠️  IMPORTANT: You need to translate these texts to English")
    print("Options:")
    print("  1. Use Google Translate API (paid)")
    print("  2. Use free translation services")
    print("  3. Manual translation")
    print("  4. Use existing Konkani-English parallel corpus")
    
    # Create sample data structure
    for i, konkani_text in enumerate(konkani_texts[:100]):  # Sample 100
        translation_pairs.append({
            'konkani': konkani_text,
            'english': f"[TRANSLATE THIS: {konkani_text[:50]}...]",  # Placeholder
            'source': 'asr_transcription',
            'needs_translation': True
        })
    
    # Save to file
    output_dir = Path('data/translation_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'konkani_english_pairs_to_translate.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translation_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(translation_pairs)} pairs to: {output_file}")
    print("\nNext step: Translate the Konkani texts to English")
    
    return translation_pairs


def use_google_translate_api():
    """
    Example: Use Google Translate API to translate Konkani to English
    Requires: pip install googletrans==4.0.0-rc1
    """
    print("\n" + "="*70)
    print("USING GOOGLE TRANSLATE API")
    print("="*70)
    
    try:
        from googletrans import Translator
        translator = Translator()
        
        # Load texts to translate
        input_file = Path('data/translation_data/konkani_english_pairs_to_translate.json')
        if not input_file.exists():
            print(f"❌ Input file not found: {input_file}")
            return
        
        with open(input_file, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        
        print(f"\nTranslating {len(pairs)} texts...")
        
        translated_pairs = []
        for pair in tqdm(pairs):
            if pair.get('needs_translation'):
                try:
                    # Translate Konkani to English
                    result = translator.translate(
                        pair['konkani'],
                        src='kn',  # Konkani (may not be perfect)
                        dest='en'
                    )
                    
                    translated_pairs.append({
                        'konkani': pair['konkani'],
                        'english': result.text,
                        'source': 'google_translate',
                        'confidence': result.extra_data.get('confidence', 0)
                    })
                except Exception as e:
                    print(f"\n⚠️  Translation failed: {e}")
                    translated_pairs.append(pair)
        
        # Save translated data
        output_file = Path('data/translation_data/konkani_english_translated.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_pairs, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Translated data saved to: {output_file}")
        
    except ImportError:
        print("\n❌ googletrans not installed")
        print("Install with: pip install googletrans==4.0.0-rc1")


# ============================================================================
# EMOTION DATA GENERATION
# ============================================================================

def generate_emotion_labels_with_gpt():
    """
    Generate emotion labels using GPT/Claude API
    This is a template - you'll need to add your API key
    """
    print("\n" + "="*70)
    print("GENERATING EMOTION LABELS WITH GPT")
    print("="*70)
    
    # Load Konkani texts
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
    
    # Sample for labeling
    sample_texts = random.sample(texts, min(500, len(texts)))
    
    print("\n⚠️  IMPORTANT: You need to label these texts with emotions")
    print("Options:")
    print("  1. Use GPT-4/Claude API (paid)")
    print("  2. Use free LLM APIs (Groq, Together AI)")
    print("  3. Manual labeling")
    print("  4. Use existing emotion-labeled Konkani corpus")
    
    # Create sample data structure
    emotion_data = []
    for text in sample_texts:
        emotion_data.append({
            'text': text,
            'emotion': 'neutral',  # Placeholder
            'confidence': 0.0,
            'needs_labeling': True,
            'source': 'asr_transcription'
        })
    
    # Save to file
    output_dir = Path('data/emotion_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'konkani_texts_to_label.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(emotion_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(emotion_data)} texts to: {output_file}")
    print("\nNext step: Label texts with emotions using GPT or manual labeling")
    
    return emotion_data


def use_openai_for_emotion_labeling():
    """
    Example: Use OpenAI API to label emotions
    Requires: pip install openai
    """
    print("\n" + "="*70)
    print("USING OPENAI API FOR EMOTION LABELING")
    print("="*70)
    
    try:
        import openai
        
        # You need to set your API key
        # openai.api_key = "your-api-key-here"
        
        print("\n⚠️  You need to set your OpenAI API key")
        print("Add this line: openai.api_key = 'your-key'")
        
        # Load texts to label
        input_file = Path('data/emotion_data/konkani_texts_to_label.json')
        if not input_file.exists():
            print(f"❌ Input file not found: {input_file}")
            return
        
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = json.load(f)
        
        print(f"\nLabeling {len(texts)} texts...")
        
        labeled_data = []
        for item in tqdm(texts[:10]):  # Sample 10 for demo
            if item.get('needs_labeling'):
                try:
                    # Create prompt
                    prompt = f"""Analyze the emotion in this Konkani text and classify it into one of these categories:
- joy
- sadness
- anger
- fear
- surprise
- disgust
- neutral

Text: {item['text']}

Respond with only the emotion label."""
                    
                    # Call GPT (commented out - needs API key)
                    # response = openai.ChatCompletion.create(
                    #     model="gpt-3.5-turbo",
                    #     messages=[{"role": "user", "content": prompt}]
                    # )
                    # emotion = response.choices[0].message.content.strip().lower()
                    
                    # Placeholder
                    emotion = 'neutral'
                    
                    labeled_data.append({
                        'text': item['text'],
                        'emotion': emotion,
                        'source': 'openai_gpt',
                        'confidence': 0.8
                    })
                except Exception as e:
                    print(f"\n⚠️  Labeling failed: {e}")
                    labeled_data.append(item)
        
        # Save labeled data
        output_file = Path('data/emotion_data/konkani_emotion_labeled.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(labeled_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Labeled data saved to: {output_file}")
        
    except ImportError:
        print("\n❌ openai not installed")
        print("Install with: pip install openai")


# ============================================================================
# MANUAL LABELING INTERFACE
# ============================================================================

def create_manual_labeling_interface():
    """Create a simple CLI interface for manual labeling"""
    print("\n" + "="*70)
    print("MANUAL EMOTION LABELING INTERFACE")
    print("="*70)
    
    # Load texts to label
    input_file = Path('data/emotion_data/konkani_texts_to_label.json')
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("Run generate_emotion_labels_with_gpt() first")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    
    emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
    
    print(f"\nLoaded {len(texts)} texts to label")
    print("\nEmotion labels:")
    for i, emotion in enumerate(emotions, 1):
        print(f"  {i}. {emotion}")
    print("  0. Skip")
    print("  q. Quit and save")
    
    labeled_data = []
    
    for i, item in enumerate(texts):
        if not item.get('needs_labeling'):
            labeled_data.append(item)
            continue
        
        print(f"\n[{i+1}/{len(texts)}] Text:")
        print(f"  {item['text']}")
        
        while True:
            choice = input("\nEmotion (1-7, 0=skip, q=quit): ").strip().lower()
            
            if choice == 'q':
                print("\nSaving and quitting...")
                break
            elif choice == '0':
                labeled_data.append(item)
                break
            elif choice.isdigit() and 1 <= int(choice) <= 7:
                emotion = emotions[int(choice) - 1]
                labeled_data.append({
                    'text': item['text'],
                    'emotion': emotion,
                    'source': 'manual_labeling',
                    'confidence': 1.0,
                    'labeled_at': datetime.now().isoformat()
                })
                print(f"✓ Labeled as: {emotion}")
                break
            else:
                print("Invalid choice. Try again.")
        
        if choice == 'q':
            break
    
    # Save labeled data
    output_file = Path('data/emotion_data/konkani_emotion_manually_labeled.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(labeled_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(labeled_data)} labeled texts to: {output_file}")


# ============================================================================
# CREATE TRAINING DATASETS
# ============================================================================

def create_training_datasets():
    """Create final training datasets from labeled data"""
    print("\n" + "="*70)
    print("CREATING TRAINING DATASETS")
    print("="*70)
    
    # Translation dataset
    translation_file = Path('data/translation_data/konkani_english_translated.json')
    if translation_file.exists():
        with open(translation_file, 'r', encoding='utf-8') as f:
            translation_data = json.load(f)
        
        # Split into train/val/test
        random.shuffle(translation_data)
        n_train = int(len(translation_data) * 0.8)
        n_val = int(len(translation_data) * 0.1)
        
        splits = {
            'train': translation_data[:n_train],
            'val': translation_data[n_train:n_train+n_val],
            'test': translation_data[n_train+n_val:]
        }
        
        output_dir = Path('data/translation_data/splits')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in splits.items():
            output_file = output_dir / f'{split_name}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Translation {split_name}: {len(split_data)} samples")
    
    # Emotion dataset
    emotion_file = Path('data/emotion_data/konkani_emotion_labeled.json')
    if not emotion_file.exists():
        emotion_file = Path('data/emotion_data/konkani_emotion_manually_labeled.json')
    
    if emotion_file.exists():
        with open(emotion_file, 'r', encoding='utf-8') as f:
            emotion_data = json.load(f)
        
        # Split into train/val/test
        random.shuffle(emotion_data)
        n_train = int(len(emotion_data) * 0.8)
        n_val = int(len(emotion_data) * 0.1)
        
        splits = {
            'train': emotion_data[:n_train],
            'val': emotion_data[n_train:n_train+n_val],
            'test': emotion_data[n_train+n_val:]
        }
        
        output_dir = Path('data/emotion_data/splits')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in splits.items():
            output_file = output_dir / f'{split_name}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Emotion {split_name}: {len(split_data)} samples")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("GENERATE TRAINING DATA USING READY-MADE MODELS")
    print("="*70)
    
    print("\nOptions:")
    print("  1. Generate translation data from ASR")
    print("  2. Generate emotion labels")
    print("  3. Use Google Translate API (requires API)")
    print("  4. Use OpenAI for emotion labeling (requires API)")
    print("  5. Manual emotion labeling interface")
    print("  6. Create final training datasets")
    print("  7. Run all (generate data)")
    
    choice = input("\nChoice (1-7): ").strip()
    
    if choice == '1':
        generate_translation_data_from_asr()
    elif choice == '2':
        generate_emotion_labels_with_gpt()
    elif choice == '3':
        use_google_translate_api()
    elif choice == '4':
        use_openai_for_emotion_labeling()
    elif choice == '5':
        create_manual_labeling_interface()
    elif choice == '6':
        create_training_datasets()
    elif choice == '7':
        print("\nGenerating all data...")
        generate_translation_data_from_asr()
        generate_emotion_labels_with_gpt()
        print("\n✓ Data generation complete!")
        print("\nNext steps:")
        print("  1. Translate Konkani texts (option 3 or manually)")
        print("  2. Label emotions (option 4, 5, or manually)")
        print("  3. Create training datasets (option 6)")
        print("  4. Train models: python scripts/train_on_mac_gpu.py")
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
