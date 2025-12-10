#!/usr/bin/env python3
"""
Prepare training data for NLLB fine-tuning
Combines all available Konkani-English pairs
"""
import json
from pathlib import Path
import random


def load_all_translation_data():
    """Load all available translation pairs"""
    all_data = []
    
    # 1. Curriculum data (clean, high quality)
    curriculum_path = Path('data/translation_data/konkani_english_curriculum_sorted.json')
    if curriculum_path.exists():
        with open(curriculum_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Exclude complex to avoid duplicates
            clean_data = [d for d in data if d.get('level') != 'complex']
            all_data.extend(clean_data)
            print(f"✓ Loaded {len(clean_data)} from curriculum")
    
    # 2. Google translated data (good quality)
    pretrained_path = Path('data/translation_data/konkani_english_pretrained.json')
    if pretrained_path.exists():
        with open(pretrained_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Only successful translations
            clean_data = [d for d in data if d.get('method') == 'google_translate' 
                         and d['english'] != d['konkani']]
            all_data.extend(clean_data)
            print(f"✓ Loaded {len(clean_data)} from Google Translate")
    
    # 3. Manual translations (if you have any)
    manual_path = Path('data/translation_data/konkani_english_manual.json')
    if manual_path.exists():
        with open(manual_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.extend(data)
            print(f"✓ Loaded {len(data)} manual translations")
    
    return all_data


def clean_and_validate(data):
    """Clean and validate translation pairs"""
    cleaned = []
    
    for item in data:
        konkani = item.get('konkani', '').strip()
        english = item.get('english', '').strip()
        
        # Skip if empty or too short
        if not konkani or not english or len(konkani) < 2 or len(english) < 2:
            continue
        
        # Skip if identical (no translation)
        if konkani == english:
            continue
        
        # Skip if too long (NLLB limit)
        if len(konkani) > 500 or len(english) > 500:
            continue
        
        cleaned.append({
            'konkani': konkani,
            'english': english
        })
    
    return cleaned


def split_data(data, train_ratio=0.85, val_ratio=0.10):
    """Split into train/val/test"""
    random.seed(42)
    random.shuffle(data)
    
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train = data[:n_train]
    val = data[n_train:n_train+n_val]
    test = data[n_train+n_val:]
    
    return train, val, test


def save_for_nllb(data, output_path):
    """Save in format for NLLB fine-tuning"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(data)} pairs to {output_path}")


def main():
    print("="*70)
    print("PREPARE NLLB FINE-TUNING DATA")
    print("="*70)
    
    # Load all data
    print("\nLoading translation data...")
    all_data = load_all_translation_data()
    print(f"\nTotal loaded: {len(all_data)} pairs")
    
    # Clean and validate
    print("\nCleaning and validating...")
    cleaned_data = clean_and_validate(all_data)
    print(f"After cleaning: {len(cleaned_data)} pairs")
    
    # Split
    print("\nSplitting data...")
    train, val, test = split_data(cleaned_data)
    
    print(f"  Train: {len(train)} pairs (85%)")
    print(f"  Val:   {len(val)} pairs (10%)")
    print(f"  Test:  {len(test)} pairs (5%)")
    
    # Save
    print("\nSaving datasets...")
    output_dir = Path('data/nllb_finetuning')
    save_for_nllb(train, output_dir / 'train.json')
    save_for_nllb(val, output_dir / 'val.json')
    save_for_nllb(test, output_dir / 'test.json')
    
    # Statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    print(f"\nTotal pairs: {len(cleaned_data)}")
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")
    
    # Sample
    print("\nSample training pairs:")
    for i, pair in enumerate(train[:5]):
        print(f"\n{i+1}. Konkani: {pair['konkani']}")
        print(f"   English: {pair['english']}")
    
    print("\n" + "="*70)
    print("NEXT STEP: Fine-tune NLLB")
    print("  python scripts/finetune_nllb.py")
    print("="*70)


if __name__ == '__main__':
    main()
