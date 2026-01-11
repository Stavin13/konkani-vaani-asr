#!/usr/bin/env python3
"""
Clean Datasets for Standard Vocabulary
=====================================
Remove or replace characters not in standard vocabulary
"""
import json
import unicodedata
from pathlib import Path
from collections import Counter
import argparse
import re

# Standard vocabulary characters
STANDARD_CHARS = {
    "<pad>", "<blank>", "<sos>", "<eos>", "<unk>",
    " ", "!", ",", "-", ".", "?",
    "ँ", "ं", "ः", "अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ए", "ऐ", "ऑ", "ओ", "औ",
    "क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण",
    "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ऱ", "ल", "ळ",
    "व", "श", "ष", "स", "ह", "ा", "ि", "ी", "ु", "ू", "ृ", "ॅ", "े", "ै", "ॉ",
    "ो", "ौ", "्", "१", "२", "९", "'", "'", """, """
}

# Character replacement mappings
CHAR_REPLACEMENTS = {
    # Smart quotes to regular quotes
    "'": "'", "'": "'", """: '"', """: '"',
    
    # Punctuation normalization
    "–": "-", "—": "-", "…": ".",
    
    # Remove parentheses and other punctuation
    "(": "", ")": "", "/": "", ":": "",
    
    # Devanagari nukta combinations (remove nukta)
    "़": "",
    
    # Other Devanagari variants
    "ॠ": "ऋ",
    
    # Remove Latin characters (replace with space or remove)
    **{chr(i): "" for i in range(ord('a'), ord('z')+1)},  # a-z
    **{chr(i): "" for i in range(ord('A'), ord('Z')+1)},  # A-Z
    **{chr(i): "" for i in range(ord('0'), ord('9')+1)},  # 0-9
}

# Add mappings for other scripts (Arabic, Bengali, Odia, etc.) - remove them
OTHER_SCRIPTS = [
    # Arabic/Urdu
    'ا', 'ب', 'ت', 'ج', 'خ', 'د', 'ر', 'ز', 'س', 'ش', 'غ', 'ف', 'ل', 'م', 'ن', 'ه', 'و', 'ي',
    'آ', 'ؤ', 'ئ', 'ٹ', 'پ', 'چ', 'ڈ', 'ڑ', 'ک', 'گ', 'ں', 'ھ', 'ہ', 'ی', 'ے',
    
    # Bengali
    'ঁ', 'ং', 'অ', 'আ', 'ই', 'উ', 'এ', 'ও', 'ক', 'খ', 'গ', 'চ', 'ছ', 'জ', 'ট', 'ঠ', 'ড',
    'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ম', 'য', 'র', 'ল', 'শ', 'স', 'হ', '়', 'া',
    'ি', 'ু', 'ে', 'ো', '্', 'ৰ',
    
    # Odia
    'ଂ', 'ଆ', 'କ', 'ଗ', 'ଜ', 'ଟ', 'ଡ', 'ତ', 'ଥ', 'ଦ', 'ନ', 'ଫ', 'ବ', 'ମ', 'ର', 'ଲ', 'ଵ',
    'ଶ', 'ଷ', 'ସ', 'ହ', 'ା', 'ି', 'ୁ', 'ୂ', 'େ', 'ୈ', 'ୋ',
    
    # Gujarati
    'સ',
    
    # Tamil
    'ஜ',
]

# Add other scripts to replacements (remove them)
for char in OTHER_SCRIPTS:
    CHAR_REPLACEMENTS[char] = ""


def clean_text(text):
    """Clean text to only contain standard vocabulary characters"""
    if not text:
        return text
    
    # Apply character replacements
    cleaned = text
    for old_char, new_char in CHAR_REPLACEMENTS.items():
        cleaned = cleaned.replace(old_char, new_char)
    
    # Remove any remaining non-standard characters
    result = ""
    for char in cleaned:
        if char in STANDARD_CHARS:
            result += char
        # Skip characters not in standard vocab
    
    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result


def clean_manifest(input_path, output_path):
    """Clean a manifest file"""
    if not input_path.exists():
        return 0, 0
    
    cleaned_samples = []
    skipped_count = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line:
                        continue
                        
                    sample = json.loads(line)
                    original_text = sample.get('text', '')
                    cleaned_text = clean_text(original_text)
                    
                    # Skip samples with empty or very short text after cleaning
                    if len(cleaned_text.strip()) < 2:
                        skipped_count += 1
                        continue
                    
                    # Update sample with cleaned text
                    sample['text'] = cleaned_text
                    cleaned_samples.append(sample)
                    
                except Exception as e:
                    skipped_count += 1
                    continue
    except Exception as e:
        print(f"    Error reading file: {e}")
        return 0, 0
    
    # Save cleaned manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in cleaned_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    return len(cleaned_samples), skipped_count


def clean_dataset(dataset_dir, output_dir):
    """Clean all manifests in a dataset"""
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    
    print(f"\nCleaning dataset: {dataset_path}")
    print(f"Output: {output_path}")
    print("-" * 50)
    
    # Find manifest files
    manifest_files = []
    for pattern in ['*.json']:
        manifest_files.extend(dataset_path.glob(pattern))
    
    total_cleaned = 0
    total_skipped = 0
    
    for manifest_file in manifest_files:
        output_file = output_path / manifest_file.name
        
        cleaned_count, skipped_count = clean_manifest(manifest_file, output_file)
        total_cleaned += cleaned_count
        total_skipped += skipped_count
        
        print(f"  {manifest_file.name:20s}: {cleaned_count:6,} cleaned, {skipped_count:4,} skipped")
    
    print(f"  Total: {total_cleaned:,} samples cleaned, {total_skipped:,} skipped")
    
    return total_cleaned, total_skipped


def main():
    parser = argparse.ArgumentParser(description='Clean datasets for standard vocabulary')
    parser.add_argument('--input_datasets', nargs='+',
                       default=[
                           'data/konkani-mega-dataset/manifests',
                           'data/konkani-raw-enhanced/manifests',
                           'data/konkani-10k',
                           'data/konkani-asr-v0/splits/manifests'
                       ],
                       help='Input dataset directories')
    parser.add_argument('--output_dir', default='data/cleaned-datasets',
                       help='Output directory for cleaned datasets')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be cleaned without making changes')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CLEANING DATASETS FOR STANDARD VOCABULARY")
    print("="*60)
    print(f"Standard vocabulary: {len(STANDARD_CHARS)} characters")
    print(f"Character replacements: {len(CHAR_REPLACEMENTS)} mappings")
    
    if args.dry_run:
        print("\n[DRY RUN] - No files will be modified")
    
    total_all_cleaned = 0
    total_all_skipped = 0
    
    for dataset_dir in args.input_datasets:
        dataset_path = Path(dataset_dir)
        
        if not dataset_path.exists():
            print(f"\n⚠️  Dataset not found: {dataset_path}")
            continue
        
        # Create output directory name
        dataset_name = dataset_path.parent.name if dataset_path.name == 'manifests' else dataset_path.name
        output_dataset_dir = Path(args.output_dir) / f"{dataset_name}-cleaned"
        
        if args.dry_run:
            print(f"\nWould clean: {dataset_path} -> {output_dataset_dir}")
            continue
        
        cleaned_count, skipped_count = clean_dataset(dataset_path, output_dataset_dir)
        total_all_cleaned += cleaned_count
        total_all_skipped += skipped_count
    
    if not args.dry_run:
        print("\n" + "="*60)
        print("CLEANING COMPLETE")
        print("="*60)
        print(f"✓ Total samples cleaned: {total_all_cleaned:,}")
        print(f"✗ Total samples skipped: {total_all_skipped:,}")
        print(f"\nCleaned datasets saved to: {args.output_dir}")
        
        print(f"\nNext steps:")
        print(f"1. Validate cleaned datasets:")
        print(f"   python scripts/validate_dataset_vocab.py --datasets {args.output_dir}/*/")
        print(f"2. Use cleaned datasets for training")
        print(f"3. All text should now be compatible with standard vocabulary")


if __name__ == '__main__':
    main()