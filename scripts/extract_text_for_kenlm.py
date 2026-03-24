#!/usr/bin/env python3
"""
Extract Devanagari Konkani text from all manifest files for KenLM training
"""
import json
import os
from pathlib import Path

def extract_text_from_manifests(manifest_dirs, output_file):
    """Extract text field from all manifest JSON files"""
    texts = []
    total_files = 0
    
    for manifest_dir in manifest_dirs:
        if not os.path.exists(manifest_dir):
            print(f"Skipping {manifest_dir} - not found")
            continue
            
        print(f"Processing {manifest_dir}...")
        
        # Process all JSON files in the directory
        for json_file in Path(manifest_dir).glob("*.json"):
            # Skip macOS metadata files
            if json_file.name.startswith('._'):
                continue
                
            print(f"  Reading {json_file.name}...")
            file_count = 0
            
            with open(json_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'text' in data and data['text'].strip():
                            # Extract Devanagari text
                            text = data['text'].strip()
                            texts.append(text)
                            file_count += 1
                    except json.JSONDecodeError:
                        continue
            
            print(f"    Extracted {file_count} texts")
            total_files += file_count
    
    # Write to output file
    print(f"\nWriting {len(texts)} texts to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    
    # Print statistics
    total_words = sum(len(text.split()) for text in texts)
    print(f"\nStatistics:")
    print(f"  Total sentences: {len(texts):,}")
    print(f"  Total words: {total_words:,}")
    print(f"  Average words per sentence: {total_words/len(texts):.2f}")
    
    return len(texts), total_words

if __name__ == "__main__":
    # Define all manifest directories
    manifest_dirs = [
        "data/konkani-mega-dataset/manifests",
        "data/konkani-raw-enhanced/manifests",
        "data/konkani-10k",
        "data/konkani-final-dataset",
        "data/konkani-full",
    ]
    
    output_file = "data/konkani_corpus_for_lm.txt"
    
    print("=" * 60)
    print("Extracting Devanagari Konkani text for KenLM")
    print("=" * 60)
    
    num_sentences, num_words = extract_text_from_manifests(manifest_dirs, output_file)
    
    print(f"\n✓ Corpus saved to: {output_file}")
    print(f"✓ Ready for KenLM training!")
