import json
import os
import re
from pathlib import Path

def extract_unigrams(manifest_paths, output_path):
    """Gathers every unique word from Konkani manifesting files for pyctcdecode-unigram-support"""
    all_words = set()
    
    # Regular expression for keeping only Konkani/Devanagari characters
    # (Matches Devanagari block and some extensions)
    # \u0900-\u097F is the standard Devanagari range
    konkani_pattern = re.compile(r'[\u0900-\u097F]+')
    
    for path in manifest_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
            
        print(f"Extracting unigrams from {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    # Tokenize by whitespace
                    words = text.split()
                    for w in words:
                        # Clean words of surrounding punctuation/symbols
                        # Only keep the Devanagari part
                        # Find all Devanagari segments (e.g. "ह्या," -> "ह्या")
                        matches = konkani_pattern.findall(w)
                        for m in matches:
                            if len(m) > 1: # Basic filter
                                all_words.add(m)
                except Exception as e:
                    # Some lines might be malformed or different format
                    continue
    
    # Sort for consistency
    sorted_unigrams = sorted(list(all_words))
    
    # Ensure output dir exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for w in sorted_unigrams:
            f.write(f"{w}\n")
            
    print(f"✅ Extracted {len(sorted_unigrams)} unique unigrams to {output_path}")

if __name__ == "__main__":
    BASE = "/Volumes/data&proj/konkani"
    MANIFESTS = [
        os.path.join(BASE, "data/konkani-combined/manifests/train.json"),
        os.path.join(BASE, "data/konkani-combined/manifests/val.json"),
        os.path.join(BASE, "data/konkani-combined/manifests/test.json")
    ]
    OUT = os.path.join(BASE, "models/language_models/unigrams.txt")
    
    extract_unigrams(MANIFESTS, OUT)
