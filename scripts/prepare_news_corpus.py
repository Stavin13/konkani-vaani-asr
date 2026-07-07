#!/usr/bin/env python3
import json
import os
from pathlib import Path
import unicodedata

def normalize_text(text):
    return unicodedata.normalize('NFC', text).strip()

def main():
    BASE = Path("/Volumes/data&proj/konkani")
    # 1. Existing corpus if it exists
    existing_corpus = BASE / "data/konkani_corpus_for_lm.txt"
    texts = []
    if existing_corpus.exists():
        print(f"Loading existing corpus from {existing_corpus}")
        with open(existing_corpus, 'r', encoding='utf-8') as f:
            texts.extend([normalize_text(line) for line in f if line.strip()])
    
    # 2. Add Newsonair Cross-Val data (the critical domain adaptation)
    news_json = BASE / "data/cross_val_newsonair/newsonair_konkani_external_aligned_lab_02-09-2021_06-55/data.json"
    if news_json.exists():
        print(f"Extracting news data from {news_json}")
        with open(news_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if 'text' in item and item['text']:
                    texts.append(normalize_text(item['text']))
    
    # 3. Add other common manifestations
    manifest_dirs = [
        BASE / "data/konkani-10k",
    ]
    for d in manifest_dirs:
        for json_file in d.glob("*.json"):
            if json_file.name.startswith('._'): continue
            print(f"Reading {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    content = f.read().strip()
                    if content.startswith('['):
                        data = json.loads(content)
                        for item in data:
                            if 'text' in item: texts.append(normalize_text(item['text']))
                    else:
                        for line in content.splitlines():
                            data = json.loads(line)
                            if 'text' in data: texts.append(normalize_text(data['text']))
                except:
                    pass

    # Unique and save
    texts = sorted(list(set(texts)))
    output_path = BASE / "data/konkani_expanded_corpus.txt"
    print(f"Saving {len(texts)} unique sentences to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for t in texts:
            f.write(t + '\n')

if __name__ == "__main__":
    main()
