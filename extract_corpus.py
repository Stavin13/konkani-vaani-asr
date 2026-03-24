import os
import json
import re
from tqdm import tqdm

MANIFEST_DIRS = [
    'data/kaggle_manifests',
    'data/konkani-20gb',
    'data/konkani-10k',
    'data/konkani-final-dataset'
]

OUTPUT_FILE = 'full_konkani_corpus.txt'

def clean_text(text):
    # Remove any transliteration tags or noise
    text = re.sub(r'\[.*?\]', '', text) # Remove [noise], [overlap] etc.
    text = re.sub(r'TEXT TRANSLITERATION :: .*', '', text)
    # Filter for Devanagari characters, spaces and standard punctuation
    # Devanagari range: \u0900-\u097f
    text = "".join([c for c in text if '\u0900' <= c <= '\u097f' or c in ' \n.,|?!'])
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()

def extract():
    total_sentences = 0
    total_words = 0
    unique_sentences = set()
    
    print(f"Scanning manifests in: {MANIFEST_DIRS}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for d in MANIFEST_DIRS:
            if not os.path.exists(d): continue
            
            for f_name in os.listdir(d):
                if f_name.endswith('.json') and f_name != 'vocab.json':
                    path = os.path.join(d, f_name)
                    print(f"  Extracting: {path}")
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            try:
                                item = json.loads(line)
                                if 'text' in item:
                                    clean = clean_text(item['text'])
                                    if clean and len(clean) > 2:
                                        if clean not in unique_sentences:
                                            out_f.write(clean + '\n')
                                            unique_sentences.add(clean)
                                            total_sentences += 1
                                            total_words += len(clean.split())
                            except:
                                continue
                                
    print(f"\n{'='*50}")
    print(f"CORPUS SUMMARY:")
    print(f"  Total Unique Sentences: {total_sentences:,}")
    print(f"  Estimated Total Words  : {total_words:,}")
    print(f"  Saved to: {OUTPUT_FILE}")
    print(f"{'='*50}")

if __name__ == '__main__':
    extract()
