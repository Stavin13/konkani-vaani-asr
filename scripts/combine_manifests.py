#!/usr/bin/env python3
import json, os
from pathlib import Path

def merge(files, out_path):
    unique_samples = {}
    total_loaded = 0
    for f in files:
        if not os.path.exists(f): 
            print(f"Skipping missing: {f}")
            continue
        with open(f, 'r', encoding='utf-8') as src:
            for line in src:
                total_loaded += 1
                s = json.loads(line)
                # Use filename as unique key to prevent duplicates
                key = os.path.basename(s['audio_filepath'])
                if key not in unique_samples:
                    unique_samples[key] = s
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as out:
        for s in unique_samples.values():
            out.write(json.dumps(s, ensure_ascii=False) + '\n')
    
    total_dur = sum(s.get('duration', 0) for s in unique_samples.values())
    print(f"Merged {files} -> {out_path}")
    print(f"  Total Lines Loaded: {total_loaded}")
    print(f"  Unique Samples: {len(unique_samples)}")
    print(f"  Total Duration: {total_dur/3600:.2f}h")

if __name__ == "__main__":
    # 110-Hour Train
    merge([
        'data/konkani-raw-corpus/manifests/train.json',
        'data/konkani-asr-v0/splits/manifests/train.json'
    ], 'data/konkani-ultimate/train.json')
    
    # 110-Hour Val
    merge([
        'data/konkani-raw-corpus/manifests/val.json',
        'data/konkani-asr-v0/splits/manifests/val.json'
    ], 'data/konkani-ultimate/val.json')
