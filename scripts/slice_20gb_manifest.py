#!/usr/bin/env python3
"""
Slice ~20GB worth of samples from the full corpus manifests.
Produces train/val/test manifests for the 20GB chunk.
Run: python scripts/slice_20gb_manifest.py
"""
import json, os, random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────
TARGET_GB      = 20.0
AVG_FILE_KB    = 1447.8          # measured avg from corpus
AVG_DUR_S      = 4.21            # measured avg duration
TARGET_SAMPLES = int((TARGET_GB * 1024**3) / (AVG_FILE_KB * 1024))

VAL_RATIO      = 0.10
TEST_RATIO     = 0.05

SOURCE_MANIFESTS = [
    'data/konkani-raw-corpus/manifests/train.json',
    'data/konkani-raw-corpus/manifests/val.json',
    'data/konkani-raw-corpus/manifests/test.json',
]
OUTPUT_DIR = 'data/konkani-20gb'

BASE_DIR = Path(__file__).resolve().parent.parent

def remap_path(unix_path: str) -> str:
    if os.path.exists(unix_path):
        return unix_path
    for prefix in ['/Volumes/data&proj/konkani/', '/Volumes/data&proj/konkani', '/Volumes/']:
        if unix_path.startswith(prefix):
            rel = unix_path[len(prefix):]
            candidate = BASE_DIR / rel.replace('/', os.sep)
            if candidate.exists():
                return str(candidate)
            parts = rel.split('/', 1)
            if len(parts) > 1:
                candidate2 = BASE_DIR / parts[1].replace('/', os.sep)
                if candidate2.exists():
                    return str(candidate2)
    fname = os.path.basename(unix_path)
    corpus_dir = BASE_DIR / 'KonkaniRawSpeechCorpus'
    if corpus_dir.exists():
        for root, _, files in os.walk(corpus_dir):
            if fname in files:
                return os.path.join(root, fname)
    return ''

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all samples from all splits
    all_samples = []
    for mf in SOURCE_MANIFESTS:
        if not os.path.exists(mf):
            print(f'  Skipping missing: {mf}')
            continue
        with open(mf, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                    local = remap_path(s['audio_filepath'])
                    if local:
                        s['audio_filepath'] = local
                        all_samples.append(s)
                except Exception:
                    pass
        print(f'  Loaded {mf}')

    print(f'\nTotal resolvable samples: {len(all_samples):,}')
    print(f'Target samples for ~{TARGET_GB}GB: {TARGET_SAMPLES:,}')

    # Shuffle and slice
    random.seed(42)
    random.shuffle(all_samples)
    chunk = all_samples[:TARGET_SAMPLES]

    total_dur = sum(s.get('duration', AVG_DUR_S) for s in chunk)
    print(f'Chunk: {len(chunk):,} samples | ~{total_dur/3600:.1f}h')

    # Split into train / val / test
    n_test  = int(len(chunk) * TEST_RATIO)
    n_val   = int(len(chunk) * VAL_RATIO)
    n_train = len(chunk) - n_val - n_test

    train_samples = chunk[:n_train]
    val_samples   = chunk[n_train:n_train + n_val]
    test_samples  = chunk[n_train + n_val:]

    splits = {
        'train': train_samples,
        'val':   val_samples,
        'test':  test_samples,
    }

    for split_name, samples in splits.items():
        out_path = os.path.join(OUTPUT_DIR, f'{split_name}.json')
        dur = sum(s.get('duration', AVG_DUR_S) for s in samples)
        with open(out_path, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print(f'  {split_name:5s}: {len(samples):,} samples | {dur/3600:.2f}h → {out_path}')

    print(f'\nDone. Manifests saved to: {OUTPUT_DIR}/')
    print('Update CONFIG in train_conformer_v2.py:')
    print(f'  train_manifest: data/konkani-20gb/train.json')
    print(f'  val_manifest  : data/konkani-20gb/val.json')

if __name__ == '__main__':
    main()
