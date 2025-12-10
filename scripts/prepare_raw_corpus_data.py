#!/usr/bin/env python3
"""
Prepare KonkaniRawSpeechCorpus data for training
Creates manifest files from the raw corpus
"""
import json
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
import random

def parse_transcript_file(txt_path):
    """Parse the transcript .txt file"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the recorded text (Devanagari)
    if 'RECORDED TEXT ::' in content:
        parts = content.split('RECORDED TEXT ::')
        if len(parts) > 1:
            text_part = parts[1].split('TEXT TRANSLITERATION ::')[0]
            text = text_part.strip()
            return text
    return None

def create_manifest_from_corpus(corpus_dir='KonkaniRawSpeechCorpus/Data',
                                output_dir='data/konkani-raw-corpus'):
    """Create manifest files from raw corpus"""
    
    corpus_path = Path(corpus_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PREPARING KONKANIRAWS PEECHCORPUS DATA")
    print("="*70)
    
    # Find all audio files
    print("\nFinding audio files...")
    wav_files = list(corpus_path.rglob('*.wav'))
    print(f"Found {len(wav_files):,} audio files")
    
    # Process each file
    print("\nProcessing files...")
    samples = []
    skipped = 0
    
    for wav_path in tqdm(wav_files):
        # Find corresponding transcript
        txt_path = wav_path.with_suffix('.txt')
        
        if not txt_path.exists():
            skipped += 1
            continue
        
        # Parse transcript
        text = parse_transcript_file(txt_path)
        if not text or len(text.strip()) == 0:
            skipped += 1
            continue
        
        # Get audio info
        try:
            info = sf.info(wav_path)
            duration = info.duration
            
            # Skip very short or very long files
            if duration < 0.5 or duration > 15.0:
                skipped += 1
                continue
            
            samples.append({
                'audio_filepath': str(wav_path.absolute()),
                'text': text,
                'duration': duration,
                'sample_rate': info.samplerate,
                'language': 'knn_Deva'
            })
        except Exception as e:
            skipped += 1
            continue
    
    print(f"\n✓ Processed {len(samples):,} samples")
    print(f"✗ Skipped {skipped:,} samples")
    
    # Shuffle and split
    random.shuffle(samples)
    
    n_train = int(len(samples) * 0.8)
    n_val = int(len(samples) * 0.1)
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train+n_val]
    test_samples = samples[n_train+n_val:]
    
    # Save manifests
    print("\nSaving manifests...")
    
    manifest_dir = output_path / 'manifests'
    manifest_dir.mkdir(exist_ok=True)
    
    for split_name, split_samples in [
        ('train', train_samples),
        ('val', val_samples),
        ('test', test_samples)
    ]:
        manifest_path = manifest_dir / f'{split_name}.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for sample in split_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        total_hours = sum(s['duration'] for s in split_samples) / 3600
        print(f"  {split_name:5s}: {len(split_samples):6,} samples ({total_hours:5.1f}h)")
    
    # Statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    all_durations = [s['duration'] for s in samples]
    all_text_lengths = [len(s['text']) for s in samples]
    
    print(f"\nAudio:")
    print(f"  Total samples: {len(samples):,}")
    print(f"  Total duration: {sum(all_durations)/3600:.1f} hours")
    print(f"  Avg duration: {sum(all_durations)/len(all_durations):.1f}s")
    print(f"  Min duration: {min(all_durations):.1f}s")
    print(f"  Max duration: {max(all_durations):.1f}s")
    
    print(f"\nText:")
    print(f"  Avg length: {sum(all_text_lengths)/len(all_text_lengths):.0f} chars")
    print(f"  Min length: {min(all_text_lengths)} chars")
    print(f"  Max length: {max(all_text_lengths)} chars")
    
    print(f"\n✓ Manifests saved to: {manifest_dir}")
    
    return manifest_dir

def combine_with_existing(raw_corpus_dir='data/konkani-raw-corpus/manifests',
                         existing_dir='data/konkani-asr-v0/splits/manifests',
                         output_dir='data/konkani-combined/manifests'):
    """Combine raw corpus with existing data"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("COMBINING DATASETS")
    print("="*70)
    
    for split in ['train', 'val', 'test']:
        combined = []
        
        # Load raw corpus
        raw_path = Path(raw_corpus_dir) / f'{split}.json'
        if raw_path.exists():
            with open(raw_path, 'r', encoding='utf-8') as f:
                raw_samples = [json.loads(line) for line in f]
            combined.extend(raw_samples)
            print(f"\n{split}: {len(raw_samples):,} from raw corpus")
        
        # Load existing
        existing_path = Path(existing_dir) / f'{split}.json'
        if existing_path.exists():
            with open(existing_path, 'r', encoding='utf-8') as f:
                existing_samples = [json.loads(line) for line in f]
            combined.extend(existing_samples)
            print(f"{split}: {len(existing_samples):,} from existing data")
        
        # Shuffle
        random.shuffle(combined)
        
        # Save
        output_file = output_path / f'{split}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in combined:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        total_hours = sum(s['duration'] for s in combined) / 3600
        print(f"{split}: {len(combined):,} total ({total_hours:.1f}h)")
    
    print(f"\n✓ Combined manifests saved to: {output_path}")

if __name__ == '__main__':
    # Step 1: Create manifests from raw corpus
    manifest_dir = create_manifest_from_corpus()
    
    # Step 2: Combine with existing data
    combine_with_existing()
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Update training config to use combined data:")
    print("   train_manifest: data/konkani-combined/manifests/train.json")
    print("   val_manifest: data/konkani-combined/manifests/val.json")
    print("\n2. Retrain from scratch with fixed config:")
    print("   - CTC weight: 0.8")
    print("   - Learning rate: 3e-4")
    print("   - Epochs: 50-100")
    print("\n3. Expected training time: 15-25 hours")
    print("\n4. Expected results:")
    print("   - Model should work by epoch 20")
    print("   - CER < 30% by epoch 50")
    print("   - Much better than current 98% blanks!")
