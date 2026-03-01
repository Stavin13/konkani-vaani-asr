import os
import json
from pathlib import Path
from tqdm import tqdm

def create_kaggle_manifests(corpus_root, output_dir, chunk_size_gb=19.0):
    """
    Scans the corpus and creates multiple manifest files, each pointing to ~20GB of data.
    """
    corpus_root = Path(corpus_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect all WAV files and their potential transcript TXT files
    all_samples = []
    print(f"Scanning {corpus_root}...")
    
    # We look for .wav files
    wav_files = list(corpus_root.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files. Verifying transcripts...")
    
    for wav_path in tqdm(wav_files):
        # Look for matching .txt file (usually same name)
        txt_path = wav_path.with_suffix('.txt')
        
        if txt_path.exists():
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                if not text: continue # Skip empty transcripts
                
                file_size = wav_path.stat().st_size
                
                # We store a relative path so it works on Kaggle regardless of mount point
                rel_path = os.path.relpath(wav_path, corpus_root)
                
                all_samples.append({
                    "audio_filepath": rel_path,
                    "text": text,
                    "size_bytes": file_size,
                    "duration": file_size / (16000 * 2) # Rough estimate for 16bit 16kHz
                })
            except:
                continue

    # 2. Shuffle to ensure diverse data in each chunk (optional but recommended)
    import random
    random.shuffle(all_samples)

    # 3. Split into 20GB Chunks
    bytes_per_chunk = chunk_size_gb * 1024 * 1024 * 1024
    
    current_chunk = []
    current_size = 0
    chunk_idx = 1
    
    for sample in all_samples:
        current_chunk.append(sample)
        current_size += sample['size_bytes']
        
        if current_size >= bytes_per_chunk:
            # Save chunk
            chunk_file = output_dir / f"kaggle_manifest_chunk_{chunk_idx}.json"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                for s in current_chunk:
                    json.dump(s, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"Saved Chunk {chunk_idx}: {len(current_chunk)} samples (~{current_size/1e9:.2f} GB)")
            
            # Reset
            current_chunk = []
            current_size = 0
            chunk_idx += 1
            
    # Save final partial chunk
    if current_chunk:
        chunk_file = output_dir / f"kaggle_manifest_chunk_{chunk_idx}.json"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            for s in current_chunk:
                json.dump(s, f, ensure_ascii=False)
                f.write('\n')
        print(f"Saved Final Chunk {chunk_idx}: {len(current_chunk)} samples (~{current_size/1e9:.2f} GB)")

if __name__ == "__main__":
    CORPUS_PATH = r"E:\konkani\KonkaniRawSpeechCorpus"
    OUT_PATH = r"E:\konkani\data\kaggle_manifests"
    create_kaggle_manifests(CORPUS_PATH, OUT_PATH)
