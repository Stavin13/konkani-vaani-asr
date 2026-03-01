import os
import json
import zipfile
from pathlib import Path
from tqdm import tqdm

def zip_specific_chunk(chunk_idx, manifest_dir, corpus_root, output_dir):
    """
    Zips all files listed in a specific manifest chunk into a single .zip file.
    """
    manifest_path = os.path.join(manifest_dir, f"kaggle_manifest_chunk_{chunk_idx}.json")
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest {manifest_path} not found.")
        return

    output_zip = os.path.join(output_dir, f"konkani_chunk_{chunk_idx}.zip")
    corpus_root = Path(corpus_root)
    
    # 1. Collect file list
    files_to_zip = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            files_to_zip.append(sample['audio_filepath'])
    
    print(f"Zipping Chunk {chunk_idx} ({len(files_to_zip)} files) into {output_zip}...")
    
    # 2. Perform Zipping
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_STORED) as zf:
        # ZIP_STORED is faster for audio since it's already compressed (WAV/MP3)
        for rel_path in tqdm(files_to_zip, desc="Packing"):
            abs_path = corpus_root / rel_path
            if abs_path.exists():
                # We store the file with the SAME relative path inside the zip
                zf.write(abs_path, arcname=rel_path)
            else:
                print(f"Warning: File missing: {abs_path}")

    print(f"Successfully created: {output_zip}")

if __name__ == "__main__":
    MANIFEST_DIR = r"E:\konkani\data\kaggle_manifests"
    CORPUS_ROOT = r"E:\konkani\KonkaniRawSpeechCorpus"
    OUTPUT_DIR = r"E:\konkani"
    
    # You can change this to 1, 2, 3, 4, 5 as you go
    CHUNK_TO_ZIP = 1
    
    zip_specific_chunk(CHUNK_TO_ZIP, MANIFEST_DIR, CORPUS_ROOT, OUTPUT_DIR)
