import os
import glob
from pathlib import Path
from tqdm import tqdm

def extract_all_transcripts(corpus_root, output_file):
    """
    Crawls the entire corpus and combines every .txt file into a single text file
    for Language Model (KenLM) training.
    """
    corpus_root = Path(corpus_root)
    txt_files = list(corpus_root.rglob("*.txt"))
    
    print(f"Found {len(txt_files)} transcript files in {corpus_root}.")
    
    unique_sentences = set()
    total_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for txt_path in tqdm(txt_files, desc="Extracting Text"):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        # We save every unique sentence
                        # This helps the LM learn structure without being biased 
                        # by repetitive data in the audio recordings.
                        unique_sentences.add(content)
                        outfile.write(content + "\n")
                        total_count += 1
            except Exception as e:
                continue

    print(f"\nExtraction Complete!")
    print(f"Total Sentences Extracted: {total_count}")
    print(f"Unique Sentences: {len(unique_sentences)}")
    print(f"File Saved: {output_file}")

if __name__ == "__main__":
    CORPUS_PATH = r"E:\konkani\KonkaniRawSpeechCorpus"
    OUT_FILE = r"E:\konkani\konkani_text_corpus.txt"
    extract_all_transcripts(CORPUS_PATH, OUT_FILE)
