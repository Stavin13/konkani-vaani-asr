import os
import glob
from pathlib import Path

def extract_all_transcripts(corpus_root, output_file):
    corpus_root = Path(corpus_root)
    # The corpus might have transcripts in JSON, CSV or TXT. Let's check common locations.
    txt_files = list(corpus_root.rglob("*.txt"))
    unique_sentences = set()
    total_count = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for txt_path in txt_files:
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content and content not in unique_sentences:
                        unique_sentences.add(content)
                        outfile.write(content + "\n")
                        total_count += 1
            except Exception as e:
                continue
    print(f"Extracted {total_count} unique sentences from {len(txt_files)} files.")

if __name__ == "__main__":
    extract_all_transcripts("/Volumes/data&proj/konkani/KonkaniRawSpeechCorpus", "/Volumes/data&proj/konkani/corpus_text.txt")
