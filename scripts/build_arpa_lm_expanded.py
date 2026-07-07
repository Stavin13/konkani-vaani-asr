import os
import math
import collections
from tqdm import tqdm
from pathlib import Path

def build_arpa(corpus_path, output_arpa, n=4):
    """
    Build an n-gram LM and save it in ARPA format using pure Python.
    Includes simple Add-k smoothing to avoid log(0).
    """
    print(f"Reading corpus: {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    ngram_counts = [collections.Counter() for _ in range(n)]
    
    print(f"Counting {n}-grams...")
    for line in tqdm(lines):
        tokens = ['<s>'] + line.strip().split() + ['</s>']
        for i in range(len(tokens)):
            for j in range(n):
                if i + j < len(tokens):
                    ngram = tuple(tokens[i : i + j + 1])
                    ngram_counts[j][ngram] += 1

    # Vocabulary size for smoothing
    vocab_size = len(ngram_counts[0])
    k = 0.01 # Smoothing constant
    
    print(f"Calculating probabilities and writing ARPA to {output_arpa}...")
    with open(output_arpa, 'w', encoding='utf-8') as f:
        f.write("\n\\data\\\n")
        for i in range(n):
            f.write(f"ngram {i+1}={len(ngram_counts[i])}\n")
        
        for i in range(n):
            f.write(f"\n\\{i+1}-grams:\n")
            
            # Pre-calculate denominator for current order
            if i > 0:
                context_counts = collections.Counter()
                for ngram, count in ngram_counts[i].items():
                    context_counts[ngram[:-1]] += count
            else:
                total_unigrams = sum(ngram_counts[0].values())
            
            for ngram, count in sorted(ngram_counts[i].items()):
                if i == 0: # Unigrams
                    # P(w) = (count + k) / (total + k*vocab)
                    prob = math.log10((count + k) / (total_unigrams + k * vocab_size))
                else: # i-grams
                    # P(w|context) = (count + k) / (context_count + k*vocab)
                    context_count = context_counts[ngram[:-1]]
                    prob = math.log10((count + k) / (context_count + k * vocab_size))
                
                ngram_str = " ".join(ngram)
                # ARPA format: [prob] [ngram] [backoff]
                f.write(f"{prob:.6f}\t{ngram_str}\t0\n")
        
        f.write("\n\\end\\\n")

if __name__ == "__main__":
    BASE = Path("/Volumes/data&proj/konkani")
    corpus_file = BASE / "data/konkani_expanded_corpus.txt"
    output_file = BASE / "models/language_models/konkani_4gram_news.arpa"
    
    if corpus_file.exists():
        build_arpa(corpus_file, output_file, n=4)
    else:
        print(f"Error: {corpus_file} not found.")
