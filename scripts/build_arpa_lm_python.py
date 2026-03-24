import os
import math
import collections
from tqdm import tqdm

def build_arpa(corpus_path, output_arpa, n=3):
    """
    Build a simple n-gram LM and save it in ARPA format using pure Python.
    This replaces the need for KenLM binaries on Windows.
    """
    print(f"Reading corpus: {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Pre-process for character-level if it's not already words
    # But for Konkani ASR, we usually use words or BPE.
    # We will assume word-level LM for the corpus file.
    
    ngram_counts = [collections.Counter() for _ in range(n)]
    
    print(f"Counting {n}-grams...")
    for line in tqdm(lines):
        # We need a space-separated corpus. If it's BPE, we should use BPE pieces.
        # For now we assume white-space separated words.
        tokens = ['<s>'] + line.strip().split() + ['</s>']
        
        for i in range(len(tokens)):
            for j in range(n):
                if i + j < len(tokens):
                    ngram = tuple(tokens[i : i + j + 1])
                    ngram_counts[j][ngram] += 1

    print("Calculating probabilities with simple Absolute Discounting...")
    # This is a basic smoothening. We could use Kneser-Ney but this is simpler for a script.
    # For now, let's just do Maximum Likelihood Estimation (MLE) with a small floor.
    
    with open(output_arpa, 'w', encoding='utf-8') as f:
        f.write("\n\\data\\\n")
        for i in range(n):
            f.write(f"ngram {i+1}={len(ngram_counts[i])}\n")
        
        for i in range(n):
            f.write(f"\n\\{i+1}-grams:\n")
            
            # Sub-counts for normalization (counts of the (n-1)-gram prefix)
            context_counts = collections.Counter()
            if i > 0:
                for ngram, count in ngram_counts[i].items():
                    context_counts[ngram[:-1]] += count
            else:
                total_unigrams = sum(ngram_counts[0].values())
            
            for ngram, count in sorted(ngram_counts[i].items()):
                if i == 0: # Unigrams
                    prob = math.log10(count / total_unigrams)
                else: # Bigrams and above
                    prob = math.log10(count / context_counts[ngram[:-1]])
                
                # ARPA format: [prob] [ngram] [backoff]
                ngram_str = " ".join(ngram)
                # We omit backoff (set to 0) for simplicity in this basic version
                f.write(f"{prob:.6f}\t{ngram_str}\t0\n")
        
        f.write("\n\\end\\\n")

    print(f"ARPA model saved to: {output_arpa}")

if __name__ == "__main__":
    corpus_file = "data/konkani_corpus_for_lm.txt"
    output_file = "models/language_models/konkani_3gram.arpa"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if os.path.exists(corpus_file):
        build_arpa(corpus_file, output_file, n=3)
    else:
        print(f"Error: {corpus_file} not found.")
