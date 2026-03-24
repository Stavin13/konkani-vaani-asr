#!/usr/bin/env python3
"""
Build KenLM language models for Konkani ASR using Python
"""
import os
import subprocess
import sys
from pathlib import Path

def check_kenlm():
    """Check if KenLM is installed"""
    try:
        subprocess.run(['lmplz', '--help'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_kenlm():
    """Provide installation instructions"""
    print("\n" + "="*60)
    print("KenLM Installation Required")
    print("="*60)
    print("\nKenLM is not installed. Please install it:")
    print("\nmacOS:")
    print("  brew install kenlm")
    print("\nLinux (Ubuntu/Debian):")
    print("  sudo apt-get install kenlm")
    print("\nOr build from source:")
    print("  git clone https://github.com/kpu/kenlm.git")
    print("  cd kenlm")
    print("  mkdir -p build")
    print("  cd build")
    print("  cmake ..")
    print("  make -j4")
    print("  sudo make install")
    print("\nAfter installation, run this script again.")
    print("="*60)
    sys.exit(1)

def build_model(corpus_file, output_dir, order, prune_args):
    """Build a KenLM model"""
    arpa_file = output_dir / f"konkani_{order}gram.arpa"
    binary_file = output_dir / f"konkani_{order}gram.binary"
    
    print(f"\n{'='*60}")
    print(f"Building {order}-gram model...")
    print(f"{'='*60}")
    
    # Build ARPA model
    print(f"Step 1/2: Training ARPA model...")
    with open(corpus_file, 'r', encoding='utf-8') as f_in:
        with open(arpa_file, 'w', encoding='utf-8') as f_out:
            cmd = ['lmplz', '-o', str(order)] + prune_args
            result = subprocess.run(
                cmd,
                stdin=f_in,
                stdout=f_out,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Print statistics from stderr
            for line in result.stderr.split('\n'):
                if 'unigram' in line.lower() or 'ngram' in line.lower():
                    print(f"  {line}")
    
    print(f"✓ ARPA model saved: {arpa_file}")
    
    # Convert to binary
    print(f"Step 2/2: Converting to binary format...")
    subprocess.run(
        ['build_binary', str(arpa_file), str(binary_file)],
        check=True,
        capture_output=True
    )
    
    print(f"✓ Binary model saved: {binary_file}")
    
    # Get file sizes
    arpa_size = arpa_file.stat().st_size / (1024 * 1024)  # MB
    binary_size = binary_file.stat().st_size / (1024 * 1024)  # MB
    
    print(f"\nModel sizes:")
    print(f"  ARPA:   {arpa_size:.2f} MB")
    print(f"  Binary: {binary_size:.2f} MB")
    
    return binary_file

def main():
    print("="*60)
    print("Building KenLM Language Models for Konkani")
    print("="*60)
    
    # Check KenLM installation
    if not check_kenlm():
        install_kenlm()
    
    # Setup paths
    corpus_file = Path("data/konkani_corpus_for_lm.txt")
    output_dir = Path("models/language_models")
    
    # Check corpus exists
    if not corpus_file.exists():
        print(f"\nERROR: Corpus file not found: {corpus_file}")
        print("Run: python scripts/extract_text_for_kenlm.py first")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count lines
    with open(corpus_file, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    
    print(f"\nCorpus file: {corpus_file}")
    print(f"Corpus size: {line_count:,} sentences")
    print(f"Output directory: {output_dir}")
    
    # Build 3-gram model
    # Prune: keep all unigrams (0), prune bigrams with count < 2, prune trigrams < 2
    model_3gram = build_model(
        corpus_file, 
        output_dir, 
        order=3,
        prune_args=['--prune', '0', '2']
    )
    
    # Build 4-gram model
    # Prune: keep all unigrams (0), prune bigrams < 3, trigrams < 3, 4-grams < 3
    model_4gram = build_model(
        corpus_file,
        output_dir,
        order=4,
        prune_args=['--prune', '0', '3', '3']
    )
    
    print(f"\n{'='*60}")
    print("✓ All models built successfully!")
    print(f"{'='*60}")
    print(f"\nModels saved:")
    print(f"  3-gram: {model_3gram}")
    print(f"  4-gram: {model_4gram}")
    print(f"\nNext steps:")
    print(f"  1. Integrate with your ASR beam search decoder")
    print(f"  2. Tune beam width (10-20) and LM weight (0.5-1.5)")
    print(f"  3. Compare 3-gram vs 4-gram on validation set")
    print(f"  4. Expected CER improvement: 20-40% relative")

if __name__ == "__main__":
    main()
