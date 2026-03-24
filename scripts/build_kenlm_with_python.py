#!/usr/bin/env python3
"""
Build KenLM models using Python kenlm package
This is easier than building from source
"""
import subprocess
import sys
from pathlib import Path

def install_kenlm_python():
    """Install kenlm Python package"""
    print("Installing kenlm Python package...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "https://github.com/kpu/kenlm/archive/master.zip"],
            check=True
        )
        print("✓ kenlm installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install kenlm")
        return False

def build_models():
    """Build the language models"""
    try:
        import kenlm
    except ImportError:
        print("kenlm not found, installing...")
        if not install_kenlm_python():
            sys.exit(1)
        import kenlm
    
    corpus_file = Path("data/konkani_corpus_for_lm.txt")
    output_dir = Path("models/language_models")
    
    if not corpus_file.exists():
        print(f"ERROR: Corpus file not found: {corpus_file}")
        print("Run: python scripts/extract_text_for_kenlm.py first")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count lines
    with open(corpus_file, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    
    print(f"\n{'='*60}")
    print(f"Building KenLM Models")
    print(f"{'='*60}")
    print(f"Corpus: {corpus_file}")
    print(f"Lines: {line_count:,}")
    print(f"Output: {output_dir}")
    
    # We still need the command-line tools for building
    # Let's provide instructions
    print(f"\n{'='*60}")
    print("To build the models, you need KenLM command-line tools.")
    print("Since Homebrew doesn't have it, here's the manual process:")
    print(f"{'='*60}")
    print("\n1. Install dependencies:")
    print("   brew install cmake boost eigen")
    print("\n2. Build KenLM:")
    print("   git clone https://github.com/kpu/kenlm.git /tmp/kenlm")
    print("   cd /tmp/kenlm")
    print("   mkdir build && cd build")
    print("   cmake .. -DKENLM_MAX_ORDER=6")
    print("   make -j4")
    print("   sudo cp bin/lmplz bin/build_binary /usr/local/bin/")
    print("\n3. Then run:")
    print("   python scripts/build_kenlm_models.py")
    print(f"\n{'='*60}")
    print("\nAlternatively, you can build models on a Linux machine or use")
    print("Google Colab which has better support for building from source.")

if __name__ == "__main__":
    build_models()
