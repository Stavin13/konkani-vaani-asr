#!/usr/bin/env python3
"""
Check the actual vocabulary file being used
"""
import json
from pathlib import Path

def check_vocab_file():
    """Check the vocabulary file"""
    
    vocab_path = Path('data/vocab.json')
    
    if not vocab_path.exists():
        print("❌ No vocab.json found")
        return
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    char2idx = vocab_data.get('char2idx', {})
    
    print(f"Vocabulary file: {vocab_path}")
    print(f"Vocabulary size: {len(char2idx)}")
    
    # Show first 20 entries
    print(f"\nFirst 20 entries:")
    for i, (char, idx) in enumerate(char2idx.items()):
        if i >= 20:
            break
        print(f"  {idx:3d}: '{char}'")
    
    # Check for Devanagari characters
    devanagari_chars = []
    latin_chars = []
    special_tokens = []
    
    for char, idx in char2idx.items():
        if char.startswith('<') and char.endswith('>'):
            special_tokens.append(char)
        elif ord(char[0]) >= 0x0900 and ord(char[0]) <= 0x097F:
            devanagari_chars.append(char)
        elif char.isalpha():
            latin_chars.append(char)
    
    print(f"\nCharacter analysis:")
    print(f"  Special tokens: {len(special_tokens)} - {special_tokens}")
    print(f"  Devanagari chars: {len(devanagari_chars)}")
    print(f"  Latin chars: {len(latin_chars)}")
    print(f"  Other chars: {len(char2idx) - len(special_tokens) - len(devanagari_chars) - len(latin_chars)}")
    
    if len(devanagari_chars) > 0:
        print(f"  Sample Devanagari: {devanagari_chars[:10]}")
    
    # This explains why the model has vocab_size=81!
    if len(char2idx) == 81:
        print(f"\n🎯 FOUND THE ISSUE!")
        print(f"   This vocab.json has exactly 81 characters")
        print(f"   The training script used this file, not the YAML config!")
    
    return len(char2idx)

if __name__ == '__main__':
    check_vocab_file()