"""
Konkani Text Tokenizer
=====================
Character-level tokenizer for Konkani Devanagari script
"""

import json
from pathlib import Path
from collections import Counter
import re


class KonkaniTokenizer:
    """Character-level tokenizer for Konkani Devanagari text"""
    
    # Special tokens
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'
    BLANK_TOKEN = '<blank>'  # For CTC
    
    def __init__(self, vocab_file=None):
        """
        Initialize tokenizer
        
        Args:
            vocab_file: Path to vocabulary JSON file (optional)
        """
        self.char2idx = {}
        self.idx2char = {}
        
        if vocab_file and Path(vocab_file).exists():
            self.load_vocab(vocab_file)
    
    def build_vocab_from_transcripts(self, transcripts, min_freq=1):
        """
        Build vocabulary from list of transcripts
        
        Args:
            transcripts: List of Konkani text strings
            min_freq: Minimum character frequency to include
        
        Returns:
            vocab_size: Size of vocabulary
        """
        # Count character frequencies
        char_counter = Counter()
        for text in transcripts:
            # Normalize text
            text = self.normalize_text(text)
            char_counter.update(text)
        
        # Filter by frequency
        chars = [char for char, freq in char_counter.items() if freq >= min_freq]
        
        # Sort for consistency
        chars = sorted(chars)
        
        # Build vocabulary (special tokens first)
        self.char2idx = {
            self.PAD_TOKEN: 0,
            self.BLANK_TOKEN: 1,  # CTC blank
            self.SOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
            self.UNK_TOKEN: 4,
        }
        
        # Add regular characters
        for char in chars:
            if char not in self.char2idx:
                self.char2idx[char] = len(self.char2idx)
        
        # Create reverse mapping
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        
        print(f"✅ Built vocabulary with {len(self.char2idx)} characters")
        print(f"   Special tokens: {list(self.char2idx.keys())[:5]}")
        print(f"   Sample chars: {list(self.char2idx.keys())[5:15]}")
        
        return len(self.char2idx)
    
    def normalize_text(self, text):
        """
        Normalize Konkani text
        
        Args:
            text: Raw text string
        
        Returns:
            normalized: Normalized text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove punctuation (optional - you may want to keep some)
        # text = re.sub(r'[^\u0900-\u097F\s]', '', text)  # Keep only Devanagari + space
        
        return text.strip()
    
    def encode(self, text, add_sos_eos=False):
        """
        Encode text to token indices
        
        Args:
            text: Konkani text string
            add_sos_eos: Whether to add SOS/EOS tokens
        
        Returns:
            indices: List of token indices
        """
        text = self.normalize_text(text)
        
        indices = []
        if add_sos_eos:
            indices.append(self.char2idx[self.SOS_TOKEN])
        
        for char in text:
            indices.append(self.char2idx.get(char, self.char2idx[self.UNK_TOKEN]))
        
        if add_sos_eos:
            indices.append(self.char2idx[self.EOS_TOKEN])
        
        return indices
    
    def decode(self, indices, remove_special=True):
        """
        Decode token indices to text
        
        Args:
            indices: List of token indices
            remove_special: Whether to remove special tokens
        
        Returns:
            text: Decoded text string
        """
        chars = []
        for idx in indices:
            char = self.idx2char.get(idx, self.UNK_TOKEN)
            
            if remove_special:
                if char in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, 
                           self.UNK_TOKEN, self.BLANK_TOKEN]:
                    continue
            
            chars.append(char)
        
        return ''.join(chars)
    
    def decode_ctc(self, indices):
        """
        Decode CTC output (remove blanks and duplicates)
        
        Args:
            indices: List of token indices from CTC
        
        Returns:
            text: Decoded text
        """
        # Remove consecutive duplicates
        collapsed = []
        prev_idx = None
        for idx in indices:
            if idx != prev_idx:
                collapsed.append(idx)
                prev_idx = idx
        
        # Remove blanks
        blank_idx = self.char2idx[self.BLANK_TOKEN]
        filtered = [idx for idx in collapsed if idx != blank_idx]
        
        # Decode
        return self.decode(filtered, remove_special=True)
    
    def save_vocab(self, vocab_file):
        """Save vocabulary to JSON file"""
        vocab_data = {
            'char2idx': self.char2idx,
            'idx2char': {str(k): v for k, v in self.idx2char.items()}
        }
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved vocabulary to {vocab_file}")
    
    def load_vocab(self, vocab_file):
        """Load vocabulary from JSON file"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.char2idx = vocab_data['char2idx']
        self.idx2char = {int(k): v for k, v in vocab_data['idx2char'].items()}
        
        print(f"✅ Loaded vocabulary with {len(self.char2idx)} characters")
    
    @property
    def vocab_size(self):
        """Get vocabulary size"""
        return len(self.char2idx)
    
    @property
    def pad_id(self):
        return self.char2idx[self.PAD_TOKEN]
    
    @property
    def sos_id(self):
        return self.char2idx[self.SOS_TOKEN]
    
    @property
    def eos_id(self):
        return self.char2idx[self.EOS_TOKEN]
    
    @property
    def blank_id(self):
        return self.char2idx[self.BLANK_TOKEN]


def create_vocab_from_manifests(manifest_files, output_file='vocab.json'):
    """
    Create vocabulary from training manifests
    
    Args:
        manifest_files: List of manifest JSON files
        output_file: Output vocabulary file
    
    Returns:
        tokenizer: KonkaniTokenizer instance
    """
    print(f"\n{'='*60}")
    print("CREATING KONKANI VOCABULARY")
    print(f"{'='*60}")
    
    # Collect all transcripts
    transcripts = []
    for manifest_file in manifest_files:
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                transcripts.append(data['text'])
    
    print(f"Loaded {len(transcripts)} transcripts")
    
    # Build vocabulary
    tokenizer = KonkaniTokenizer()
    vocab_size = tokenizer.build_vocab_from_transcripts(transcripts, min_freq=1)
    
    # Save vocabulary
    tokenizer.save_vocab(output_file)
    
    # Test encoding/decoding
    sample_text = transcripts[0]
    print(f"\n📝 Testing tokenizer:")
    print(f"   Original: {sample_text[:100]}...")
    encoded = tokenizer.encode(sample_text)
    print(f"   Encoded: {encoded[:20]}...")
    decoded = tokenizer.decode(encoded)
    print(f"   Decoded: {decoded[:100]}...")
    
    return tokenizer


if __name__ == "__main__":
    # Create vocabulary from training manifests
    manifest_files = [
        'data/konkani-asr-v0/splits/manifests/train.json',
        'data/konkani-asr-v0/splits/manifests/val.json',
        'data/konkani-asr-v0/splits/manifests/test.json'
    ]
    
    tokenizer = create_vocab_from_manifests(manifest_files, 'vocab.json')
