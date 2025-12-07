"""
Custom tokenizer for Konkani text (both Devanagari and Roman).
"""

import re
import pickle
from typing import List, Dict
from tqdm import tqdm


class KonkaniTokenizer:
    """Custom tokenizer for Konkani text (both Devanagari and Roman)"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.word_freq: Dict[str, int] = {}
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        print("Building vocabulary...")
        
        # Count word frequencies
        for text in tqdm(texts, desc="Counting words"):
            words = self._tokenize(text)
            for word in words:
                self.word_freq[word] = self.word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top vocab_size
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Add to vocabulary
        for word, freq in sorted_words[:self.vocab_size - 4]:  # -4 for special tokens
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Most common words: {sorted_words[:10]}")
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (can be enhanced)"""
        # Basic tokenization - split on whitespace and punctuation
        # Keep Devanagari and Roman characters
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens
    
    def encode(self, text: str, max_length: int = 128) -> List[int]:
        """Convert text to token indices"""
        words = self._tokenize(text)
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        # Truncate or pad
        if len(indices) > max_length:
            indices = indices[:max_length]
        
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Convert indices back to text"""
        words = [self.idx2word.get(idx, '<UNK>') for idx in indices]
        return ' '.join(words)
    
    def save(self, path: str):
        """Save tokenizer"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_freq': self.word_freq,
                'vocab_size': self.vocab_size
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """Load tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.word2idx = data['word2idx']
        tokenizer.idx2word = data['idx2word']
        tokenizer.word_freq = data['word_freq']
        return tokenizer
    
    def __len__(self):
        """Return vocabulary size"""
        return len(self.word2idx)
