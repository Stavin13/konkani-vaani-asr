#!/usr/bin/env python3
"""Generate complete 10K training notebook with all code"""
import json

# Read the base notebook
with open('notebooks/KAGGLE_TRAIN_10K_DUAL_GPU.ipynb', 'r') as f:
    nb = json.load(f)

# Training script content (split into manageable chunks)
training_code_part1 = """import json
import math
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

# Tokenizer
class KonkaniTokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.char2idx = data['char2idx']
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        self.pad_id = self.char2idx['<pad>']
        self.blank_id = self.char2idx['<blank>']
        self.sos_id = self.char2idx['<sos>']
        self.eos_id = self.char2idx['<eos>']
        self.unk_id = self.char2idx['<unk>']
    
    def encode(self, text):
        tokens = [self.sos_id]
        for char in text:
            tokens.append(self.char2idx.get(char, self.unk_id))
        tokens.append(self.eos_id)
        return tokens
    
    def decode(self, tokens):
        chars = []
        for token in tokens:
            if token in [self.pad_id, self.blank_id, self.sos_id, self.eos_id]:
                continue
            chars.append(self.idx2char.get(token, '<unk>'))
        return ''.join(chars)

print('✓ Tokenizer defined')"""

# Add the training code cells
nb["cells"].extend([
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 5. Data Processing"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [training_code_part1]
    }
])

# Save updated notebook
with open('notebooks/KAGGLE_TRAIN_10K_DUAL_GPU.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✓ Updated notebook - now has {len(nb['cells'])} cells")
