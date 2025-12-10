# Kaggle Troubleshooting Guide

## ❌ Error: FileNotFoundError: '/kaggle/input/scripts1/models/konkanivani_asr.py'

### 🔍 **Root Cause:**
The error means Kaggle can't find your model file in the expected location.

### ✅ **Solutions:**

#### **Solution 1: Check Dataset Structure**
1. **In Kaggle notebook**, add this cell to check your dataset structure:
```python
import os
print("Dataset contents:")
for root, dirs, files in os.walk('/kaggle/input/scripts1'):
    level = root.replace('/kaggle/input/scripts1', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f'{subindent}{file}')
```

#### **Solution 2: Use the Clean Notebook**
- **Upload**: `KonkaniVani_Fixed_Vocab_Training_Clean.ipynb` (not the old one)
- **This notebook**: Uses proper imports, not exec() statements

#### **Solution 3: Alternative Model Loading**
If imports still fail, add this cell before model loading:
```python
# Alternative: Load model code directly
model_code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KonkaniVaniASR(nn.Module):
    def __init__(self, vocab_size, input_dim=80, d_model=256, encoder_layers=12, 
                 decoder_layers=6, num_heads=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=conv_kernel_size//2)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=conv_kernel_size//2)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, time, features)
        x = self.input_projection(x)
        
        # Convolutional layers
        x = x.transpose(1, 2)  # (batch, features, time)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (batch, time, features)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Output projection
        x = self.dropout(x)
        x = self.output_projection(x)
        
        return x
'''

exec(model_code)
print("✅ Model loaded directly")
```

### 🎯 **Quick Fix Steps:**
1. **Use the clean notebook**: `KonkaniVani_Fixed_Vocab_Training_Clean.ipynb`
2. **Check your dataset name**: Make sure it's exactly `scripts1`
3. **Verify dataset contents**: Run the structure check above
4. **If still failing**: Use the alternative model loading code

### 📋 **Expected Dataset Structure:**
```
/kaggle/input/scripts1/
├── models/
│   └── konkanivani_asr.py
├── data/
│   └── audio_processing/
│       ├── audio_processor.py
│       ├── dataset.py
│       └── text_tokenizer.py
├── vocab.json
└── initial_model_vocab200.pt
```

The clean notebook should work perfectly! 🚀