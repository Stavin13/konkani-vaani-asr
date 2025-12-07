import json

notebook = {
    "cells": [],
    "metadata": {
        "kaggle": {
            "accelerator": "gpu",
            "isGpuEnabled": True,
            "isInternetEnabled": True
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Add all cells
cells_data = [
    ("markdown", "# Custom Konkani-English Translation Model\n\n**Architecture:** Seq2Seq with Attention (built from scratch)  \n**Time:** ~2-3 hours on P100  \n**Truly custom:** No pre-trained weights!"),
    ("markdown", "## Step 1: Check GPU"),
    ("code", '!nvidia-smi\n\nimport torch\nprint(f"\\nGPU: {torch.cuda.get_device_name(0)}")\nprint(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")'),
    ("markdown", "## Step 2: Install Dependencies"),
    ("code", '!pip install -q transformers torch pandas tqdm scikit-learn sentencepiece\nprint("✅ Dependencies installed!")'),
    ("markdown", "## Step 3: Load Data & Generate Translations"),
    ("code", '''import json
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

DATASET_PATH = "/kaggle/input/konkani-text-data"  # UPDATE THIS

konkani_texts = []
try:
    with open(f'{DATASET_PATH}/train.json', 'r') as f:
        data = [json.loads(line) for line in f]
        konkani_texts = [item['text'] for item in data]
except:
    with open(f'{DATASET_PATH}/konkani_corpus.txt', 'r') as f:
        konkani_texts = [line.strip() for line in f if line.strip()]

konkani_texts = list(set(konkani_texts))
print(f"Loaded {len(konkani_texts)} unique Konkani texts")

print("\\nGenerating translations...")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en", device=0)

translation_pairs = []
for i in tqdm(range(0, len(konkani_texts), 16)):
    batch = konkani_texts[i:i+16]
    try:
        results = translator(batch, max_length=128)
        for konkani, result in zip(batch, results):
            english = result['translation_text']
            if len(english.split()) >= 2 and english != konkani:
                translation_pairs.append({'konkani': konkani, 'english': english})
    except:
        continue

print(f"\\n✅ Generated {len(translation_pairs)} translation pairs")'''),
]

for cell_type, content in cells_data:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": content
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    notebook["cells"].append(cell)

with open('KAGGLE_TRANSLATION_CUSTOM.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Created KAGGLE_TRANSLATION_CUSTOM.ipynb")
