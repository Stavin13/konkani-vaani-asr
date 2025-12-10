#!/usr/bin/env python3
"""
Create Kaggle notebook for 10K dataset training with dual GPU
"""
import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Cell 1: Title
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Konkani ASR Training - 10K Dataset with Dual GPU\n",
        "\n",
        "Train Konkani ASR model on 10K samples using dual GPU for faster training.\n",
        "\n",
        "**Expected Results:**\n",
        "- Training time: ~9 hours (vs 18 hours on single GPU)\n",
        "- Should produce actual transcriptions (not blanks)\n",
        "- Target validation loss: < 2.5\n",
        "\n",
        "**Hardware:** Kaggle P100 x2 GPUs"
    ]
})

# Cell 2: GPU Check
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 1. Setup & GPU Check"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Check GPU availability\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "print(f\"PyTorch version: {torch.__version__}\")\n",
        "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
        "print(f\"Number of GPUs: {torch.cuda.device_count()}\")\n",
        "for i in range(torch.cuda.device_count()):\n",
        "    print(f\"  GPU {i}: {torch.cuda.get_device_name(i)}\")\n",
        "    print(f\"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB\")"
    ]
})

# Cell 3: Install Dependencies
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 2. Install Dependencies"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["!pip install -q librosa soundfile torchaudio tensorboard"]
})

# Cell 4: Mount Dataset
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Mount Dataset\n",
        "\n",
        "Upload your `konkani_10k_dataset.zip` as a Kaggle dataset and add it to this notebook."
    ]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "import zipfile\n",
        "from pathlib import Path\n",
        "\n",
        "# Kaggle dataset path (update with your dataset name)\n",
        "DATASET_PATH = '/kaggle/input/konkani-10k-dataset'\n",
        "\n",
        "# Check if dataset exists\n",
        "if os.path.exists(DATASET_PATH):\n",
        "    print(f\"✓ Dataset found at: {DATASET_PATH}\")\n",
        "    print(\"\\nContents:\")\n",
        "    !ls -lh {DATASET_PATH}\n",
        "else:\n",
        "    print(\"❌ Dataset not found!\")\n",
        "    print(\"Please add the konkani-10k-dataset to this notebook.\")"
    ]
})

# Cell 5: Extract Dataset
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 4. Extract Dataset"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Extract if needed\n",
        "WORK_DIR = Path('/kaggle/working')\n",
        "DATA_DIR = WORK_DIR / 'data'\n",
        "\n",
        "if not DATA_DIR.exists():\n",
        "    print(\"Extracting dataset...\")\n",
        "    zip_file = Path(DATASET_PATH) / 'konkani_10k_dataset.zip'\n",
        "    \n",
        "    if zip_file.exists():\n",
        "        with zipfile.ZipFile(zip_file, 'r') as zip_ref:\n",
        "            zip_ref.extractall(WORK_DIR)\n",
        "        print(\"✓ Dataset extracted\")\n",
        "    else:\n",
        "        print(\"❌ ZIP file not found\")\n",
        "else:\n",
        "    print(\"✓ Dataset already extracted\")\n",
        "\n",
        "# Verify structure\n",
        "print(\"\\nDataset structure:\")\n",
        "!ls -lh {DATA_DIR}"
    ]
})

# Save notebook
output_path = "notebooks/KAGGLE_TRAIN_10K_DUAL_GPU.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✓ Created notebook: {output_path}")
print(f"  Total cells: {len(notebook['cells'])}")
