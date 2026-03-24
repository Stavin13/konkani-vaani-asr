# KenLM Setup Guide for Konkani ASR

## Summary

We've extracted **250,225 sentences** (~1.85M words) of Devanagari Konkani text from your corpus.
The text is ready at: `data/konkani_corpus_for_lm.txt`

Now we need to build 2 language models:
- 3-gram model (smaller, faster)
- 4-gram model (more accurate, recommended)

## Problem on macOS

KenLM doesn't work well with the latest Boost 1.90 on macOS. You have 3 options:

## Option 1: Use Linux/Kaggle (Recommended)

Build the models on a Linux machine or Kaggle notebook where KenLM installs easily:

```bash
# On Linux (Ubuntu/Debian)
sudo apt-get install build-essential cmake libboost-all-dev libeigen3-dev

# Clone and build KenLM
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir build && cd build
cmake ..
make -j4
sudo make install

# Build your models
cd /path/to/your/project
./scripts/build_kenlm_models.sh
```

## Option 2: Use Docker on macOS

```bash
# Pull Ubuntu image
docker run -it -v $(pwd):/workspace ubuntu:22.04 bash

# Inside container
apt-get update
apt-get install -y build-essential cmake libboost-all-dev libeigen3-dev git

# Build KenLM
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir build && cd build
cmake ..
make -j4
cp bin/lmplz bin/build_binary /usr/local/bin/

# Build models
cd /workspace
bash scripts/build_kenlm_models.sh
```

## Option 3: Manual Commands (if you get KenLM working)

Once KenLM is installed, run these commands:

```bash
# 3-gram model
lmplz -o 3 --prune 0 2 < data/konkani_corpus_for_lm.txt > models/language_models/konkani_3gram.arpa
build_binary models/language_models/konkani_3gram.arpa models/language_models/konkani_3gram.binary

# 4-gram model  
lmplz -o 4 --prune 0 3 3 < data/konkani_corpus_for_lm.txt > models/language_models/konkani_4gram.arpa
build_binary models/language_models/konkani_4gram.arpa models/language_models/konkani_4gram.binary
```

## What You'll Get

After building, you'll have:
- `models/language_models/konkani_3gram.binary` (~100-200MB)
- `models/language_models/konkani_4gram.binary` (~150-300MB)

## Next Steps

1. Build the models using one of the options above
2. Integrate with your ASR beam search decoder
3. Tune parameters:
   - Beam width: 10-20
   - LM weight (alpha): 0.5-1.5
4. Test on validation set
5. Expected CER improvement: 20-40% relative

## Files Created

- `data/konkani_corpus_for_lm.txt` - Your extracted corpus (250k sentences)
- `scripts/extract_text_for_kenlm.py` - Extraction script (already run)
- `scripts/build_kenlm_models.sh` - Build script (ready to use on Linux)
- `scripts/build_kenlm_models.py` - Python version (ready to use on Linux)

## Recommendation

Use **Kaggle** to build the models since you're already using it for training. Upload the corpus file and run the build script there. Then download the `.binary` files back to your Mac.
