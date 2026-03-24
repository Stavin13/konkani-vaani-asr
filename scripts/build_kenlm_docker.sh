#!/bin/bash
# Build KenLM models using Docker (works on macOS)

set -e

echo "=========================================="
echo "Building KenLM Models with Docker"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found!"
    echo "Install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if corpus exists
if [ ! -f "data/konkani_corpus_for_lm.txt" ]; then
    echo "ERROR: Corpus file not found!"
    echo "Run: python scripts/extract_text_for_kenlm.py first"
    exit 1
fi

# Create output directory
mkdir -p models/language_models

echo ""
echo "Starting Docker container..."
echo "This will:"
echo "  1. Pull Ubuntu image (if needed)"
echo "  2. Install KenLM"
echo "  3. Build 3-gram and 4-gram models"
echo "  4. Save models to models/language_models/"
echo ""

# Run Docker container with volume mount
docker run --rm \
    -v "$(pwd)/data:/data" \
    -v "$(pwd)/models:/models" \
    ubuntu:22.04 \
    bash -c '
        set -e
        echo "Installing dependencies..."
        apt-get update -qq
        apt-get install -y -qq build-essential cmake libboost-all-dev libeigen3-dev git wget > /dev/null 2>&1
        
        echo "Building KenLM..."
        cd /tmp
        git clone https://github.com/kpu/kenlm.git > /dev/null 2>&1
        cd kenlm
        mkdir build && cd build
        cmake .. > /dev/null 2>&1
        make -j4 > /dev/null 2>&1
        
        echo ""
        echo "=========================================="
        echo "Building 3-gram model..."
        echo "=========================================="
        /tmp/kenlm/build/bin/lmplz -o 3 --prune 0 2 --discount_fallback < /data/konkani_corpus_for_lm.txt > /models/language_models/konkani_3gram.arpa
        /tmp/kenlm/build/bin/build_binary /models/language_models/konkani_3gram.arpa /models/language_models/konkani_3gram.binary
        rm /models/language_models/konkani_3gram.arpa
        
        echo ""
        echo "=========================================="
        echo "Building 4-gram model..."
        echo "=========================================="
        /tmp/kenlm/build/bin/lmplz -o 4 --prune 0 3 3 --discount_fallback < /data/konkani_corpus_for_lm.txt > /models/language_models/konkani_4gram.arpa
        /tmp/kenlm/build/bin/build_binary /models/language_models/konkani_4gram.arpa /models/language_models/konkani_4gram.binary
        rm /models/language_models/konkani_4gram.arpa
        
        echo ""
        echo "=========================================="
        echo "Models built successfully!"
        echo "=========================================="
        ls -lh /models/language_models/*.binary
    '

echo ""
echo "=========================================="
echo "✓ Done!"
echo "=========================================="
echo ""
echo "Models saved:"
ls -lh models/language_models/*.binary
echo ""
echo "Next steps:"
echo "  1. Integrate with your ASR beam search decoder"
echo "  2. Tune beam width (10-20) and LM weight (0.5-1.5)"
echo "  3. Test on validation set"
