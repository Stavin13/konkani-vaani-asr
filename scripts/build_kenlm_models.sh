#!/bin/bash
# Build KenLM language models for Konkani ASR

set -e

echo "=========================================="
echo "Building KenLM Language Models"
echo "=========================================="

# Check if KenLM is installed
if ! command -v lmplz &> /dev/null; then
    echo "ERROR: KenLM not found!"
    echo ""
    echo "Install KenLM:"
    echo "  macOS: brew install kenlm"
    echo "  Linux: sudo apt-get install kenlm"
    echo "  Or build from source: https://github.com/kpu/kenlm"
    exit 1
fi

# Input and output paths
CORPUS_FILE="data/konkani_corpus_for_lm.txt"
OUTPUT_DIR="models/language_models"

# Check if corpus exists
if [ ! -f "$CORPUS_FILE" ]; then
    echo "ERROR: Corpus file not found: $CORPUS_FILE"
    echo "Run: python scripts/extract_text_for_kenlm.py first"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Corpus file: $CORPUS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Count lines in corpus
LINE_COUNT=$(wc -l < "$CORPUS_FILE")
echo "Corpus size: $LINE_COUNT sentences"
echo ""

# Build 3-gram model
echo "=========================================="
echo "Building 3-gram model..."
echo "=========================================="
echo "Step 1/2: Training ARPA model..."
lmplz -o 3 --prune 0 2 < "$CORPUS_FILE" > "$OUTPUT_DIR/konkani_3gram.arpa"

echo "Step 2/2: Converting to binary format..."
build_binary "$OUTPUT_DIR/konkani_3gram.arpa" "$OUTPUT_DIR/konkani_3gram.binary"

echo "✓ 3-gram model complete!"
echo "  ARPA: $OUTPUT_DIR/konkani_3gram.arpa"
echo "  Binary: $OUTPUT_DIR/konkani_3gram.binary"
echo ""

# Build 4-gram model
echo "=========================================="
echo "Building 4-gram model..."
echo "=========================================="
echo "Step 1/2: Training ARPA model..."
lmplz -o 4 --prune 0 3 3 < "$CORPUS_FILE" > "$OUTPUT_DIR/konkani_4gram.arpa"

echo "Step 2/2: Converting to binary format..."
build_binary "$OUTPUT_DIR/konkani_4gram.arpa" "$OUTPUT_DIR/konkani_4gram.binary"

echo "✓ 4-gram model complete!"
echo "  ARPA: $OUTPUT_DIR/konkani_4gram.arpa"
echo "  Binary: $OUTPUT_DIR/konkani_4gram.binary"
echo ""

# Show file sizes
echo "=========================================="
echo "Model sizes:"
echo "=========================================="
ls -lh "$OUTPUT_DIR"/*.binary | awk '{print $9, "-", $5}'
echo ""

echo "=========================================="
echo "✓ All models built successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Test models with your ASR decoder"
echo "  2. Tune beam width (10-20) and LM weight (0.5-1.5)"
echo "  3. Compare 3-gram vs 4-gram on validation set"
