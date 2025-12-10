#!/bin/bash
# Split large dataset into smaller uploadable chunks

DATASET_ZIP="konkani_10k_dataset.zip"
CHUNK_SIZE="1G"  # 1GB chunks
OUTPUT_DIR="dataset_chunks"

echo "Splitting $DATASET_ZIP into $CHUNK_SIZE chunks..."

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Split the zip file
split -b "$CHUNK_SIZE" "$DATASET_ZIP" "$OUTPUT_DIR/konkani_10k_part_"

echo "Split complete! Files created:"
ls -lh "$OUTPUT_DIR"

echo ""
echo "To upload each chunk as a separate dataset:"
echo "1. Create datasets on Kaggle: konkani-10k-part1, konkani-10k-part2, etc."
echo "2. Upload each chunk:"
echo "   kaggle datasets version -p $OUTPUT_DIR/konkani_10k_part_aa -m 'Part 1'"
echo ""
echo "To reassemble in Kaggle notebook:"
echo "   cat /kaggle/input/konkani-10k-part*/konkani_10k_part_* > konkani_10k_dataset.zip"
echo "   unzip konkani_10k_dataset.zip"
