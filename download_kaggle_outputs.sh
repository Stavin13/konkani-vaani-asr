#!/bin/bash
# Download outputs from Kaggle notebook

# Create destination directory
DEST_DIR="kaggle_downloads/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEST_DIR"

echo "Downloading Kaggle notebook outputs..."
echo "Destination: $DEST_DIR"
echo ""

# Download the outputs
kaggle kernels output stavinfernandes/kaggle-train-10k-dual-gpu-new1a184c2110 -p "$DEST_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Download complete!"
    echo "Files saved to: $DEST_DIR"
    echo ""
    echo "Contents:"
    ls -lh "$DEST_DIR"
else
    echo ""
    echo "❌ Download failed. Check your Kaggle API credentials and kernel slug."
fi
