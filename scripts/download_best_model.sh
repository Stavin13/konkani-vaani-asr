#!/bin/bash
# Download best_model.pt from Kaggle kernel output

set -e

echo "================================================"
echo "Downloading best_model.pt from Kaggle"
echo "================================================"

# Create destination directory
DEST_DIR="./kaggle_best_model"
mkdir -p "$DEST_DIR"

echo ""
echo "Downloading from kernel: stavin12/final"
echo "Destination: $DEST_DIR"
echo ""

# Download kernel output
kaggle kernels output stavin12/final -p "$DEST_DIR"

echo ""
echo "================================================"
echo "Download complete!"
echo "================================================"

# List downloaded files
echo ""
echo "Downloaded files:"
ls -lh "$DEST_DIR"

# Check if best_model.pt exists
if [ -f "$DEST_DIR/best_model.pt" ]; then
    echo ""
    echo "✓ best_model.pt found!"
    echo "  Size: $(ls -lh "$DEST_DIR/best_model.pt" | awk '{print $5}')"
    
    # Test if it's valid
    echo ""
    echo "Testing checkpoint validity..."
    python3 << 'PYEOF'
import torch
import sys

try:
    checkpoint = torch.load("./kaggle_best_model/best_model.pt", map_location='cpu')
    print("✓ Checkpoint loads successfully!")
    
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")
    if 'model_state_dict' in checkpoint:
        print(f"  Model weights: {len(checkpoint['model_state_dict'])} layers")
    
    sys.exit(0)
except Exception as e:
    print(f"✗ Error loading checkpoint: {e}")
    sys.exit(1)
PYEOF
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Checkpoint is valid!"
        echo ""
        echo "To use this model, copy it to your checkpoints directory:"
        echo "  cp $DEST_DIR/best_model.pt kaggle_asr_outputs/checkpoints/best_model_fixed.pt"
    else
        echo ""
        echo "⚠ Checkpoint may be corrupted"
    fi
else
    echo ""
    echo "⚠ best_model.pt not found in downloaded files"
    echo "Available files:"
    ls -la "$DEST_DIR"
fi

echo ""
echo "================================================"
