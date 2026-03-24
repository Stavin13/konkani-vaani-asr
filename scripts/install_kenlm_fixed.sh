#!/bin/bash
# Install KenLM from source on macOS (fixed for Boost 1.90)

set -e

echo "=========================================="
echo "Installing KenLM from source"
echo "=========================================="

# Check if already installed
if command -v lmplz &> /dev/null; then
    echo "✓ KenLM is already installed!"
    exit 0
fi

# Create temp directory
TEMP_DIR=$(mktemp -d)
echo "Working in: $TEMP_DIR"
cd "$TEMP_DIR"

# Clone KenLM
echo "Cloning KenLM..."
git clone https://github.com/kpu/kenlm.git
cd kenlm

# Build with specific Boost settings
echo "Building KenLM..."
mkdir -p build
cd build

# Use cmake with Boost_NO_BOOST_CMAKE to avoid the config issue
cmake .. \
    -DKENLM_MAX_ORDER=6 \
    -DBoost_NO_BOOST_CMAKE=ON \
    -DBOOST_ROOT=/opt/homebrew

make -j4

# Install
echo "Installing binaries..."
sudo cp bin/lmplz /usr/local/bin/
sudo cp bin/build_binary /usr/local/bin/

# Verify
if command -v lmplz &> /dev/null; then
    echo "✓ KenLM installed successfully!"
else
    echo "✗ Installation failed"
    exit 1
fi

# Cleanup
rm -rf "$TEMP_DIR"

echo "You can now run: python scripts/build_kenlm_models.py"
