#!/bin/bash
# Install KenLM from source on macOS

set -e

echo "=========================================="
echo "Installing KenLM from source"
echo "=========================================="

# Check if already installed
if command -v lmplz &> /dev/null; then
    echo "✓ KenLM is already installed!"
    lmplz --help 2>&1 | head -5
    exit 0
fi

# Check dependencies
echo "Checking dependencies..."

if ! command -v cmake &> /dev/null; then
    echo "Installing cmake..."
    brew install cmake
fi

# Check for Boost
if ! brew list boost &> /dev/null; then
    echo "Installing boost..."
    brew install boost
fi

if ! command -v git &> /dev/null; then
    echo "ERROR: git not found. Please install git first."
    exit 1
fi

# Create temp directory
TEMP_DIR=$(mktemp -d)
echo "Working in: $TEMP_DIR"

cd "$TEMP_DIR"

# Clone KenLM
echo ""
echo "Cloning KenLM repository..."
git clone https://github.com/kpu/kenlm.git
cd kenlm

# Build
echo ""
echo "Building KenLM (this may take a few minutes)..."
mkdir -p build
cd build
cmake ..
make -j4

# Install binaries to /usr/local/bin
echo ""
echo "Installing binaries..."
sudo cp bin/lmplz /usr/local/bin/
sudo cp bin/build_binary /usr/local/bin/
sudo cp bin/query /usr/local/bin/

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

if command -v lmplz &> /dev/null; then
    echo "✓ lmplz installed successfully"
    lmplz --help 2>&1 | head -3
else
    echo "✗ Installation failed"
    exit 1
fi

if command -v build_binary &> /dev/null; then
    echo "✓ build_binary installed successfully"
else
    echo "✗ build_binary installation failed"
    exit 1
fi

# Cleanup
echo ""
echo "Cleaning up..."
cd ~
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "✓ KenLM installed successfully!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  python scripts/build_kenlm_models.py"
