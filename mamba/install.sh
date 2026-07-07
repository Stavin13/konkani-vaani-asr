#!/bin/bash
# =============================================================================
# Full setup script for Mamba-130M ASR post-correction training
# Optimized for a 20GB VRAM NVIDIA GPU with CUDA support.
# =============================================================================

set -e  # exit on any error

echo "================================================================"
echo "Mamba-130M ASR Post-Correction Environment Setup"
echo "================================================================"

# ---------------------------------------------------------------------
# 1. Check for Python 3.8+
# ---------------------------------------------------------------------
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ $(echo "$PYTHON_VERSION < 3.8" | bc) -eq 1 ]]; then
    echo "ERROR: Python $PYTHON_VERSION is too old. Need 3.8+."
    exit 1
fi
echo "Using Python $PYTHON_VERSION"

# ---------------------------------------------------------------------
# 2. Create and activate a virtual environment
# ---------------------------------------------------------------------
ENV_NAME="mamba_asr"
if [ -d "$ENV_NAME" ]; then
    echo "Virtual environment '$ENV_NAME' already exists. Removing it..."
    rm -rf "$ENV_NAME"
fi

echo "Creating virtual environment '$ENV_NAME'..."
python3 -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"

# ---------------------------------------------------------------------
# 3. Upgrade pip and install PyTorch with CUDA
# ---------------------------------------------------------------------
echo "Upgrading pip..."
pip install --upgrade pip

# Detect CUDA version (driver) and choose appropriate PyTorch index.
# If detection fails, default to CUDA 11.8.
CUDA_VERSION=""
if command -v nvidia-smi &> /dev/null; then
    CUDA_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    echo "Detected NVIDIA driver version: $CUDA_DRIVER"
    # Map driver version to a compatible CUDA toolkit version (simplified)
    # For most modern drivers, CUDA 11.8 or 12.1 work.
    # We'll default to 11.8 for broad compatibility.
    CUDA_VERSION="118"
else
    echo "WARNING: nvidia-smi not found. Assuming CUDA 11.8."
    CUDA_VERSION="118"
fi

# You can change this to "cu121" if your driver is newer (>= 525)
# For 20GB GPUs (RTX 3090/4090, A10, etc.) CUDA 11.8 is safe.
TORCH_INDEX="https://download.pytorch.org/whl/cu${CUDA_VERSION}"

echo "Installing PyTorch from $TORCH_INDEX..."
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"

# ---------------------------------------------------------------------
# 4. Install core ML / NLP packages
# ---------------------------------------------------------------------
echo "Installing transformers, peft, sentencepiece, and other dependencies..."
pip install transformers peft sentencepiece pandas numpy scikit-learn pytorch-lightning

# ---------------------------------------------------------------------
# 5. Install Mamba-specific CUDA kernels (mamba-ssm, causal-conv1d)
# ---------------------------------------------------------------------
echo "Installing mamba-ssm and causal-conv1d (may take a few minutes)..."
# These packages have compiled CUDA extensions – they need a C++ compiler.
# If you have an older CUDA toolkit, you may need to install from source.
# We try pip first; if it fails, we give instructions.
if ! pip install mamba-ssm causal-conv1d; then
    echo "ERROR: pip install of mamba-ssm/causal-conv1d failed."
    echo "This usually happens because pre-built wheels are not available for your CUDA version."
    echo "To build from source, install: ninja, and ensure CUDA toolkit is installed."
    echo "Then run: pip install mamba-ssm causal-conv1d --no-build-isolation"
    echo "Alternatively, try: pip install mamba-ssm causal-conv1d --index-url https://download.pytorch.org/whl/cu118"
    exit 1
fi

# ---------------------------------------------------------------------
# 6. (Optional) Install additional utilities
# ---------------------------------------------------------------------
# wandb is not used in the script, but you might want it for logging.
# pip install wandb

# ---------------------------------------------------------------------
# 7. Verify installation
# ---------------------------------------------------------------------
echo "Verifying key packages..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); import mamba_ssm; import causal_conv1d; print('mamba-ssm and causal-conv1d loaded successfully.')" || {
    echo "Verification failed. Please check the installation."
    exit 1
}

# ---------------------------------------------------------------------
# 8. Finish
# ---------------------------------------------------------------------
echo "================================================================"
echo "Setup completed successfully!"
echo "To activate the environment, run:"
echo "    source $ENV_NAME/bin/activate"
echo "Then you can launch your training script with:"
echo "    python your_script.py"
echo "================================================================"