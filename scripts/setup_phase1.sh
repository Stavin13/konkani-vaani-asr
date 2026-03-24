#!/bin/bash
# Setup Phase 1: Install dependencies for beam search + LM

echo "=========================================="
echo "Phase 1 Setup: Beam Search + Language Model"
echo "=========================================="

# Check if virtual environment is active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "WARNING: No virtual environment detected!"
    echo "It's recommended to activate your venv first:"
    echo "  source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Installing required packages..."
echo "=========================================="

# Install pyctcdecode
echo ""
echo "1. Installing pyctcdecode..."
pip install pyctcdecode

# Install KenLM Python bindings
echo ""
echo "2. Installing KenLM Python bindings..."
pip install https://github.com/kpu/kenlm/archive/master.zip

# Install jiwer for metrics
echo ""
echo "3. Installing jiwer (for CER/WER calculation)..."
pip install jiwer

# Install tqdm for progress bars
echo ""
echo "4. Installing tqdm..."
pip install tqdm

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="

# Check if LM files exist
echo ""
echo "Checking for language models..."
if [ -f "models/language_models/konkani_4gram.binary" ]; then
    echo "✓ 4-gram LM found: models/language_models/konkani_4gram.binary"
else
    echo "✗ 4-gram LM not found!"
    echo "  Run: ./scripts/build_kenlm_docker.sh"
fi

if [ -f "models/language_models/konkani_3gram.binary" ]; then
    echo "✓ 3-gram LM found: models/language_models/konkani_3gram.binary"
else
    echo "✗ 3-gram LM not found!"
fi

# Check if model exists
echo ""
echo "Checking for trained model..."
if [ -f "kaggle_asr_outputs/checkpoints/best_model.pt" ]; then
    echo "✓ Model found: kaggle_asr_outputs/checkpoints/best_model.pt"
else
    echo "✗ Model not found: kaggle_asr_outputs/checkpoints/best_model.pt"
    echo "  Make sure your trained model is in the correct location"
fi

# Check if vocab exists
echo ""
echo "Checking for vocabulary..."
if [ -f "data/konkani-mega-dataset/vocab.json" ]; then
    echo "✓ Vocab found: data/konkani-mega-dataset/vocab.json"
else
    echo "✗ Vocab not found: data/konkani-mega-dataset/vocab.json"
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Quick test on single audio file:"
echo "   python scripts/beam_search_decoder.py \\"
echo "     --model kaggle_asr_outputs/checkpoints/best_model.pt \\"
echo "     --vocab data/konkani-mega-dataset/vocab.json \\"
echo "     --audio <path_to_audio.wav> \\"
echo "     --lm models/language_models/konkani_4gram.binary \\"
echo "     --beam-width 15"
echo ""
echo "2. Compare all strategies on test set:"
echo "   python scripts/test_beam_search_improvements.py \\"
echo "     --max-samples 50  # Start with 50 samples for quick test"
echo ""
echo "3. Tune LM parameters on validation set:"
echo "   python scripts/tune_lm_parameters.py \\"
echo "     --max-samples 100"
echo ""
echo "4. Full evaluation on test set:"
echo "   python scripts/test_beam_search_improvements.py"
echo ""
