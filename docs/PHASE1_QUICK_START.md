# Phase 1 Quick Start Guide

## What is Phase 1?

Phase 1 adds beam search decoding with KenLM language model to your existing trained model. This requires **NO retraining** - just better decoding at inference time.

**Expected improvement**: 20% CER → 14% CER (30% relative improvement)

## Prerequisites

✅ Trained model: `outputs/conformer_ctc_run1/best_conformer_ctc.pt`
✅ 4-gram LM: `models/language_models/konkani_4gram.binary`
✅ Vocabulary: `data/konkani-mega-dataset/vocab.json`
✅ Test data: `data/konkani-mega-dataset/manifests/test.json`

## Step 1: Install Dependencies

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run setup script
./scripts/setup_phase1.sh
```

This installs:
- `pyctcdecode` - Beam search decoder with LM support
- `kenlm` - Language model library
- `jiwer` - For CER/WER metrics
- `tqdm` - Progress bars

## Step 2: Quick Test on Single Audio

Test the decoder on a single audio file:

```bash
python scripts/beam_search_decoder.py \
  --model kaggle_asr_outputs/checkpoints/best_model.pt \
  --vocab data/konkani-mega-dataset/vocab.json \
  --audio data/konkani-mega-dataset/audio/sample.wav \
  --lm models/language_models/konkani_4gram.binary \
  --beam-width 15 \
  --alpha 1.0 \
  --beta 0.0
```

This will show:
1. Greedy decoding output
2. Beam search output

## Step 3: Compare Strategies (Quick Test)

Compare all 4 strategies on a small subset:

```bash
python scripts/test_beam_search_improvements.py \
  --max-samples 50 \
  --device cpu
```

This tests:
1. Greedy decoding (baseline)
2. Beam search (no LM)
3. Beam search + 3-gram LM
4. Beam search + 4-gram LM

**Expected results** (on 50 samples):
```
Strategy                       CER        WER        Time (s)    Speed
--------------------------------------------------------------------------------
greedy                        20.00%     35.00%        10.00     1.00x
beam_no_lm                    18.00%     32.00%        25.00     0.40x
beam_3gram                    15.00%     28.00%        30.00     0.33x
beam_4gram                    14.00%     26.00%        32.00     0.31x
```

## Step 4: Tune LM Parameters (Optional)

Find optimal beam_width, alpha, and beta on validation set:

```bash
python scripts/tune_lm_parameters.py \
  --max-samples 100 \
  --beam-widths 10 15 20 \
  --alphas 0.5 1.0 1.5 \
  --betas 0.0 1.0 2.0
```

This runs grid search over 27 combinations (3×3×3) and finds the best parameters.

**Typical best parameters**:
- `beam_width`: 15-20
- `alpha` (LM weight): 1.0-1.5
- `beta` (word bonus): 0.0-1.0

## Step 5: Full Evaluation

Run full evaluation on entire test set:

```bash
python scripts/test_beam_search_improvements.py \
  --beam-width 15 \
  --alpha 1.0 \
  --beta 0.0 \
  --device cpu
```

Results saved to: `outputs/beam_search_comparison.json`

## Step 6: Use in Production

Update your inference code to use beam search + LM:

```python
from scripts.beam_search_decoder import BeamSearchDecoder, load_model, decode_audio

# Load model
model, vocab_size = load_model('kaggle_asr_outputs/checkpoints/best_model.pt')

# Create decoder with LM
decoder = BeamSearchDecoder(
    vocab_path='data/konkani-mega-dataset/vocab.json',
    lm_path='models/language_models/konkani_4gram.binary',
    alpha=1.0,  # Use tuned value
    beta=0.0    # Use tuned value
)

# Decode audio
text = decode_audio(model, 'audio.wav', decoder, beam_width=15)
print(text)
```

## Troubleshooting

### Error: "No module named 'pyctcdecode'"

```bash
pip install pyctcdecode
```

### Error: "No module named '_kenlm'"

```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### Error: "vocab.json not found"

Make sure you're using the correct vocab path. Check:
- `data/konkani-mega-dataset/vocab.json`
- `data/vocab.json`
- `kaggle_retrain_fixed/vocab.json`

### Slow inference on CPU

Beam search is slower than greedy decoding. Options:
1. Use smaller beam width (10 instead of 15)
2. Use GPU: `--device cuda`
3. Process in batches (not yet implemented)

### Poor results with LM

Try tuning parameters:
1. Increase alpha (LM weight): 1.0 → 1.5
2. Adjust beta (word bonus): 0.0 → 1.0
3. Use larger beam width: 15 → 20

## Expected Timeline

- Setup: 5 minutes
- Quick test (50 samples): 2-5 minutes
- Parameter tuning (100 samples): 10-20 minutes
- Full evaluation (all test samples): 30-60 minutes

## What's Next?

After Phase 1 shows good results:

**Phase 2**: Retrain with improvements
- Enable FP16 mixed precision
- Add SpecAugment
- Increase batch size with gradient accumulation
- Train for more epochs

**Phase 3**: Advanced improvements
- Switch to BPE/subword tokens
- Quantize model (INT8)
- Optimize for deployment

## Files Created

- `scripts/beam_search_decoder.py` - Core decoder implementation
- `scripts/test_beam_search_improvements.py` - Comparison script
- `scripts/tune_lm_parameters.py` - Parameter tuning
- `scripts/setup_phase1.sh` - Installation script
- `docs/PHASE1_QUICK_START.md` - This guide

## Support

If you encounter issues:
1. Check that all prerequisites exist
2. Verify vocab.json format matches model
3. Test on single audio file first
4. Use `--max-samples 10` for quick debugging
