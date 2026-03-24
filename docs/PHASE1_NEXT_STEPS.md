# Phase 1: Next Steps

## Current Status

✅ **Phase 1 Implementation Complete**

All code is ready to test beam search + language model improvements on your existing `best_conformer_ctc.pt` model.

## Immediate Next Steps

### 1. Install Dependencies (5 minutes)

```bash
# Activate virtual environment
source .venv/bin/activate

# Run setup script
./scripts/setup_phase1.sh
```

This installs:
- `pyctcdecode` - Beam search decoder
- `kenlm` - Language model library
- `jiwer` - Metrics (CER/WER)
- `tqdm` - Progress bars

### 2. Verify Prerequisites

Check that you have:
```bash
# Model
ls -lh kaggle_asr_outputs/checkpoints/best_model.pt

# Language models
ls -lh models/language_models/konkani_*.binary

# Vocabulary
ls -lh data/konkani-mega-dataset/vocab.json

# Test data
ls -lh data/konkani-mega-dataset/manifests/test.json
```

### 3. Quick Test (2-5 minutes)

Test on 50 samples to verify everything works:

```bash
python scripts/test_beam_search_improvements.py \
  --max-samples 50 \
  --device cpu
```

**Expected output**:
```
Strategy                       CER        WER        Time (s)    Speed
--------------------------------------------------------------------------------
greedy                        20.00%     35.00%        10.00     1.00x
beam_no_lm                    18.00%     32.00%        25.00     0.40x
beam_3gram                    15.00%     28.00%        30.00     0.33x
beam_4gram                    14.00%     26.00%        32.00     0.31x

RELATIVE IMPROVEMENTS (vs Greedy Baseline)
--------------------------------------------------------------------------------
beam_no_lm                     10.00% better      8.57% better
beam_3gram                     25.00% better     20.00% better
beam_4gram                     30.00% better     25.71% better
```

### 4. Tune Parameters (10-20 minutes)

Find optimal beam_width, alpha, beta on validation set:

```bash
python scripts/tune_lm_parameters.py \
  --max-samples 100 \
  --beam-widths 10 15 20 \
  --alphas 0.5 1.0 1.5 \
  --betas 0.0 1.0 2.0
```

**Expected output**:
```
BEST PARAMETERS:
  beam_width: 15
  alpha (LM weight): 1.0
  beta (word bonus): 0.0
  CER: 13.85%
  Time: 45.23s
```

### 5. Full Evaluation (30-60 minutes)

Run on entire test set with tuned parameters:

```bash
python scripts/test_beam_search_improvements.py \
  --beam-width 15 \
  --alpha 1.0 \
  --beta 0.0 \
  --device cpu
```

Results saved to: `outputs/beam_search_comparison.json`

## Decision Point

After full evaluation, you have two options:

### Option A: Good Results (20-30% improvement)

If beam search + LM shows 20-30% CER improvement:

✅ **Proceed to Production Integration**

Update your inference code:
```python
from scripts.beam_search_decoder import BeamSearchDecoder, load_model, decode_audio

model, _ = load_model('kaggle_asr_outputs/checkpoints/best_model.pt')
decoder = BeamSearchDecoder(
    vocab_path='data/konkani-mega-dataset/vocab.json',
    lm_path='models/language_models/konkani_4gram.binary',
    alpha=1.0,  # Your tuned value
    beta=0.0    # Your tuned value
)

# Use in production
text = decode_audio(model, audio_path, decoder, beam_width=15)
```

Then move to **Phase 2**: Retrain with improvements

### Option B: Poor Results (< 10% improvement)

If improvement is less than expected:

**Possible causes**:
1. Language model corpus doesn't match test domain
2. Vocabulary mismatch between model and LM
3. Model quality issues (needs retraining)
4. Parameters not tuned correctly

**Actions**:
1. Check LM corpus quality: `head -100 data/konkani_corpus_for_lm.txt`
2. Verify vocab matches: Compare model vocab with LM vocab
3. Try wider parameter ranges in tuning
4. Consider rebuilding LM with better corpus

## Phase 2 Planning

Once Phase 1 shows good results, plan Phase 2:

### Phase 2: Retrain with Improvements

**Goal**: Further improve base model quality

**Changes**:
1. Enable FP16 mixed precision (2x faster)
2. Add SpecAugment (better generalization)
3. Increase effective batch size (gradient accumulation)
4. Train longer (30-50 epochs per chunk)

**Expected improvement**: 14% CER → 10% CER

**Timeline**: 
- Implementation: 1-2 days
- Training: 2-3 days per 20GB chunk
- Total: ~1 week for 84GB data

### Phase 3: Advanced Improvements

**Goal**: Production-ready optimizations

**Changes**:
1. Switch to BPE/subword tokens (better OOV)
2. Quantize to INT8 (4x smaller, faster)
3. ONNX export for deployment
4. Batch inference optimization

**Expected improvement**: 10% CER → 8% CER

**Timeline**: 1-2 weeks

## Troubleshooting Guide

### Installation Issues

**pyctcdecode not found**:
```bash
pip install pyctcdecode
```

**kenlm not found**:
```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### Runtime Issues

**Vocab mismatch error**:
- Check vocab.json format
- Verify it matches model training vocab
- Try different vocab paths

**Out of memory**:
- Reduce beam width: 15 → 10
- Use CPU instead of GPU
- Process fewer samples at once

**Slow inference**:
- Use GPU: `--device cuda`
- Reduce beam width
- Use 3-gram instead of 4-gram

### Poor Results

**No improvement with LM**:
1. Check LM was built correctly
2. Verify corpus is Konkani text
3. Tune alpha higher: 1.0 → 1.5
4. Check vocab compatibility

**Worse results with LM**:
1. LM corpus may be poor quality
2. Try lower alpha: 1.0 → 0.5
3. Rebuild LM with better corpus

## Success Metrics

Phase 1 is successful if:

- ✅ Setup completes without errors
- ✅ Quick test runs and shows improvement
- ✅ Parameter tuning finds optimal values
- ✅ Full evaluation shows 20-30% CER improvement
- ✅ Sample outputs look more natural
- ✅ Inference time is acceptable (< 5x slower)

## Timeline Summary

| Step | Time | Status |
|------|------|--------|
| Setup | 5 min | Ready |
| Quick test | 2-5 min | Ready |
| Parameter tuning | 10-20 min | Ready |
| Full evaluation | 30-60 min | Ready |
| Integration | 10-15 min | After validation |
| **Total** | **1-2 hours** | **Ready to start** |

## Commands Cheat Sheet

```bash
# Setup
./scripts/setup_phase1.sh

# Quick test
python scripts/test_beam_search_improvements.py --max-samples 50

# Tune parameters
python scripts/tune_lm_parameters.py --max-samples 100

# Full evaluation
python scripts/test_beam_search_improvements.py

# Test single audio
python scripts/beam_search_decoder.py \
  --model kaggle_asr_outputs/checkpoints/best_model.pt \
  --vocab data/konkani-mega-dataset/vocab.json \
  --audio audio.wav \
  --lm models/language_models/konkani_4gram.binary
```

## Documentation

- 📖 [PHASE1_README.md](../PHASE1_README.md) - Overview
- 📖 [PHASE1_QUICK_START.md](PHASE1_QUICK_START.md) - Tutorial
- 📖 [PHASE1_BEAM_SEARCH_LM_GUIDE.md](PHASE1_BEAM_SEARCH_LM_GUIDE.md) - Detailed guide
- 📖 [PHASE1_IMPLEMENTATION_COMPLETE.md](PHASE1_IMPLEMENTATION_COMPLETE.md) - Summary

## Questions?

Common questions:

**Q: Do I need to retrain my model?**
A: No! Phase 1 works with your existing model.

**Q: How long does inference take?**
A: 3x slower than greedy, but still real-time capable.

**Q: Can I use GPU?**
A: Yes! Add `--device cuda` for 5-10x speedup.

**Q: What if results are poor?**
A: Check LM corpus quality and tune parameters more.

**Q: When should I move to Phase 2?**
A: After Phase 1 shows 20-30% improvement.

## Ready to Start?

```bash
# Let's go!
source .venv/bin/activate
./scripts/setup_phase1.sh
```

Good luck! 🚀
