# Phase 1: Add Beam Search + Language Model

## Overview

Add beam search decoding with KenLM language model fusion to your existing `best_conformer_ctc.pt` model. This requires NO retraining - just better decoding at inference time.

## Expected Improvements

- Greedy decoding (current): Baseline
- Beam search (no LM): +10% relative improvement
- Beam search + 4-gram LM: +30% relative improvement

Example: If current CER is 20%, you'll get down to ~14% CER.

## What You Need

✅ Your trained model: `outputs/conformer_ctc_run1/best_conformer_ctc.pt`
✅ 4-gram LM: `models/language_models/konkani_4gram.binary`
✅ Test data: `data/konkani-mega-dataset/manifests/test.json`

## Implementation Steps

### Step 1: Install pyctcdecode

```bash
pip install pyctcdecode
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### Step 2: Create Beam Search Decoder

See `scripts/beam_search_decoder.py` for implementation.

### Step 3: Test and Compare

Run `scripts/test_beam_search_improvements.py` to compare:
- Greedy vs Beam (no LM) vs Beam + LM

### Step 4: Integrate into Production

Update your inference code to use beam search + LM by default.

## Parameters to Tune

- **beam_width**: Start with 15, try 10-20
- **lm_weight (alpha)**: Start with 1.0, try 0.5-1.5
- **word_bonus (beta)**: Start with 0, try 0-2.0

Tune on validation set, not test set!

## Files Created

- `scripts/beam_search_decoder.py` - Beam search implementation
- `scripts/test_beam_search_improvements.py` - Testing script
- `scripts/tune_lm_parameters.py` - Parameter tuning script

## Next Steps

After Phase 1 shows good results:
- Phase 2: Retrain with FP16, SpecAugment, larger batches
- Phase 3: Switch to BPE tokens, quantization
