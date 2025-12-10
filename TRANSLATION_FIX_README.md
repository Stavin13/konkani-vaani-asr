# Fix Translation Model - Quick Guide

## Problem
Your translation model outputs "I am stading" for everything because it was trained on poor quality data.

## Solution
Use pre-trained models (Google Translate, IndicTrans2, etc.) to generate clean training data, then retrain.

---

## Quick Fix (5 minutes)

### Step 1: Install Google Translate
```bash
pip install googletrans==4.0.0-rc1
```

### Step 2: Generate Clean Translations
```bash
python scripts/quick_translate_with_google.py
```

This will:
- Load all your Konkani texts (~10k from emotion dataset)
- Translate them using Google Translate
- Save to `data/translation_data/konkani_english_google.json`

### Step 3: Retrain Model
```bash
python scripts/retrain_translation_with_clean_data.py
```

This will:
- Load the clean translations
- Train a new model (20-30 minutes on Mac GPU)
- Save best model to `checkpoints/translation_model/translation_model_best.pt`

### Step 4: Test
```bash
python scripts/test_translation_best.py
```

---

## Alternative Methods

### Method 1: Google Translate (Easiest)
```bash
pip install googletrans==4.0.0-rc1
python scripts/quick_translate_with_google.py
```
- ✅ Free, no API key
- ✅ Fast
- ⚠️ Uses Hindi as proxy for Konkani

### Method 2: IndicTrans2 (Best Quality)
```bash
pip install transformers torch
python scripts/generate_translation_data_with_pretrained.py --method indictrans2
```
- ✅ Trained on Indian languages
- ✅ Better for Konkani
- ⚠️ Requires ~2GB download

### Method 3: Meta M2M100
```bash
pip install transformers torch
python scripts/generate_translation_data_with_pretrained.py --method m2m100
```
- ✅ Supports 100 languages
- ⚠️ Large model

---

## Files Created

1. **scripts/quick_translate_with_google.py** - Fast translation with Google
2. **scripts/generate_translation_data_with_pretrained.py** - Multiple model support
3. **scripts/retrain_translation_with_clean_data.py** - Retrain with clean data
4. **docs/PRETRAINED_TRANSLATION_GUIDE.md** - Detailed guide

---

## Expected Results

### Before (Current)
```
Konkani: नमस्कार
English: I am stading

Konkani: तुका कसें आसा?
English: I am stading
```

### After (With Clean Data)
```
Konkani: नमस्कार
English: Hello

Konkani: तुका कसें आसा?
English: How are you?
```

---

## Troubleshooting

### "googletrans not found"
```bash
pip install googletrans==4.0.0-rc1
```

### "No clean translation data found"
Run the translation generation first:
```bash
python scripts/quick_translate_with_google.py
```

### "Rate limit exceeded"
Google Translate has rate limits. The script includes delays, but if you hit limits:
- Wait a few minutes
- Try IndicTrans2 instead (no rate limits)

### "Translations still poor"
1. Check the generated data quality:
   ```bash
   head -100 data/translation_data/konkani_english_google.json
   ```
2. Manually fix important phrases
3. Try IndicTrans2 for better quality

---

## Next Steps

1. **Generate data:** `python scripts/quick_translate_with_google.py`
2. **Review quality:** Check first 50 translations in the JSON file
3. **Retrain:** `python scripts/retrain_translation_with_clean_data.py`
4. **Test:** `python scripts/test_translation_best.py`
5. **Iterate:** Add more data or corrections as needed

---

## Data Quality Tips

Good translations should:
- ✅ Be in proper English (not Devanagari)
- ✅ Preserve meaning from Konkani
- ✅ Have correct grammar
- ✅ Be natural sounding

Bad translations:
- ❌ Mixed scripts (Devanagari in English)
- ❌ Gibberish or corrupted text
- ❌ Untranslated (same as input)
- ❌ Incomplete sentences

---

## Full Documentation

See `docs/PRETRAINED_TRANSLATION_GUIDE.md` for complete details on all methods.
