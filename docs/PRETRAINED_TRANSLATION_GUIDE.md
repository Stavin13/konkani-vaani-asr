# Using Pre-trained Models for Translation Data Generation

## Problem
Your current translation model outputs the same text ("I am stading") for all inputs because:
- Poor quality training data (corrupted translations)
- Model collapse during training
- Insufficient high-quality translation pairs

## Solution
Use pre-trained translation models to generate high-quality training data, then train your custom model.

---

## Method 1: Google Translate (Easiest, Free)

### Install
```bash
pip install googletrans==4.0.0-rc1
```

### Generate Data
```bash
python scripts/quick_translate_with_google.py
```

**Pros:**
- Free, no API key needed
- Fast and simple
- Works reasonably well for Devanagari script

**Cons:**
- Rate limited (need delays between requests)
- Konkani not directly supported (uses Hindi as proxy)
- Quality varies

**Output:** `data/translation_data/konkani_english_google.json`

---

## Method 2: IndicTrans2 (Best for Indian Languages)

### Install
```bash
pip install transformers torch sentencepiece
```

### Generate Data
```bash
python scripts/generate_translation_data_with_pretrained.py --method indictrans2
```

**Pros:**
- Specifically trained on Indian languages
- Better understanding of Konkani/Hindi/Marathi
- High quality translations
- No rate limits

**Cons:**
- Requires downloading large model (~1-2GB)
- Slower than API calls
- Needs GPU for reasonable speed

**Model:** `ai4bharat/indictrans2-en-indic-1B`

---

## Method 3: Meta M2M100 (Multilingual)

### Install
```bash
pip install transformers torch sentencepiece
```

### Generate Data
```bash
python scripts/generate_translation_data_with_pretrained.py --method m2m100
```

**Pros:**
- Supports 100 languages
- Good quality
- No API key needed

**Cons:**
- Large model download
- Slower inference
- Uses Hindi as proxy for Konkani

**Model:** `facebook/m2m100_418M`

---

## Method 4: Combine Multiple Methods

Generate translations using multiple models and pick the best:

```bash
# Generate with all methods
python scripts/generate_translation_data_with_pretrained.py --method all

# Or generate separately and combine
python scripts/quick_translate_with_google.py
python scripts/generate_translation_data_with_pretrained.py --method indictrans2
python scripts/generate_translation_data_with_pretrained.py --method m2m100
```

Then manually review and select the best translations.

---

## Recommended Workflow

### Step 1: Quick Start (Google Translate)
```bash
# Install
pip install googletrans==4.0.0-rc1

# Generate translations
python scripts/quick_translate_with_google.py
```

This will translate all your Konkani texts from the emotion dataset.

### Step 2: Review & Clean
Open `data/translation_data/konkani_english_google.json` and:
- Check for obvious errors
- Fix any nonsensical translations
- Remove duplicates
- Add manual corrections

### Step 3: Augment Data
Add common phrases manually:
```json
[
  {"konkani": "नमस्कार", "english": "Hello", "confidence": 1.0},
  {"konkani": "धन्यवाद", "english": "Thank you", "confidence": 1.0},
  {"konkani": "हांव बरो आसां", "english": "I am fine", "confidence": 1.0},
  {"konkani": "तुका कसें आसा?", "english": "How are you?", "confidence": 1.0}
]
```

### Step 4: Train Custom Model
Update the training script to use the new data:

```python
# In scripts/train_translation_only.py, change data path:
data_path = Path('data/translation_data/konkani_english_google.json')
```

Then train:
```bash
python scripts/train_translation_only.py
```

---

## Data Quality Tips

### Good Translation Pairs
✅ Clear, natural English
✅ Preserves meaning from Konkani
✅ Proper grammar and spelling
✅ Consistent style

### Bad Translation Pairs
❌ Gibberish or corrupted text
❌ Mixed languages in output
❌ Incomplete translations
❌ Devanagari in English output

### Example Quality Check
```python
# Good
{"konkani": "मका भूक लागली", "english": "I am hungry"}

# Bad (current data)
{"konkani": "मका भूक लागली", "english": "मका भूक लागली"}  # Not translated!
```

---

## Advanced: Use Multiple Models for Consensus

Create a script that:
1. Translates with Google, IndicTrans2, and M2M100
2. Compares outputs
3. Picks the best translation (or flags for manual review)

```python
def get_best_translation(konkani_text):
    translations = {
        'google': translate_google(konkani_text),
        'indictrans': translate_indictrans(konkani_text),
        'm2m100': translate_m2m100(konkani_text)
    }
    
    # If 2+ agree, use that
    # Otherwise, flag for manual review
    return best_translation
```

---

## Estimated Data Needs

For good translation quality:
- **Minimum:** 1,000 high-quality pairs
- **Good:** 5,000-10,000 pairs
- **Excellent:** 50,000+ pairs

Your emotion dataset has ~10k texts, which is a great starting point!

---

## Quick Commands

```bash
# 1. Install Google Translate
pip install googletrans==4.0.0-rc1

# 2. Generate translations
python scripts/quick_translate_with_google.py

# 3. Check output
head -50 data/translation_data/konkani_english_google.json

# 4. Train model with new data
python scripts/train_translation_only.py
```

---

## Troubleshooting

### "googletrans not found"
```bash
pip install googletrans==4.0.0-rc1
```

### "Rate limit exceeded"
- Add longer delays in the script
- Use smaller batches
- Try IndicTrans2 instead (no rate limits)

### "Model download failed"
- Check internet connection
- Try smaller model first
- Use Google Translate as fallback

### "Translations are poor quality"
- Konkani is similar to Marathi/Hindi, so Hindi proxy works reasonably
- Consider using IndicTrans2 for better Indian language support
- Manually review and correct important phrases

---

## Next Steps

1. **Generate data:** Run `python scripts/quick_translate_with_google.py`
2. **Review quality:** Check first 50-100 translations
3. **Train model:** Use the new clean data
4. **Test:** Run `python scripts/test_translation_best.py`
5. **Iterate:** Add more data or fix errors as needed
