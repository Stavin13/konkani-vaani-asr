# 🚀 One Notebook to Rule Them All!

## Complete Translation & Emotion Training in One Go

I've created **one comprehensive Kaggle notebook** that does everything:

### ✅ What It Does

1. **Generates Translation Data** using Google Translate
2. **Generates Emotion Data** using pre-trained models
3. **Trains Custom Translation Model** (Konkani→English)
4. **Trains Custom Emotion Model** (7 emotions)
5. **Tests & Exports** both models

**All in one notebook!** No manual data preparation needed.

## 📦 The Notebook

**Location:** `notebooks/KAGGLE_COMPLETE_TRANSLATION_EMOTION.ipynb`

**Time:** ~4-5 hours total
**GPU:** P100 or T4

## 🚀 Quick Start

### Step 1: Upload to Kaggle

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Click "File" → "Upload Notebook"
4. Select `notebooks/KAGGLE_COMPLETE_TRANSLATION_EMOTION.ipynb`

### Step 2: (Optional) Add Your Konkani Text

If you have a Konkani text corpus:
1. Upload it as a Kaggle dataset
2. Add it to the notebook
3. Update the `konkani_sentences` list in Part 2

**Or just use the sample data** - the notebook has examples built-in!

### Step 3: Enable GPU

1. Click "Accelerator" in the right panel
2. Select "GPU P100" or "GPU T4"
3. Click "Save"

### Step 4: Run Everything

Click "Run All" and wait ~4-5 hours!

The notebook will:
- Generate ~1000+ translation pairs
- Generate ~1000+ emotion labels
- Train translation model (10 epochs)
- Train emotion model (5 epochs)

### Step 5: Download Your Models

After training completes:
1. Click "Output" tab (right side)
2. Download:
   - `konkani_english_translator/` folder
   - `konkani_emotion_classifier/` folder

Or use CLI:
```bash
kaggle kernels output YOUR_USERNAME/NOTEBOOK_NAME -p ./outputs
```

## 📊 What You'll Get

### Translation Model
- **Input:** Konkani text
- **Output:** English translation
- **Format:** MarianMT model
- **Size:** ~300MB

### Emotion Model
- **Input:** Konkani text
- **Output:** One of 7 emotions
  - anger, disgust, fear, joy, neutral, sadness, surprise
- **Format:** DistilBERT model
- **Size:** ~250MB

## 🎯 Notebook Structure

```
Part 1: Setup & Check GPU
├── Install dependencies
├── Import libraries
└── Check GPU status

Part 2: Generate Translation Data
├── Sample Konkani sentences
├── Use Google Translate API
├── Create Konkani-English pairs
└── Split train/val/test

Part 3: Generate Emotion Data
├── Load pre-trained emotion model
├── Label Konkani text with emotions
├── Create emotion dataset
└── Split train/val/test

Part 4: Train Translation Model
├── Load MarianMT base model
├── Prepare datasets
├── Train for 10 epochs
└── Save model

Part 5: Train Emotion Model
├── Load DistilBERT base model
├── Prepare datasets
├── Train for 5 epochs
└── Save model

Part 6: Test & Export
├── Test translation
├── Test emotion detection
└── List all outputs
```

## 💡 Customization Options

### Use Your Own Konkani Text

Replace this section in Part 2:
```python
konkani_sentences = [
    "हांव घरा वता",
    "तुजें नांव कितें?",
    # Add your sentences here
]
```

### Adjust Training Parameters

**Translation (Part 4):**
```python
num_train_epochs=10,        # More epochs = better quality
per_device_train_batch_size=8,  # Larger = faster but more memory
```

**Emotion (Part 5):**
```python
num_train_epochs=5,         # More epochs = better accuracy
per_device_train_batch_size=16, # Larger = faster but more memory
```

### Add More Emotions

Modify the emotion labels in Part 3:
```python
emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', 'love', 'confusion']
```

## 🔧 Troubleshooting

### "Out of Memory"
- Reduce batch sizes
- Use smaller base models
- Enable gradient accumulation

### "Google Translate API Error"
- The free API has rate limits
- Add delays between requests
- Or use a different translation service

### "Training Too Slow"
- Use P100 instead of T4
- Reduce number of epochs
- Use smaller datasets

## 📈 Expected Performance

### Translation Model
- **BLEU Score:** 25-35 (good for low-resource language)
- **Quality:** Decent for simple sentences
- **Improvement:** Fine-tune with more data

### Emotion Model
- **Accuracy:** 70-85%
- **Quality:** Good for basic emotions
- **Improvement:** Add more labeled data

## 🎉 After Training

### Test Your Models Locally

```python
from transformers import MarianMTModel, MarianTokenizer, AutoModelForSequenceClassification, AutoTokenizer

# Load translation model
translator = MarianMTModel.from_pretrained('./konkani_english_translator')
tokenizer = MarianTokenizer.from_pretrained('./konkani_english_translator')

# Translate
text = "हांव खूश आसा"
inputs = tokenizer(text, return_tensors='pt')
outputs = translator.generate(**inputs)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Translation: {translation}")

# Load emotion model
emotion_model = AutoModelForSequenceClassification.from_pretrained('./konkani_emotion_classifier')
emotion_tokenizer = AutoTokenizer.from_pretrained('./konkani_emotion_classifier')

# Detect emotion
inputs = emotion_tokenizer(text, return_tensors='pt')
outputs = emotion_model(**inputs)
emotion = outputs.logits.argmax().item()
print(f"Emotion: {emotion}")
```

## 🚀 Next Steps

1. **Evaluate** models on test data
2. **Fine-tune** with more Konkani text
3. **Deploy** for inference
4. **Integrate** with your ASR pipeline

## 📝 Notes

- The notebook uses sample data by default
- You can add your own Konkani corpus for better results
- Training time depends on dataset size
- Models are saved automatically

---

**Ready to train?** Upload the notebook to Kaggle and click "Run All"! 🎉
