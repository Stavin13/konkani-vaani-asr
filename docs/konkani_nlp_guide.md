# Complete Technical Guide: Building NLP Models for Konkani Language

## Purpose
A comprehensive technical guide for building NLP models for Konkani language, covering the entire pipeline from data collection to model deployment. Konkani is written in multiple scripts (Devanagari, Kannada, Latin, Malayalam) with Devanagari being most common.

## Core Workflow
```
Define task → Collect data → Annotate → Clean/normalize → Tokenize → Split datasets → Save in multiple formats
```

## Dataset Size Guidelines

| Stage | Examples | Use Case |
|-------|----------|----------|
| Prototype | 1K-5K | Initial testing |
| Production | 10K-50K | Basic deployment |
| Strong Production | 50K-200K+ | Robust systems |
| Translation | 100K+ pairs | Machine translation |

## Three Example Tasks Covered

### 1. Sentiment Classification
Classifies Konkani reviews/comments as positive/negative/neutral

### 2. Named Entity Recognition (NER)
Token-level classification using BIO format (Person, Location, Organization)

### 3. Machine Translation
Parallel corpus format (English ↔ Konkani, Marathi ↔ Konkani)

---

## Task 1: Sentiment Classification

### Data Collection & Annotation

#### Sample Data Format (CSV)
```csv
text,label
"हो चित्रपट खूब बरो आसा",positive
"सेवा खूब वायट आशिल्ली",negative
"ठीक आसा पूण काय खास ना",neutral
"Hea restaurantacho khana khup baro",positive
```

### Data Preparation Script

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import h5py

# Load and prepare data
def prepare_sentiment_data(csv_path):
    """
    Prepare Konkani sentiment data with proper UTF-8 encoding
    """
    # Read CSV with UTF-8 encoding (critical for Konkani Devanagari/Latin)
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Clean and normalize text
    df['text'] = df['text'].str.strip()
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
    
    # Encode labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    df['label_encoded'] = df['label'].map(label_map)
    
    # Split: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    return train_df, val_df, test_df, label_map

# Save in multiple formats
def save_datasets(train_df, val_df, test_df, output_dir='data/sentiment'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. CSV format (easy inspection)
    train_df.to_csv(f'{output_dir}/train.csv', index=False, encoding='utf-8')
    val_df.to_csv(f'{output_dir}/val.csv', index=False, encoding='utf-8')
    test_df.to_csv(f'{output_dir}/test.csv', index=False, encoding='utf-8')
    
    # 2. JSONL format (streaming)
    train_df.to_json(f'{output_dir}/train.jsonl', orient='records', lines=True, force_ascii=False)
    val_df.to_json(f'{output_dir}/val.jsonl', orient='records', lines=True, force_ascii=False)
    test_df.to_json(f'{output_dir}/test.jsonl', orient='records', lines=True, force_ascii=False)
    
    # 3. HDF5 format (efficient binary storage)
    with h5py.File(f'{output_dir}/dataset.h5', 'w') as f:
        for name, data in [('train', train_df), ('val', val_df), ('test', test_df)]:
            grp = f.create_group(name)
            grp.create_dataset('text', data=data['text'].values.astype('S'))
            grp.create_dataset('label', data=data['label_encoded'].values)

# Usage
train_df, val_df, test_df, label_map = prepare_sentiment_data('konkani_reviews.csv')
save_datasets(train_df, val_df, test_df)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

### Model Training: TensorFlow/Keras (LSTM-based)

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Tokenization for Konkani
def create_tokenizer(texts, max_words=10000):
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    return tokenizer

# Prepare sequences
tokenizer = create_tokenizer(train_df['text'])
max_length = 100

X_train = pad_sequences(tokenizer.texts_to_sequences(train_df['text']), maxlen=max_length)
X_val = pad_sequences(tokenizer.texts_to_sequences(val_df['text']), maxlen=max_length)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_df['text']), maxlen=max_length)

y_train = train_df['label_encoded'].values
y_val = val_df['label_encoded'].values
y_test = test_df['label_encoded'].values

# Build LSTM model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_length),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: positive, neutral, negative
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
)

# Save model
model.save('konkani_sentiment_lstm.h5')

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### Model Training: PyTorch/HuggingFace (Transformer-based)

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Custom Dataset
class KonkaniSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# Use multilingual BERT (supports Devanagari script)
model_name = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare datasets
train_dataset = KonkaniSentimentDataset(
    train_df['text'].tolist(),
    train_df['label_encoded'].tolist(),
    tokenizer
)
val_dataset = KonkaniSentimentDataset(
    val_df['text'].tolist(),
    val_df['label_encoded'].tolist(),
    tokenizer
)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train
trainer.train()

# Save model
model.save_pretrained('./konkani_sentiment_bert')
tokenizer.save_pretrained('./konkani_sentiment_bert')

# Evaluate
test_dataset = KonkaniSentimentDataset(
    test_df['text'].tolist(),
    test_df['label_encoded'].tolist(),
    tokenizer
)
results = trainer.evaluate(test_dataset)
print(f"Test Results: {results}")
```

---

## Task 2: Named Entity Recognition (NER)

### Data Format (BIO Tagging)

#### Sample Data (JSONL)
```json
{"tokens": ["रमेश", "गोव्यांत", "रावता"], "tags": ["B-PER", "B-LOC", "O"]}
{"tokens": ["कोंकणी", "साहित्य", "अकादेमी", "मुंबयांत", "आसा"], "tags": ["B-ORG", "I-ORG", "I-ORG", "B-LOC", "O"]}
{"tokens": ["Ramesh", "Goyant", "ravta"], "tags": ["B-PER", "B-LOC", "O"]}
```

### Data Preparation

```python
import json

def load_ner_data(jsonl_path):
    """Load NER data in BIO format"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def prepare_ner_datasets(data_path):
    """Split NER data into train/val/test"""
    data = load_ner_data(data_path)
    
    # Shuffle
    np.random.seed(42)
    np.random.shuffle(data)
    
    # Split
    n = len(data)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    return train_data, val_data, test_data

def save_ner_datasets(train_data, val_data, test_data, output_dir='data/ner'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, dataset in [('train', train_data), ('val', val_data), ('test', test_data)]:
        with open(f'{output_dir}/{name}.jsonl', 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Create tag vocabulary
def create_tag_vocab(data):
    tags = set()
    for item in data:
        tags.update(item['tags'])
    tag2id = {tag: idx for idx, tag in enumerate(sorted(tags))}
    tag2id['PAD'] = len(tag2id)
    id2tag = {idx: tag for tag, idx in tag2id.items()}
    return tag2id, id2tag

# Usage
train_data, val_data, test_data = prepare_ner_datasets('konkani_ner.jsonl')
save_ner_datasets(train_data, val_data, test_data)
tag2id, id2tag = create_tag_vocab(train_data + val_data + test_data)
print(f"Tags: {tag2id}")
```

### NER Model Training: PyTorch/HuggingFace

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from datasets import Dataset as HFDataset

# Prepare data for HuggingFace
def prepare_hf_ner_data(data, tag2id):
    formatted_data = []
    for item in data:
        formatted_data.append({
            'tokens': item['tokens'],
            'ner_tags': [tag2id[tag] for tag in item['tags']]
        })
    return HFDataset.from_list(formatted_data)

# Tokenize and align labels
def tokenize_and_align_labels(examples, tokenizer, tag2id):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # Subword tokens
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Load model and tokenizer
model_name = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare datasets
train_hf = prepare_hf_ner_data(train_data, tag2id)
val_hf = prepare_hf_ner_data(val_data, tag2id)

train_tokenized = train_hf.map(
    lambda x: tokenize_and_align_labels(x, tokenizer, tag2id),
    batched=True
)
val_tokenized = val_hf.map(
    lambda x: tokenize_and_align_labels(x, tokenizer, tag2id),
    batched=True
)

# Model
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(tag2id),
    id2label=id2tag,
    label2id=tag2id
)

# Training
data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir='./ner_results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

# Save
model.save_pretrained('./konkani_ner_bert')
tokenizer.save_pretrained('./konkani_ner_bert')
```

---

## Task 3: Machine Translation (English ↔ Konkani)

### Data Format (Parallel Corpus)

#### Sample Data (TSV)
```tsv
en	kok
Hello, how are you?	नमस्कार, तूं कसो आसा?
I am going to the market	हांव बाजारांत वता
This food is delicious	हें जेवण खूब बरें आसा
Good morning	सकाळीं नमस्कार
Where is the temple?	मंदीर खंय आसा?
```

### Data Preparation

```python
def load_translation_data(tsv_path):
    """Load parallel corpus"""
    data = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                data.append({'en': parts[0], 'kok': parts[1]})
    return data

def prepare_translation_datasets(tsv_path):
    """Split translation data"""
    data = load_translation_data(tsv_path)
    
    # Shuffle
    np.random.seed(42)
    np.random.shuffle(data)
    
    # Split
    n = len(data)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    return train_data, val_data, test_data

def save_translation_datasets(train_data, val_data, test_data, output_dir='data/translation'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, dataset in [('train', train_data), ('val', val_data), ('test', test_data)]:
        # TSV format
        with open(f'{output_dir}/{name}.tsv', 'w', encoding='utf-8') as f:
            f.write('en\tkok\n')
            for item in dataset:
                f.write(f"{item['en']}\t{item['kok']}\n")
        
        # JSONL format
        with open(f'{output_dir}/{name}.jsonl', 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Usage
train_data, val_data, test_data = prepare_translation_datasets('en_kok_parallel.tsv')
save_translation_datasets(train_data, val_data, test_data)
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
```

### Translation Model Training: PyTorch/HuggingFace (mBART)

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import Dataset as HFDataset

# Prepare dataset
def prepare_translation_hf_data(data):
    return HFDataset.from_list(data)

# Tokenization function
def preprocess_function(examples, tokenizer, src_lang='en_XX', tgt_lang='hi_IN'):
    # Note: Konkani not directly supported, using Hindi (hi_IN) as proxy for Devanagari
    inputs = [ex for ex in examples['en']]
    targets = [ex for ex in examples['kok']]
    
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Load model and tokenizer
model_name = 'facebook/mbart-large-50-many-to-many-mmt'
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Set source and target languages
tokenizer.src_lang = 'en_XX'
tokenizer.tgt_lang = 'hi_IN'  # Using Hindi as proxy for Konkani Devanagari

# Prepare datasets
train_hf = prepare_translation_hf_data(train_data)
val_hf = prepare_translation_hf_data(val_data)

train_tokenized = train_hf.map(
    lambda x: preprocess_function(x, tokenizer),
    batched=True
)
val_tokenized = val_hf.map(
    lambda x: preprocess_function(x, tokenizer),
    batched=True
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./translation_results',
    num_train_epochs=15,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    predict_with_generate=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer
)

trainer.train()

# Save
model.save_pretrained('./konkani_translation_mbart')
tokenizer.save_pretrained('./konkani_translation_mbart')

# Inference example
def translate(text, model, tokenizer):
    tokenizer.src_lang = 'en_XX'
    encoded = tokenizer(text, return_tensors='pt')
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id['hi_IN']
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Test
sample_text = "Hello, how are you?"
translation = translate(sample_text, model, tokenizer)
print(f"EN: {sample_text}")
print(f"KOK: {translation}")
```

---

## Konkani-Specific Considerations

### 1. Script Handling
Konkani is written in multiple scripts:
- **Devanagari** (most common): नमस्कार
- **Kannada**: ನಮಸ್ಕಾರ
- **Latin/Roman**: Namaskar
- **Malayalam**: നമസ്കാരം

```python
# Script detection
def detect_konkani_script(text):
    """Detect which script is used"""
    if any('\u0900' <= c <= '\u097F' for c in text):
        return 'devanagari'
    elif any('\u0C80' <= c <= '\u0CFF' for c in text):
        return 'kannada'
    elif any('\u0D00' <= c <= '\u0D7F' for c in text):
        return 'malayalam'
    else:
        return 'latin'

# Normalize to single script (optional)
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def normalize_to_devanagari(text, source_script):
    """Convert other scripts to Devanagari"""
    if source_script == 'kannada':
        return transliterate(text, sanscript.KANNADA, sanscript.DEVANAGARI)
    elif source_script == 'malayalam':
        return transliterate(text, sanscript.MALAYALAM, sanscript.DEVANAGARI)
    return text
```

### 2. Text Preprocessing

```python
import re

def preprocess_konkani_text(text):
    """Clean and normalize Konkani text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize Unicode (NFC form)
    import unicodedata
    text = unicodedata.normalize('NFC', text)
    
    # Remove special characters (keep Devanagari, Latin, punctuation)
    text = re.sub(r'[^\u0900-\u097F\u0C80-\u0CFF\u0D00-\u0D7Fa-zA-Z0-9\s.,!?।]', '', text)
    
    return text

# Apply to dataset
df['text'] = df['text'].apply(preprocess_konkani_text)
```

### 3. Tokenization Strategies

```python
# For Devanagari Konkani - use Indic tokenizers
from indicnlp.tokenize import indic_tokenize

def tokenize_konkani_devanagari(text):
    """Tokenize Konkani text in Devanagari script"""
    return indic_tokenize.trivial_tokenize(text)

# For Latin/Roman Konkani - use standard tokenizers
from nltk.tokenize import word_tokenize

def tokenize_konkani_latin(text):
    """Tokenize Konkani text in Latin script"""
    return word_tokenize(text.lower())

# Unified tokenizer
def tokenize_konkani(text):
    """Auto-detect script and tokenize"""
    script = detect_konkani_script(text)
    if script == 'devanagari':
        return tokenize_konkani_devanagari(text)
    else:
        return tokenize_konkani_latin(text)
```

### 4. Data Augmentation for Low-Resource Language

```python
# Back-translation augmentation
def augment_via_backtranslation(texts, en_to_kok_model, kok_to_en_model):
    """Augment data using back-translation"""
    augmented = []
    for text in texts:
        # Translate to English
        en_text = kok_to_en_model.translate(text)
        # Translate back to Konkani
        back_translated = en_to_kok_model.translate(en_text)
        augmented.append(back_translated)
    return augmented

# Synonym replacement (manual dictionary)
konkani_synonyms = {
    'बरो': ['चांगलो', 'उत्तम'],
    'वायट': ['खराब', 'निकामी'],
    'खूब': ['बरेच', 'फार']
}

def synonym_replacement(text, synonym_dict, n=1):
    """Replace n words with synonyms"""
    words = text.split()
    for _ in range(n):
        idx = np.random.randint(0, len(words))
        word = words[idx]
        if word in synonym_dict:
            words[idx] = np.random.choice(synonym_dict[word])
    return ' '.join(words)
```

---

## Evaluation Metrics

### Sentiment Classification
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print(classification_report(y_test, y_pred_classes, 
                          target_names=['negative', 'neutral', 'positive']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix - Konkani Sentiment')
plt.savefig('confusion_matrix.png')
```

### NER Evaluation
```python
from seqeval.metrics import classification_report as seq_classification_report

# Get predictions
predictions = trainer.predict(test_tokenized)
pred_labels = np.argmax(predictions.predictions, axis=2)

# Convert to tag names
true_labels = [[id2tag[l] for l in label if l != -100] 
               for label in test_tokenized['labels']]
pred_labels = [[id2tag[p] for (p, l) in zip(pred, label) if l != -100]
               for pred, label in zip(pred_labels, test_tokenized['labels'])]

# Report
print(seq_classification_report(true_labels, pred_labels))
```

### Translation Evaluation (BLEU Score)
```python
from sacrebleu import corpus_bleu

# Generate translations
references = [item['kok'] for item in test_data]
hypotheses = [translate(item['en'], model, tokenizer) for item in test_data]

# Calculate BLEU
bleu = corpus_bleu(hypotheses, [references])
print(f"BLEU Score: {bleu.score:.2f}")
```

---

## Model Deployment

### 1. Save Model Artifacts

```python
import pickle
import json

# Save tokenizer vocabulary
with open('tokenizer_config.pkl', 'wb') as f:
    pickle.dump({
        'tokenizer': tokenizer,
        'max_length': max_length,
        'label_map': label_map
    }, f)

# Save metadata
metadata = {
    'model_type': 'sentiment_classification',
    'language': 'konkani',
    'scripts': ['devanagari', 'latin'],
    'num_classes': 3,
    'training_samples': len(train_df),
    'test_accuracy': float(test_acc),
    'created_date': '2025-12-03'
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

### 2. Create Inference API (Flask)

```python
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model and tokenizer
model = tf.keras.models.load_model('konkani_sentiment_lstm.h5')
with open('tokenizer_config.pkl', 'rb') as f:
    config = pickle.load(f)
    tokenizer = config['tokenizer']
    max_length = config['max_length']
    label_map = config['label_map']

# Reverse label map
id2label = {v: k for k, v in label_map.items()}

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for Konkani text"""
    data = request.json
    text = data.get('text', '')
    
    # Preprocess
    text = preprocess_konkani_text(text)
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length)
    
    # Predict
    prediction = model.predict(padded)
    predicted_class = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class])
    
    return jsonify({
        'text': text,
        'sentiment': id2label[predicted_class],
        'confidence': confidence,
        'probabilities': {
            id2label[i]: float(prediction[0][i]) 
            for i in range(len(prediction[0]))
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'konkani_sentiment'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 3. Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY konkani_sentiment_lstm.h5 .
COPY tokenizer_config.pkl .
COPY model_metadata.json .
COPY app.py .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
```

```txt
# requirements.txt
tensorflow==2.13.0
flask==2.3.0
numpy==1.24.3
scikit-learn==1.3.0
```

```bash
# Build and run
docker build -t konkani-nlp-api .
docker run -p 5000:5000 konkani-nlp-api
```

---

## Data Collection Resources

### Konkani Language Sources

1. **News Websites**
   - Gomantak Times (Konkani edition)
   - Sunaparant
   - Vauraddeancho Ixtt

2. **Social Media**
   - Twitter/X hashtags: #Konkani, #कोंकणी
   - Facebook groups: Konkani language communities
   - YouTube comments on Konkani content

3. **Literature & Corpora**
   - Konkani Sahitya Akademi publications
   - Digital libraries with Konkani texts
   - Religious texts (Bible, Bhagavad Gita in Konkani)

4. **Existing Datasets**
   - AI4Bharat IndicCorp (includes Konkani)
   - FLORES-200 (multilingual parallel corpus)
   - Wikipedia dumps (Konkani Wikipedia)

### Web Scraping Example

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_konkani_news(url):
    """Scrape Konkani news articles"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    articles = []
    for article in soup.find_all('article'):
        title = article.find('h2').text.strip()
        content = article.find('div', class_='content').text.strip()
        articles.append({
            'title': title,
            'text': content,
            'source': url
        })
    
    return pd.DataFrame(articles)

# Save scraped data
df = scrape_konkani_news('https://example-konkani-news.com')
df.to_csv('konkani_news_raw.csv', index=False, encoding='utf-8')
```

---

## Best Practices Checklist

### Data Quality
- [ ] Ensure UTF-8 encoding for all files
- [ ] Verify script consistency (Devanagari/Latin/Kannada)
- [ ] Remove duplicates and near-duplicates
- [ ] Balance class distributions (for classification)
- [ ] Validate annotation quality (inter-annotator agreement)

### Data Splits
- [ ] Strict train/val/test separation (no data leakage)
- [ ] Stratified splits for classification tasks
- [ ] Random shuffling before splitting
- [ ] Document split ratios and random seeds

### Model Training
- [ ] Use multilingual pre-trained models (mBERT, XLM-R, mBART)
- [ ] Implement early stopping to prevent overfitting
- [ ] Track metrics on validation set
- [ ] Save best model checkpoints
- [ ] Log hyperparameters and results

### Evaluation
- [ ] Test on held-out test set only once
- [ ] Report multiple metrics (accuracy, F1, precision, recall)
- [ ] Analyze errors and failure cases
- [ ] Test on different scripts if applicable
- [ ] Compare with baseline models

### Deployment
- [ ] Version control for models and data
- [ ] Document model limitations and biases
- [ ] Implement input validation and error handling
- [ ] Monitor model performance in production
- [ ] Plan for model updates and retraining

---

## Installation Requirements

### Python Environment Setup

```bash
# Create virtual environment
python -m venv konkani_nlp_env
source konkani_nlp_env/bin/activate  # On Windows: konkani_nlp_env\Scripts\activate

# Install core dependencies
pip install --upgrade pip

# For TensorFlow/Keras
pip install tensorflow==2.13.0
pip install keras==2.13.1

# For PyTorch/HuggingFace
pip install torch==2.0.1
pip install transformers==4.30.0
pip install datasets==2.13.0

# Data processing
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install h5py==3.9.0

# Indic language support
pip install indic-nlp-library==0.92
pip install indic-transliteration==2.3.44

# Evaluation
pip install seqeval==1.2.2
pip install sacrebleu==2.3.1

# Visualization
pip install matplotlib==3.7.2
pip install seaborn==0.12.2

# Web scraping (optional)
pip install beautifulsoup4==4.12.2
pip install requests==2.31.0

# API deployment (optional)
pip install flask==2.3.0
```

### System Requirements

**Minimum:**
- RAM: 8GB
- Storage: 10GB free space
- CPU: 4 cores

**Recommended:**
- RAM: 16GB+
- Storage: 50GB+ SSD
- GPU: NVIDIA GPU with 8GB+ VRAM (for transformer models)
- CUDA: 11.8+ (for GPU acceleration)

---

## Quick Start Example

```python
# Complete workflow in one script
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load data
df = pd.read_csv('konkani_reviews.csv', encoding='utf-8')

# 2. Preprocess
df['text'] = df['text'].str.strip()
label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
df['label_encoded'] = df['label'].map(label_map)

# 3. Split
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

# 4. Train (using HuggingFace)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

class KonkaniDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)

train_dataset = KonkaniDataset(train_df['text'].tolist(), train_df['label_encoded'].tolist(), tokenizer)
val_dataset = KonkaniDataset(val_df['text'].tolist(), val_df['label_encoded'].tolist(), tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy='epoch',
    save_strategy='epoch'
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)
trainer.train()

# 5. Save
model.save_pretrained('./konkani_model')
tokenizer.save_pretrained('./konkani_model')

print("✓ Model trained and saved successfully!")
```

---

## Troubleshooting

### Common Issues

**Issue: Unicode encoding errors**
```python
# Solution: Always specify UTF-8 encoding
df = pd.read_csv('file.csv', encoding='utf-8')
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()
```

**Issue: Out of memory during training**
```python
# Solution: Reduce batch size
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # Reduce from 16
    gradient_accumulation_steps=2   # Simulate larger batch
)
```

**Issue: Poor performance on mixed scripts**
```python
# Solution: Normalize to single script or train separate models
df['script'] = df['text'].apply(detect_konkani_script)
devanagari_df = df[df['script'] == 'devanagari']
latin_df = df[df['script'] == 'latin']
```

**Issue: Tokenizer doesn't handle Konkani well**
```python
# Solution: Use Indic-specific tokenizers or multilingual models
from indicnlp.tokenize import indic_tokenize
tokens = indic_tokenize.trivial_tokenize(text)
```

---

## Summary

This guide provides production-ready code for building Konkani NLP systems across three core tasks:

1. **Sentiment Classification**: LSTM and BERT-based models for opinion mining
2. **Named Entity Recognition**: Token-level classification with BIO tagging
3. **Machine Translation**: Seq2seq models for English↔Konkani translation

Key takeaways:
- Always use UTF-8 encoding for Konkani text
- Handle multiple scripts (Devanagari, Latin, Kannada, Malayalam)
- Use multilingual pre-trained models (mBERT, XLM-R, mBART)
- Maintain strict train/val/test splits
- Save models in multiple formats (.h5, .pt, HuggingFace)
- Implement comprehensive evaluation before deployment

The code templates are copy-paste ready and can be adapted for other low-resource Indic languages.
