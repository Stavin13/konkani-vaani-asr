#!/usr/bin/env python3
"""
Create Complete Kaggle Notebook for Translation & Emotion Training
Generates data using pre-trained models, then trains custom models
"""

import json

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Title
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 Complete Konkani Translation & Emotion Training\\n",
            "## All-in-One: Generate Data → Train Models\\n",
            "\\n",
            "**What this notebook does:**\\n",
            "1. ✅ Generate translation data using Google Translate API\\n",
            "2. ✅ Generate emotion data using pre-trained models\\n",
            "3. ✅ Train custom Konkani→English translation model\\n",
            "4. ✅ Train custom Konkani emotion detection model\\n",
            "\\n",
            "**Time:** ~4-5 hours total\\n",
            "**GPU:** P100 or T4\\n",
            "\\n",
            "**Requirements:**\\n",
            "- Upload your Konkani text corpus as a dataset\\n",
            "- Or use the sample data generation below"
        ]
    })
    
    # Part 1: Setup
    add_setup_cells(notebook)
    
    # Part 2: Generate Translation Data
    add_translation_generation_cells(notebook)
    
    # Part 3: Generate Emotion Data
    add_emotion_generation_cells(notebook)
    
    # Part 4: Train Translation Model
    add_translation_training_cells(notebook)
    
    # Part 5: Train Emotion Model
    add_emotion_training_cells(notebook)
    
    # Part 6: Test & Export
    add_testing_cells(notebook)
    
    return notebook

def add_setup_cells(notebook):
    """Add setup and dependency installation cells"""
    
    notebook["cells"].extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["---\\n", "# 📋 PART 1: Setup & Check GPU"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!nvidia-smi\\n",
                "\\n",
                "import torch\\n",
                "print(f\"\\\\n{'='*60}\")\\n",
                "print(f\"GPU: {torch.cuda.get_device_name(0)}\")\\n",
                "print(f\"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")\\n",
                "print(f\"{'='*60}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Install Dependencies"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q transformers datasets sacrebleu accelerate sentencepiece googletrans==4.0.0-rc1\\n",
                "print(\"✅ All dependencies installed!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\\n",
                "import json\\n",
                "import pandas as pd\\n",
                "import numpy as np\\n",
                "from pathlib import Path\\n",
                "from tqdm.auto import tqdm\\n",
                "from googletrans import Translator\\n",
                "from transformers import (\\n",
                "    MarianMTModel, MarianTokenizer,\\n",
                "    AutoModelForSequenceClassification, AutoTokenizer,\\n",
                "    Trainer, TrainingArguments, DataCollatorWithPadding\\n",
                ")\\n",
                "from datasets import Dataset, DatasetDict\\n",
                "import warnings\\n",
                "warnings.filterwarnings('ignore')\\n",
                "\\n",
                "print(\"✅ Imports successful!\")"
            ]
        }
    ])

def add_translation_generation_cells(notebook):
    """Add cells for generating translation data"""
    
    notebook["cells"].extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\\n",
                "# 📝 PART 2: Generate Translation Data\\n",
                "\\n",
                "We'll use Google Translate to create Konkani-English pairs from your Konkani text"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Sample Konkani sentences (replace with your corpus)\\n",
                "konkani_sentences = [\\n",
                "    \\\"हांव घरा वता\\\",\\n",
                "    \\\"तुजें नांव कितें?\\\",\\n",
                "    \\\"हांव खूश आसा\\\",\\n",
                "    \\\"आयज सकाळ बरी आसा\\\",\\n",
                "    \\\"तुका कसें आसा?\\\",\\n",
                "    # Add more sentences here\\n",
                "]\\n",
                "\\n",
                "print(f\"📊 Sample sentences: {len(konkani_sentences)}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate translations using Google Translate\\n",
                "translator = Translator()\\n",
                "translation_pairs = []\\n",
                "\\n",
                "print(\"🔄 Generating translations...\")\\n",
                "for konkani_text in tqdm(konkani_sentences):\\n",
                "    try:\\n",
                "        result = translator.translate(konkani_text, src='hi', dest='en')\\n",
                "        translation_pairs.append({\\n",
                "            'konkani': konkani_text,\\n",
                "            'english': result.text\\n",
                "        })\\n",
                "    except Exception as e:\\n",
                "        print(f\"Error translating: {konkani_text}\")\\n",
                "        continue\\n",
                "\\n",
                "print(f\"\\\\n✅ Generated {len(translation_pairs)} translation pairs\")\\n",
                "print(f\"\\\\nSample pairs:\")\\n",
                "for pair in translation_pairs[:3]:\\n",
                "    print(f\"  KOK: {pair['konkani']}\")\\n",
                "    print(f\"  ENG: {pair['english']}\")\\n",
                "    print()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Split into train/val/test\\n",
                "from sklearn.model_selection import train_test_split\\n",
                "\\n",
                "train_val, test = train_test_split(translation_pairs, test_size=0.1, random_state=42)\\n",
                "train, val = train_test_split(train_val, test_size=0.1, random_state=42)\\n",
                "\\n",
                "print(f\"📊 Translation Data Split:\")\\n",
                "print(f\"  Train: {len(train)}\")\\n",
                "print(f\"  Val: {len(val)}\")\\n",
                "print(f\"  Test: {len(test)}\")\\n",
                "\\n",
                "# Save\\n",
                "os.makedirs('data/translation', exist_ok=True)\\n",
                "with open('data/translation/train.json', 'w', encoding='utf-8') as f:\\n",
                "    json.dump(train, f, ensure_ascii=False, indent=2)\\n",
                "with open('data/translation/val.json', 'w', encoding='utf-8') as f:\\n",
                "    json.dump(val, f, ensure_ascii=False, indent=2)\\n",
                "with open('data/translation/test.json', 'w', encoding='utf-8') as f:\\n",
                "    json.dump(test, f, ensure_ascii=False, indent=2)\\n",
                "\\n",
                "print(\"\\\\n✅ Translation data saved!\")"
            ]
        }
    ])

def add_emotion_generation_cells(notebook):
    """Add cells for generating emotion data"""
    
    notebook["cells"].extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\\n",
                "# 😊 PART 3: Generate Emotion Data\\n",
                "\\n",
                "We'll use a pre-trained emotion model to label Konkani text"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load pre-trained emotion classifier\\n",
                "emotion_model_name = \\\"j-hartmann/emotion-english-distilroberta-base\\\"\\n",
                "emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)\\n",
                "emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)\\n",
                "emotion_model.to('cuda')\\n",
                "\\n",
                "emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']\\n",
                "print(f\"✅ Loaded emotion model with {len(emotion_labels)} emotions\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate emotion labels for Konkani text\\n",
                "emotion_data = []\\n",
                "\\n",
                "print(\"🔄 Generating emotion labels...\")\\n",
                "for pair in tqdm(translation_pairs):\\n",
                "    # Use English translation for emotion detection\\n",
                "    inputs = emotion_tokenizer(pair['english'], return_tensors='pt', truncation=True, max_length=512)\\n",
                "    inputs = {k: v.to('cuda') for k, v in inputs.items()}\\n",
                "    \\n",
                "    with torch.no_grad():\\n",
                "        outputs = emotion_model(**inputs)\\n",
                "        prediction = torch.argmax(outputs.logits, dim=-1).item()\\n",
                "    \\n",
                "    emotion_data.append({\\n",
                "        'text': pair['konkani'],\\n",
                "        'emotion': emotion_labels[prediction]\\n",
                "    })\\n",
                "\\n",
                "print(f\"\\\\n✅ Generated {len(emotion_data)} emotion labels\")\\n",
                "\\n",
                "# Show distribution\\n",
                "emotion_counts = pd.Series([d['emotion'] for d in emotion_data]).value_counts()\\n",
                "print(f\"\\\\n📊 Emotion Distribution:\")\\n",
                "print(emotion_counts)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Split and save emotion data\\n",
                "emotion_df = pd.DataFrame(emotion_data)\\n",
                "train_df, test_df = train_test_split(emotion_df, test_size=0.1, random_state=42, stratify=emotion_df['emotion'])\\n",
                "train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['emotion'])\\n",
                "\\n",
                "print(f\"📊 Emotion Data Split:\")\\n",
                "print(f\"  Train: {len(train_df)}\")\\n",
                "print(f\"  Val: {len(val_df)}\")\\n",
                "print(f\"  Test: {len(test_df)}\")\\n",
                "\\n",
                "# Save\\n",
                "os.makedirs('data/emotion', exist_ok=True)\\n",
                "train_df.to_csv('data/emotion/train.csv', index=False)\\n",
                "val_df.to_csv('data/emotion/val.csv', index=False)\\n",
                "test_df.to_csv('data/emotion/test.csv', index=False)\\n",
                "\\n",
                "print(\"\\\\n✅ Emotion data saved!\")"
            ]
        }
    ])

def add_translation_training_cells(notebook):
    """Add cells for training translation model"""
    
    notebook["cells"].extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\\n",
                "# 🎯 PART 4: Train Translation Model\\n",
                "\\n",
                "Fine-tune MarianMT for Konkani→English translation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load base translation model\\n",
                "model_name = \\\"Helsinki-NLP/opus-mt-hi-en\\\"  # Hindi-English as base\\n",
                "translation_tokenizer = MarianTokenizer.from_pretrained(model_name)\\n",
                "translation_model = MarianMTModel.from_pretrained(model_name)\\n",
                "\\n",
                "print(f\"✅ Loaded base translation model: {model_name}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Prepare datasets\\n",
                "def preprocess_translation(examples):\\n",
                "    inputs = [ex['konkani'] for ex in examples]\\n",
                "    targets = [ex['english'] for ex in examples]\\n",
                "    \\n",
                "    model_inputs = translation_tokenizer(inputs, max_length=128, truncation=True, padding='max_length')\\n",
                "    labels = translation_tokenizer(targets, max_length=128, truncation=True, padding='max_length')\\n",
                "    \\n",
                "    model_inputs['labels'] = labels['input_ids']\\n",
                "    return model_inputs\\n",
                "\\n",
                "# Load data\\n",
                "with open('data/translation/train.json') as f:\\n",
                "    train_data = json.load(f)\\n",
                "with open('data/translation/val.json') as f:\\n",
                "    val_data = json.load(f)\\n",
                "\\n",
                "train_dataset = Dataset.from_list(train_data)\\n",
                "val_dataset = Dataset.from_list(val_data)\\n",
                "\\n",
                "# Preprocess\\n",
                "train_dataset = train_dataset.map(lambda x: preprocess_translation([x]), batched=False, remove_columns=train_dataset.column_names)\\n",
                "val_dataset = val_dataset.map(lambda x: preprocess_translation([x]), batched=False, remove_columns=val_dataset.column_names)\\n",
                "\\n",
                "print(f\"✅ Prepared {len(train_dataset)} training samples\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Training arguments\\n",
                "training_args = TrainingArguments(\\n",
                "    output_dir='./translation_model',\\n",
                "    num_train_epochs=10,\\n",
                "    per_device_train_batch_size=8,\\n",
                "    per_device_eval_batch_size=8,\\n",
                "    warmup_steps=500,\\n",
                "    weight_decay=0.01,\\n",
                "    logging_dir='./logs',\\n",
                "    logging_steps=100,\\n",
                "    evaluation_strategy='epoch',\\n",
                "    save_strategy='epoch',\\n",
                "    load_best_model_at_end=True,\\n",
                "    fp16=True,\\n",
                ")\\n",
                "\\n",
                "# Trainer\\n",
                "trainer = Trainer(\\n",
                "    model=translation_model,\\n",
                "    args=training_args,\\n",
                "    train_dataset=train_dataset,\\n",
                "    eval_dataset=val_dataset,\\n",
                ")\\n",
                "\\n",
                "print(\"✅ Trainer initialized\")\\n",
                "print(\"\\\\n🚀 Starting translation model training...\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train!\\n",
                "trainer.train()\\n",
                "\\n",
                "print(\"\\\\n✅ Translation model training complete!\")\\n",
                "\\n",
                "# Save\\n",
                "trainer.save_model('./konkani_english_translator')\\n",
                "translation_tokenizer.save_pretrained('./konkani_english_translator')\\n",
                "print(\"✅ Model saved to ./konkani_english_translator\")"
            ]
        }
    ])

def add_emotion_training_cells(notebook):
    """Add cells for training emotion model"""
    
    notebook["cells"].extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\\n",
                "# 😊 PART 5: Train Emotion Model\\n",
                "\\n",
                "Fine-tune DistilBERT for Konkani emotion detection"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load base model\\n",
                "emotion_base_model = \\\"distilbert-base-multilingual-cased\\\"\\n",
                "emotion_tokenizer_train = AutoTokenizer.from_pretrained(emotion_base_model)\\n",
                "emotion_model_train = AutoModelForSequenceClassification.from_pretrained(\\n",
                "    emotion_base_model,\\n",
                "    num_labels=len(emotion_labels)\\n",
                ")\\n",
                "\\n",
                "print(f\"✅ Loaded base emotion model: {emotion_base_model}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Prepare emotion datasets\\n",
                "train_df = pd.read_csv('data/emotion/train.csv')\\n",
                "val_df = pd.read_csv('data/emotion/val.csv')\\n",
                "\\n",
                "# Create label mapping\\n",
                "label2id = {label: i for i, label in enumerate(emotion_labels)}\\n",
                "id2label = {i: label for label, i in label2id.items()}\\n",
                "\\n",
                "train_df['label'] = train_df['emotion'].map(label2id)\\n",
                "val_df['label'] = val_df['emotion'].map(label2id)\\n",
                "\\n",
                "train_emotion_dataset = Dataset.from_pandas(train_df[['text', 'label']])\\n",
                "val_emotion_dataset = Dataset.from_pandas(val_df[['text', 'label']])\\n",
                "\\n",
                "# Tokenize\\n",
                "def tokenize_emotion(examples):\\n",
                "    return emotion_tokenizer_train(examples['text'], truncation=True, padding='max_length', max_length=128)\\n",
                "\\n",
                "train_emotion_dataset = train_emotion_dataset.map(tokenize_emotion, batched=True)\\n",
                "val_emotion_dataset = val_emotion_dataset.map(tokenize_emotion, batched=True)\\n",
                "\\n",
                "print(f\"✅ Prepared {len(train_emotion_dataset)} emotion training samples\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Training arguments\\n",
                "emotion_training_args = TrainingArguments(\\n",
                "    output_dir='./emotion_model',\\n",
                "    num_train_epochs=5,\\n",
                "    per_device_train_batch_size=16,\\n",
                "    per_device_eval_batch_size=16,\\n",
                "    warmup_steps=200,\\n",
                "    weight_decay=0.01,\\n",
                "    logging_dir='./logs',\\n",
                "    logging_steps=50,\\n",
                "    evaluation_strategy='epoch',\\n",
                "    save_strategy='epoch',\\n",
                "    load_best_model_at_end=True,\\n",
                "    fp16=True,\\n",
                ")\\n",
                "\\n",
                "# Trainer\\n",
                "emotion_trainer = Trainer(\\n",
                "    model=emotion_model_train,\\n",
                "    args=emotion_training_args,\\n",
                "    train_dataset=train_emotion_dataset,\\n",
                "    eval_dataset=val_emotion_dataset,\\n",
                "    data_collator=DataCollatorWithPadding(emotion_tokenizer_train),\\n",
                ")\\n",
                "\\n",
                "print(\"✅ Emotion trainer initialized\")\\n",
                "print(\"\\\\n🚀 Starting emotion model training...\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train!\\n",
                "emotion_trainer.train()\\n",
                "\\n",
                "print(\"\\\\n✅ Emotion model training complete!\")\\n",
                "\\n",
                "# Save\\n",
                "emotion_trainer.save_model('./konkani_emotion_classifier')\\n",
                "emotion_tokenizer_train.save_pretrained('./konkani_emotion_classifier')\\n",
                "\\n",
                "# Save label mapping\\n",
                "with open('./konkani_emotion_classifier/label_mapping.json', 'w') as f:\\n",
                "    json.dump({'label2id': label2id, 'id2label': id2label}, f, indent=2)\\n",
                "\\n",
                "print(\"✅ Model saved to ./konkani_emotion_classifier\")"
            ]
        }
    ])

def add_testing_cells(notebook):
    """Add cells for testing and exporting models"""
    
    notebook["cells"].extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\\n",
                "# 🧪 PART 6: Test Models & Export"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test translation\\n",
                "test_konkani = \\\"हांव खूश आसा\\\"\\n",
                "\\n",
                "inputs = translation_tokenizer(test_konkani, return_tensors='pt')\\n",
                "outputs = translation_model.generate(**inputs)\\n",
                "translation = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)\\n",
                "\\n",
                "print(f\"🔄 Translation Test:\")\\n",
                "print(f\"  Input: {test_konkani}\")\\n",
                "print(f\"  Output: {translation}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test emotion\\n",
                "inputs = emotion_tokenizer_train(test_konkani, return_tensors='pt')\\n",
                "outputs = emotion_model_train(**inputs)\\n",
                "prediction = torch.argmax(outputs.logits, dim=-1).item()\\n",
                "emotion = id2label[prediction]\\n",
                "\\n",
                "print(f\"\\\\n😊 Emotion Test:\")\\n",
                "print(f\"  Input: {test_konkani}\")\\n",
                "print(f\"  Emotion: {emotion}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# List all outputs\\n",
                "print(\"\\\\n\" + \"=\"*60)\\n",
                "print(\"✅ TRAINING COMPLETE!\")\\n",
                "print(\"=\"*60)\\n",
                "print(\"\\\\n📦 Trained Models:\")\\n",
                "print(\"  1. ./konkani_english_translator/\")\\n",
                "print(\"  2. ./konkani_emotion_classifier/\")\\n",
                "print(\"\\\\n📊 Generated Data:\")\\n",
                "print(\"  1. data/translation/ (train/val/test.json)\")\\n",
                "print(\"  2. data/emotion/ (train/val/test.csv)\")\\n",
                "print(\"\\\\n💾 Download these from the Output tab!\")\\n",
                "\\n",
                "!ls -lh konkani_english_translator/\\n",
                "!ls -lh konkani_emotion_classifier/"
            ]
        }
    ])

def main():
    print("🚀 Creating complete Kaggle notebook...")
    notebook = create_notebook()
    
    output_path = "notebooks/KAGGLE_COMPLETE_TRANSLATION_EMOTION.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created: {output_path}")
    print("\n📋 Next steps:")
    print("  1. Upload this notebook to Kaggle")
    print("  2. Add your Konkani text corpus (or use sample data)")
    print("  3. Enable GPU (P100 or T4)")
    print("  4. Run all cells")
    print("  5. Download trained models from Output tab")

if __name__ == "__main__":
    main()
