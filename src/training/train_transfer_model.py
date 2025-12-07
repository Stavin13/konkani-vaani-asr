#!/usr/bin/env python3
"""
Konkani Sentiment Analysis Model Training Script
Trains a custom transformer model on the Konkani sentiment dataset
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML/DL Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Transformers
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
import torch

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


class KonkaniSentimentTrainer:
    """Train a sentiment analysis model for Konkani language"""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        output_dir: str = "./konkani_sentiment_model",
        max_length: int = 128
    ):
        """
        Initialize the trainer
        
        Args:
            model_name: HuggingFace model to use as base
            output_dir: Directory to save the trained model
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        
        # Label mapping
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        
        # Initialize tokenizer
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self, data_path: str, format: str = "csv") -> pd.DataFrame:
        """
        Load dataset from file
        
        Args:
            data_path: Path to dataset file
            format: File format ('csv' or 'jsonl')
            
        Returns:
            DataFrame with text and label columns
        """
        print(f"\nLoading data from: {data_path}")
        
        if format == "csv":
            df = pd.read_csv(data_path)
        elif format == "jsonl":
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Ensure required columns exist
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must have 'text' and 'label' columns")
        
        # Clean data
        df = df.dropna(subset=['text', 'label'])
        df = df[df['label'].isin(['positive', 'negative', 'neutral'])]
        
        print(f"Loaded {len(df)} samples")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def prepare_datasets(
        self,
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> DatasetDict:
        """
        Prepare train/val/test splits
        
        Args:
            df: Input DataFrame
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
            
        Returns:
            DatasetDict with train/val/test splits
        """
        print("\nPreparing train/val/test splits...")
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_df['label']
        )
        
        print(f"Train samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Convert to HuggingFace Dataset format
        train_dataset = Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True))
        val_dataset = Dataset.from_pandas(val_df[['text', 'label']].reset_index(drop=True))
        test_dataset = Dataset.from_pandas(test_df[['text', 'label']].reset_index(drop=True))
        
        # Create DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        return dataset_dict
    
    def tokenize_function(self, examples):
        """Tokenize text examples"""
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
    
    def encode_labels(self, examples):
        """Convert string labels to integers"""
        examples['labels'] = [self.label2id[label] for label in examples['label']]
        return examples
    
    def prepare_model_inputs(self, dataset_dict: DatasetDict) -> DatasetDict:
        """
        Tokenize and encode the datasets
        
        Args:
            dataset_dict: DatasetDict with raw text
            
        Returns:
            DatasetDict with tokenized inputs
        """
        print("\nTokenizing datasets...")
        
        # Encode labels
        dataset_dict = dataset_dict.map(self.encode_labels, batched=True)
        
        # Tokenize
        dataset_dict = dataset_dict.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text', 'label']
        )
        
        # Set format for PyTorch
        dataset_dict.set_format('torch')
        
        return dataset_dict
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(
        self,
        dataset_dict: DatasetDict,
        num_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        save_steps: int = 500,
        eval_steps: int = 500,
        early_stopping_patience: int = 3
    ):
        """
        Train the model
        
        Args:
            dataset_dict: Prepared datasets
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            early_stopping_patience: Early stopping patience
        """
        print("\n" + "="*60)
        print("TRAINING KONKANI SENTIMENT ANALYSIS MODEL")
        print("="*60)
        
        # Load model
        print(f"\nLoading model: {self.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,
            save_total_limit=3,
            report_to="none",  # Disable wandb/tensorboard
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict['train'],
            eval_dataset=dataset_dict['validation'],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )
        
        # Train
        print("\nStarting training...")
        train_result = trainer.train()
        
        # Save final model
        print(f"\nSaving model to: {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        return trainer
    
    def evaluate(self, trainer: Trainer, dataset_dict: DatasetDict):
        """
        Evaluate model on test set
        
        Args:
            trainer: Trained Trainer object
            dataset_dict: Dataset with test split
        """
        print("\n" + "="*60)
        print("EVALUATING ON TEST SET")
        print("="*60)
        
        # Evaluate
        test_results = trainer.evaluate(dataset_dict['test'])
        
        print("\nTest Results:")
        for key, value in test_results.items():
            print(f"  {key}: {value:.4f}")
        
        # Get predictions
        predictions = trainer.predict(dataset_dict['test'])
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(
            true_labels,
            pred_labels,
            target_names=['negative', 'neutral', 'positive']
        ))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive']
        )
        plt.title('Confusion Matrix - Konkani Sentiment Analysis')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = f"{self.output_dir}/confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {cm_path}")
        
        # Save test results
        test_results_path = f"{self.output_dir}/test_results.json"
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"Test results saved to: {test_results_path}")
        
        return test_results
    
    def save_model_card(self, test_results: Dict):
        """Save model card with information"""
        model_card = f"""---
language: kon
tags:
- konkani
- sentiment-analysis
- text-classification
license: apache-2.0
---

# Konkani Sentiment Analysis Model

This model is a fine-tuned version of `{self.model_name}` for sentiment analysis in Konkani language (both Devanagari and Romanized text).

## Model Description

- **Language:** Konkani (kon)
- **Task:** Sentiment Analysis (3-class: positive, neutral, negative)
- **Base Model:** {self.model_name}
- **Training Date:** {datetime.now().strftime('%Y-%m-%d')}

## Performance

Test set metrics:
- **Accuracy:** {test_results.get('eval_accuracy', 0):.4f}
- **F1 Score:** {test_results.get('eval_f1', 0):.4f}
- **Precision:** {test_results.get('eval_precision', 0):.4f}
- **Recall:** {test_results.get('eval_recall', 0):.4f}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_name = "{self.output_dir}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Predict
text = "हें फोन खूब छान आसा"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
predicted_class = torch.argmax(predictions, dim=-1).item()

labels = ["negative", "neutral", "positive"]
print(f"Sentiment: {{labels[predicted_class]}}")
print(f"Confidence: {{predictions[0][predicted_class].item():.4f}}")
```

## Training Data

Custom Konkani sentiment dataset with ~48,000 entries including:
- Word-level sentiment annotations
- Sentence-level sentiment annotations
- Multiple Romanization variants
- Balanced label distribution

## Limitations

- Trained primarily on custom-generated data
- May not generalize well to all Konkani dialects
- Performance may vary between Devanagari and Romanized text

## Citation

If you use this model, please cite:

```
@misc{{konkani-sentiment-2024,
  author = {{Custom Konkani NLP}},
  title = {{Konkani Sentiment Analysis Model}},
  year = {{2024}},
  publisher = {{HuggingFace}},
}}
```
"""
        
        model_card_path = f"{self.output_dir}/README.md"
        with open(model_card_path, 'w', encoding='utf-8') as f:
            f.write(model_card)
        
        print(f"\nModel card saved to: {model_card_path}")


def main():
    """Main training pipeline"""
    
    # Configuration
    DATA_PATH = "custom_konkani_sentiment.csv"  # or .jsonl
    DATA_FORMAT = "csv"  # or "jsonl"
    MODEL_NAME = "distilbert-base-multilingual-cased"  # Fast and efficient
    # Alternative models:
    # - "bert-base-multilingual-cased" (more accurate but slower)
    # - "xlm-roberta-base" (best for multilingual)
    # - "google/muril-base-cased" (optimized for Indian languages)
    
    OUTPUT_DIR = "./konkani_sentiment_model"
    
    # Training hyperparameters
    NUM_EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 128
    
    # Initialize trainer
    trainer = KonkaniSentimentTrainer(
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        max_length=MAX_LENGTH
    )
    
    # Load data
    df = trainer.load_data(DATA_PATH, format=DATA_FORMAT)
    
    # Prepare datasets
    dataset_dict = trainer.prepare_datasets(df)
    
    # Prepare model inputs
    dataset_dict = trainer.prepare_model_inputs(dataset_dict)
    
    # Train model
    trained_model = trainer.train(
        dataset_dict,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # Evaluate
    test_results = trainer.evaluate(trained_model, dataset_dict)
    
    # Save model card
    trainer.save_model_card(test_results)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("\nTo use your model:")
    print(f"  from transformers import pipeline")
    print(f"  classifier = pipeline('sentiment-analysis', model='{OUTPUT_DIR}')")
    print(f"  result = classifier('हें फोन खूब छान आसा')")
    print(f"  print(result)")
    

if __name__ == "__main__":
    main()
