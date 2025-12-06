#!/usr/bin/env python3
"""
Train custom BiLSTM sentiment model for Konkani.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from konkani.core import KonkaniTokenizer, KonkaniDataset, collate_fn
from konkani.models.sentiment.bilstm import CustomKonkaniSentimentModel
from konkani.training.sentiment_trainer import SentimentTrainer
from konkani.utils.io import save_json
from config.paths import Paths
from config.model_config import BiLSTMConfig
from config.training_config import TrainingConfig


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("CUSTOM KONKANI SENTIMENT ANALYSIS MODEL")
    print("Built from Scratch with PyTorch")
    print("="*60)
    
    # Ensure directories exist
    Paths.ensure_dirs()
    
    # Configuration
    model_config = BiLSTMConfig(
        vocab_size=10000,
        embedding_dim=256,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        max_length=128
    )
    
    training_config = TrainingConfig(
        batch_size=32,
        num_epochs=50,
        learning_rate=0.001,
        patience=3
    )
    
    # Paths
    data_path = Paths.DATA_PROCESSED / "custom_konkani_sentiment_fixed.csv"
    output_dir = Paths.MODELS_SENTIMENT / "custom_konkani_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Prepare labels
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    df['label_id'] = df['label'].map(label_map)
    
    # Save label map
    print(f"\nSaving label map...")
    save_json(
        {'label2id': label_map, 'id2label': id2label},
        output_dir / "label_map.json"
    )
    print(f"✓ Label map saved")
    
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Build tokenizer
    tokenizer = KonkaniTokenizer(vocab_size=model_config.vocab_size)
    tokenizer.build_vocab(df['text'].tolist())
    tokenizer.save(str(output_dir / "tokenizer.pkl"))
    
    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['text'].tolist(), df['label_id'].tolist(),
        test_size=0.3, random_state=42, stratify=df['label_id']
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_texts)}")
    print(f"  Validation: {len(val_texts)}")
    print(f"  Test: {len(test_texts)}")
    
    # Create datasets
    train_dataset = KonkaniDataset(
        train_texts, train_labels, tokenizer, model_config.max_length
    )
    val_dataset = KonkaniDataset(
        val_texts, val_labels, tokenizer, model_config.max_length
    )
    test_dataset = KonkaniDataset(
        test_texts, test_labels, tokenizer, model_config.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model
    print("\nInitializing custom model...")
    model = CustomKonkaniSentimentModel(
        vocab_size=len(tokenizer),
        embedding_dim=model_config.embedding_dim,
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        bidirectional=model_config.bidirectional
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = SentimentTrainer(
        model,
        tokenizer,
        device=training_config.device,
        output_dir=str(output_dir)
    )
    
    # Train
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=training_config.num_epochs,
        learning_rate=training_config.learning_rate,
        patience=training_config.patience
    )
    
    # Evaluate
    results = trainer.evaluate(test_loader)
    
    # Save final model
    trainer.save_model(output_dir / "final_model.pt")
    
    # Save metadata
    trainer.save_metadata(total_params, trainable_params, results)
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
    print(f"\nModel saved to: {output_dir}")
    print("\nTo use your model:")
    print("  python scripts/sentiment/predict.py")


if __name__ == "__main__":
    main()
