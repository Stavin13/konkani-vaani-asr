#!/usr/bin/env python3
"""
Custom Konkani Sentiment Analysis Model - Built from Scratch
Train a completely custom neural network for Konkani sentiment analysis
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import pickle
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ML utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class KonkaniTokenizer:
    """Custom tokenizer for Konkani text (both Devanagari and Roman)"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.word_freq = {}
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        print("Building vocabulary...")
        
        # Count word frequencies
        for text in tqdm(texts):
            words = self._tokenize(text)
            for word in words:
                self.word_freq[word] = self.word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top vocab_size
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Add to vocabulary
        for word, freq in sorted_words[:self.vocab_size - 4]:  # -4 for special tokens
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Most common words: {sorted_words[:10]}")
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (can be enhanced)"""
        # Basic tokenization - split on whitespace and punctuation
        import re
        # Keep Devanagari and Roman characters
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens
    
    def encode(self, text: str, max_length: int = 128) -> List[int]:
        """Convert text to token indices"""
        words = self._tokenize(text)
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        # Truncate or pad
        if len(indices) > max_length:
            indices = indices[:max_length]
        
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Convert indices back to text"""
        words = [self.idx2word.get(idx, '<UNK>') for idx in indices]
        return ' '.join(words)
    
    def save(self, path: str):
        """Save tokenizer"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_freq': self.word_freq,
                'vocab_size': self.vocab_size
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """Load tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.word2idx = data['word2idx']
        tokenizer.idx2word = data['idx2word']
        tokenizer.word_freq = data['word_freq']
        return tokenizer


class KonkaniDataset(Dataset):
    """Custom Dataset for Konkani sentiment data"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: KonkaniTokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        encoded = self.tokenizer.encode(text, self.max_length)
        
        return {
            'input_ids': torch.tensor(encoded, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'length': len(encoded)
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    input_ids = [item['input_ids'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    lengths = [item['length'] for item in batch]
    
    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    return {
        'input_ids': input_ids_padded,
        'labels': labels,
        'lengths': lengths
    }


class CustomKonkaniSentimentModel(nn.Module):
    """
    Custom Neural Network for Konkani Sentiment Analysis
    Architecture: Embedding → BiLSTM → Attention → Dense → Output
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(CustomKonkaniSentimentModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Linear(lstm_output_dim, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def attention_layer(self, lstm_output, lengths):
        """Apply attention mechanism"""
        # Calculate attention weights
        attention_weights = torch.tanh(self.attention(lstm_output))
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended = torch.sum(attention_weights * lstm_output, dim=1)
        
        return attended, attention_weights
    
    def forward(self, input_ids, lengths=None):
        # Embedding
        embedded = self.embedding(input_ids)  # [batch, seq_len, embedding_dim]
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)  # [batch, seq_len, hidden_dim*2]
        
        # Attention
        attended, attention_weights = self.attention_layer(lstm_out, lengths)
        
        # Fully connected layers
        x = self.relu(self.fc1(attended))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits, attention_weights


class CustomModelTrainer:
    """Trainer for custom Konkani sentiment model"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: KonkaniTokenizer,
        device: str = 'auto',
        output_dir: str = './custom_konkani_model'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        patience: int = 3
    ):
        """Train the model"""
        print("\n" + "="*60)
        print("TRAINING CUSTOM KONKANI SENTIMENT MODEL")
        print("="*60)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping (DISABLED - will train for full epochs)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best model
                self.save_model(f"{self.output_dir}/best_model.pt")
                print("✓ Saved best model")
            
            # Early stopping disabled - training for full NUM_EPOCHS
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     patience_counter = 0
            # else:
            #     patience_counter += 1
            #     if patience_counter >= patience:
            #         print(f"\nEarly stopping triggered after {epoch+1} epochs")
            #         break
        
        # Load best model
        self.load_model(f"{self.output_dir}/best_model.pt")
        
        # Plot training history
        self._plot_history()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
    
    def _train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch['lengths']
            
            # Forward pass
            optimizer.zero_grad()
            logits, _ = self.model(input_ids, lengths)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        return total_loss / len(train_loader), correct / total
    
    def _validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths']
                
                # Forward pass
                logits, _ = self.model(input_ids, lengths)
                loss = criterion(logits, labels)
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        return total_loss / len(val_loader), correct / total
    
    def evaluate(self, test_loader: DataLoader):
        """Evaluate model on test set"""
        print("\n" + "="*60)
        print("EVALUATING ON TEST SET")
        print("="*60)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths']
                
                logits, _ = self.model(input_ids, lengths)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        print(f"\nTest Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(
            all_labels, all_predictions,
            target_names=['negative', 'neutral', 'positive']
        ))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        self._plot_confusion_matrix(cm)
        
        # Save results
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
        
        with open(f"{self.output_dir}/test_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _plot_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_history.png", dpi=300, bbox_inches='tight')
        print(f"\nTraining history saved to: {self.output_dir}/training_history.png")
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive']
        )
        plt.title('Confusion Matrix - Custom Konkani Sentiment Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {self.output_dir}/confusion_matrix.png")
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'vocab_size': self.model.embedding.num_embeddings,
                'embedding_dim': self.model.embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'bidirectional': self.model.bidirectional
            }
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("CUSTOM KONKANI SENTIMENT ANALYSIS MODEL")
    print("Built from Scratch with PyTorch")
    print("="*60)
    
    # Configuration
    DATA_PATH = "../../data/processed/custom_konkani_sentiment_fixed.csv"
    OUTPUT_DIR = "../../models/custom_konkani_model"
    
    # Hyperparameters
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    MAX_LENGTH = 128
    
    # Load data
    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Prepare labels
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    df['label_id'] = df['label'].map(label_map)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save label map
    print(f"\nSaving label map...")
    with open(f"{OUTPUT_DIR}/label_map.json", 'w') as f:
        json.dump({'label2id': label_map, 'id2label': id2label}, f, indent=2)
    print(f"✓ Label map saved")
    
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Build tokenizer
    tokenizer = KonkaniTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.build_vocab(df['text'].tolist())
    tokenizer.save(f"{OUTPUT_DIR}/tokenizer.pkl")
    
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
    train_dataset = KonkaniDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = KonkaniDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = KonkaniDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    print("\nInitializing custom model...")
    model = CustomKonkaniSentimentModel(
        vocab_size=len(tokenizer.word2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = CustomModelTrainer(model, tokenizer, output_dir=OUTPUT_DIR)
    
    # Train
    trainer.train(
        train_loader, val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=3
    )
    
    # Evaluate
    results = trainer.evaluate(test_loader)
    
    # Save final model
    trainer.save_model(f"{OUTPUT_DIR}/final_model.pt")
    
    # Save comprehensive metadata
    import hashlib
    
    def generate_checksum(filepath):
        """Generate SHA256 checksum for file"""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    print("\nGenerating model metadata and checksums...")
    
    # Model info (backward compatible)
    model_info = {
        'architecture': 'BiLSTM with Attention',
        'vocab_size': len(tokenizer.word2idx),
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'test_accuracy': results['accuracy'],
        'test_f1': results['f1'],
        'created': datetime.now().isoformat()
    }
    
    with open(f"{OUTPUT_DIR}/model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Comprehensive metadata
    metadata = {
        'model_name': 'konkani-sentiment-bilstm-v1',
        'version': '1.0.0',
        'created': datetime.now().isoformat(),
        'framework': 'PyTorch',
        'architecture': {
            'type': 'BiLSTM with Attention',
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'bidirectional': True,
            'dropout': DROPOUT
        },
        'dataset': {
            'name': 'Custom Konkani Sentiment Dataset',
            'total_size': len(df),
            'train_size': len(train_texts),
            'val_size': len(val_texts),
            'test_size': len(test_texts),
            'vocab_size': len(tokenizer.word2idx),
            'label_distribution': df['label'].value_counts().to_dict()
        },
        'hyperparameters': {
            'vocab_size': VOCAB_SIZE,
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'max_length': MAX_LENGTH
        },
        'performance': {
            'test_accuracy': float(results['accuracy']),
            'test_f1': float(results['f1']),
            'test_precision': float(results['precision']),
            'test_recall': float(results['recall'])
        },
        'model_parameters': {
            'total': total_params,
            'trainable': trainable_params
        },
        'files': {
            'model': 'best_model.pt',
            'tokenizer': 'tokenizer.pkl',
            'label_map': 'label_map.json',
            'config': 'model_info.json',
            'metadata': 'metadata.json',
            'checksums': 'checksums.json'
        }
    }
    
    with open(f"{OUTPUT_DIR}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Generate checksums
    checksums = {
        'best_model.pt': generate_checksum(f"{OUTPUT_DIR}/best_model.pt"),
        'final_model.pt': generate_checksum(f"{OUTPUT_DIR}/final_model.pt"),
        'tokenizer.pkl': generate_checksum(f"{OUTPUT_DIR}/tokenizer.pkl"),
        'label_map.json': generate_checksum(f"{OUTPUT_DIR}/label_map.json"),
        'metadata.json': generate_checksum(f"{OUTPUT_DIR}/metadata.json")
    }
    
    with open(f"{OUTPUT_DIR}/checksums.json", 'w') as f:
        json.dump(checksums, f, indent=2)
    
    print("✓ Metadata and checksums generated")
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("\nTo use your model:")
    print("  python test_custom_model.py")


if __name__ == "__main__":
    main()
