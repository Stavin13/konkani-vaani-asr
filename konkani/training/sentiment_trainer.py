"""
Sentiment model trainer for Konkani.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from datetime import datetime

from ..core.metrics import calculate_metrics, get_confusion_matrix
from ..utils.visualization import plot_training_history, plot_confusion_matrix
from ..utils.io import save_json, generate_checksums


class SentimentTrainer:
    """Trainer for Konkani sentiment models"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = 'auto',
        output_dir: str = './models/sentiment/custom'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
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
        print("TRAINING KONKANI SENTIMENT MODEL")
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
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(self.output_dir / "best_model.pt")
                print("✓ Saved best model")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model
        self.load_model(self.output_dir / "best_model.pt")
        
        # Plot training history
        plot_training_history(
            self.history,
            save_path=self.output_dir / "training_history.png"
        )
        
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
        metrics = calculate_metrics(all_labels, all_predictions)
        
        print(f"\nTest Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        
        # Confusion matrix
        cm = get_confusion_matrix(all_labels, all_predictions)
        plot_confusion_matrix(
            cm,
            save_path=self.output_dir / "confusion_matrix.png",
            title="Confusion Matrix - Konkani Sentiment Model"
        )
        
        # Save results
        save_json(metrics, self.output_dir / "test_results.json")
        
        return metrics
    
    def save_model(self, path: Path):
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
    
    def load_model(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def save_metadata(self, total_params: int, trainable_params: int, results: Dict):
        """Save comprehensive model metadata"""
        metadata = {
            'model_name': 'konkani-sentiment-bilstm',
            'version': '1.0.0',
            'created': datetime.now().isoformat(),
            'framework': 'PyTorch',
            'architecture': {
                'type': 'BiLSTM with Attention',
                'embedding_dim': self.model.embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'bidirectional': self.model.bidirectional,
            },
            'model_parameters': {
                'total': total_params,
                'trainable': trainable_params
            },
            'performance': results,
            'files': {
                'model': 'best_model.pt',
                'tokenizer': 'tokenizer.pkl',
                'label_map': 'label_map.json',
                'metadata': 'metadata.json',
                'checksums': 'checksums.json'
            }
        }
        
        save_json(metadata, self.output_dir / "metadata.json")
        
        # Generate checksums
        checksums = generate_checksums(
            self.output_dir,
            patterns=['*.pt', '*.pkl', '*.json']
        )
        save_json(checksums, self.output_dir / "checksums.json")
        
        print("✓ Metadata and checksums generated")
