"""
Train Custom Konkani NER Model
Using auto-labeled data from pre-trained model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np

from models.konkani_ner import create_ner_model


class NERDataset(Dataset):
    """Dataset for NER training"""
    
    def __init__(self, data_file, word2id=None, char2id=None):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Build vocabularies if not provided
        if word2id is None:
            self.word2id = self._build_word_vocab()
        else:
            self.word2id = word2id
        
        if char2id is None:
            self.char2id = self._build_char_vocab()
        else:
            self.char2id = char2id
        
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2char = {v: k for k, v in self.char2id.items()}
    
    def _build_word_vocab(self):
        """Build word vocabulary"""
        word2id = {'<PAD>': 0, '<UNK>': 1}
        for item in self.data:
            for token in item['tokens']:
                if token not in word2id:
                    word2id[token] = len(word2id)
        return word2id
    
    def _build_char_vocab(self):
        """Build character vocabulary"""
        char2id = {'<PAD>': 0, '<UNK>': 1}
        for item in self.data:
            for token in item['tokens']:
                for char in token:
                    if char not in char2id:
                        char2id[char] = len(char2id)
        return char2id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Convert tokens to IDs
        word_ids = [self.word2id.get(token, 1) for token in item['tokens']]
        
        # Convert characters to IDs (for each word)
        char_ids = []
        for token in item['tokens']:
            char_id_list = [self.char2id.get(char, 1) for char in token]
            char_ids.append(char_id_list)
        
        # Get label IDs
        label_ids = item['label_ids']
        
        return {
            'word_ids': torch.tensor(word_ids, dtype=torch.long),
            'char_ids': char_ids,  # Will be padded in collate_fn
            'label_ids': torch.tensor(label_ids, dtype=torch.long),
            'text': item['text']
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    word_ids = [item['word_ids'] for item in batch]
    label_ids = [item['label_ids'] for item in batch]
    char_ids = [item['char_ids'] for item in batch]
    
    # Pad word sequences
    word_ids_padded = pad_sequence(word_ids, batch_first=True, padding_value=0)
    label_ids_padded = pad_sequence(label_ids, batch_first=True, padding_value=0)
    
    # Pad character sequences
    max_seq_len = word_ids_padded.size(1)
    max_char_len = max(max(len(chars) for chars in item) for item in char_ids)
    
    char_ids_padded = torch.zeros(len(batch), max_seq_len, max_char_len, dtype=torch.long)
    for i, item_chars in enumerate(char_ids):
        for j, word_chars in enumerate(item_chars):
            char_ids_padded[i, j, :len(word_chars)] = torch.tensor(word_chars, dtype=torch.long)
    
    # Create mask (1 for real tokens, 0 for padding)
    mask = (word_ids_padded != 0).long()
    
    return {
        'word_ids': word_ids_padded,
        'char_ids': char_ids_padded,
        'label_ids': label_ids_padded,
        'mask': mask
    }


class NERTrainer:
    """Trainer for NER model"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_f1 = 0.0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            word_ids = batch['word_ids'].to(self.device)
            char_ids = batch['char_ids'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            loss = self.model(word_ids, char_ids, label_ids, mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, epoch):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                word_ids = batch['word_ids'].to(self.device)
                char_ids = batch['char_ids'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # Compute loss
                loss = self.model(word_ids, char_ids, label_ids, mask)
                total_loss += loss.item()
                
                # Get predictions
                predictions = self.model(word_ids, char_ids, mask=mask)
                
                # Collect for metrics
                for i in range(len(predictions)):
                    seq_len = mask[i].sum().item()
                    pred = predictions[i][:seq_len] if isinstance(predictions, list) else predictions[i][:seq_len].tolist()
                    true = label_ids[i][:seq_len].tolist()
                    
                    all_predictions.extend(pred)
                    all_labels.extend(true)
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Compute F1 score
        f1 = self.compute_f1(all_predictions, all_labels)
        
        return avg_loss, f1
    
    def compute_f1(self, predictions, labels):
        """Compute token-level F1 score"""
        # Filter out padding (label 0)
        valid_indices = [i for i, label in enumerate(labels) if label != 0]
        predictions = [predictions[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        if len(labels) == 0:
            return 0.0
        
        # Simple accuracy for now (can be improved with seqeval)
        correct = sum(p == l for p, l in zip(predictions, labels))
        accuracy = correct / len(labels)
        
        return accuracy
    
    def save_checkpoint(self, epoch, val_f1, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_f1': val_f1,
            'config': self.config
        }
        
        # Save latest
        checkpoint_path = self.checkpoint_dir / f'ner_checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best_ner_model.pt'
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model with F1: {val_f1:.4f}")
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"\n{'='*70}")
        print("STARTING KONKANI NER TRAINING")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"{'='*70}\n")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            val_loss, val_f1 = self.evaluate(epoch)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val F1: {val_f1:.4f}")
            
            # Save checkpoint
            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, val_f1, is_best)
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"Best F1 score: {self.best_val_f1:.4f}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Konkani NER')
    parser.add_argument('--data_file', type=str, 
                        default='data/ner_labeled_data.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/ner')
    parser.add_argument('--use_crf', action='store_true', help='Use CRF layer')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load data
    print(f"\n📂 Loading data from: {args.data_file}")
    dataset = NERDataset(args.data_file)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Val samples: {len(val_dataset)}")
    
    # Create model
    vocab_size = len(dataset.word2id)
    char_vocab_size = len(dataset.char2id)
    num_tags = 9  # B/I for PER/ORG/LOC/MISC + O
    
    model = create_ner_model(
        vocab_size=vocab_size,
        char_vocab_size=char_vocab_size,
        num_tags=num_tags,
        use_crf=args.use_crf,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Save vocabularies
    vocab_file = Path(args.checkpoint_dir) / 'vocabularies.json'
    vocab_file.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump({
            'word2id': dataset.word2id,
            'char2id': dataset.char2id
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved vocabularies to: {vocab_file}")
    
    # Training config
    config = {
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'checkpoint_dir': args.checkpoint_dir
    }
    
    # Create trainer
    trainer = NERTrainer(model, train_loader, val_loader, device, config)
    
    # Train
    trainer.train(args.num_epochs)


if __name__ == "__main__":
    main()
