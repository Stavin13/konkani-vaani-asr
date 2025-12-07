"""
Train Custom Konkani → English Translation Model
Using auto-translated data from pre-trained model
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
import re

from models.konkani_translator import create_translator


class TranslationDataset(Dataset):
    """Dataset for translation training"""
    
    def __init__(self, data_file, src_word2id=None, tgt_word2id=None):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Build vocabularies if not provided
        if src_word2id is None:
            self.src_word2id = self._build_vocab([item['konkani'] for item in self.data])
        else:
            self.src_word2id = src_word2id
        
        if tgt_word2id is None:
            self.tgt_word2id = self._build_vocab([item['english'] for item in self.data])
        else:
            self.tgt_word2id = tgt_word2id
        
        self.src_id2word = {v: k for k, v in self.src_word2id.items()}
        self.tgt_id2word = {v: k for k, v in self.tgt_word2id.items()}
    
    def _build_vocab(self, texts):
        """Build vocabulary from texts"""
        word2id = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        
        for text in texts:
            tokens = self._tokenize(text)
            for token in tokens:
                if token not in word2id:
                    word2id[token] = len(word2id)
        
        return word2id
    
    def _tokenize(self, text):
        """Simple tokenization"""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\S+', text.lower())
        return tokens
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        src_tokens = self._tokenize(item['konkani'])
        tgt_tokens = self._tokenize(item['english'])
        
        # Convert to IDs
        src_ids = [self.src_word2id.get(token, 1) for token in src_tokens]
        tgt_ids = [2] + [self.tgt_word2id.get(token, 1) for token in tgt_tokens] + [3]  # Add <SOS> and <EOS>
        
        return {
            'src_ids': torch.tensor(src_ids, dtype=torch.long),
            'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long),
            'src_text': item['konkani'],
            'tgt_text': item['english']
        }


def collate_fn(batch):
    """Custom collate function"""
    src_ids = [item['src_ids'] for item in batch]
    tgt_ids = [item['tgt_ids'] for item in batch]
    
    # Get lengths
    src_lengths = torch.tensor([len(ids) for ids in src_ids], dtype=torch.long)
    
    # Pad sequences
    src_ids_padded = pad_sequence(src_ids, batch_first=True, padding_value=0)
    tgt_ids_padded = pad_sequence(tgt_ids, batch_first=True, padding_value=0)
    
    return {
        'src_ids': src_ids_padded,
        'tgt_ids': tgt_ids_padded,
        'src_lengths': src_lengths
    }


class TranslationTrainer:
    """Trainer for translation model"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
        
        # Loss function (ignore padding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Checkpointing
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            src_ids = batch['src_ids'].to(self.device)
            tgt_ids = batch['tgt_ids'].to(self.device)
            src_lengths = batch['src_lengths']
            
            # Forward pass
            outputs = self.model(src_ids, src_lengths, tgt_ids, teacher_forcing_ratio=0.5)
            
            # Calculate loss (ignore first token which is <SOS>)
            output_dim = outputs.shape[-1]
            outputs = outputs[:, 1:].reshape(-1, output_dim)
            tgt_ids = tgt_ids[:, 1:].reshape(-1)
            
            loss = self.criterion(outputs, tgt_ids)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, epoch):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                src_ids = batch['src_ids'].to(self.device)
                tgt_ids = batch['tgt_ids'].to(self.device)
                src_lengths = batch['src_lengths']
                
                # Forward pass (no teacher forcing)
                outputs = self.model(src_ids, src_lengths, tgt_ids, teacher_forcing_ratio=0)
                
                # Calculate loss
                output_dim = outputs.shape[-1]
                outputs = outputs[:, 1:].reshape(-1, output_dim)
                tgt_ids = tgt_ids[:, 1:].reshape(-1)
                
                loss = self.criterion(outputs, tgt_ids)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save latest
        checkpoint_path = self.checkpoint_dir / f'translation_checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best_translation_model.pt'
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model with val_loss: {val_loss:.4f}")
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"\n{'='*70}")
        print("STARTING KONKANI TRANSLATION TRAINING")
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
            val_loss = self.evaluate(epoch)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Konkani Translation')
    parser.add_argument('--data_file', type=str, 
                        default='data/translation_data.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/translation')
    
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
    dataset = TranslationDataset(args.data_file)
    
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
    src_vocab_size = len(dataset.src_word2id)
    tgt_vocab_size = len(dataset.tgt_word2id)
    
    model = create_translator(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embedding_dim=256,
        hidden_dim=512,
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
            'src_word2id': dataset.src_word2id,
            'tgt_word2id': dataset.tgt_word2id
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved vocabularies to: {vocab_file}")
    
    # Training config
    config = {
        'learning_rate': args.learning_rate,
        'checkpoint_dir': args.checkpoint_dir
    }
    
    # Create trainer
    trainer = TranslationTrainer(model, train_loader, val_loader, device, config)
    
    # Train
    trainer.train(args.num_epochs)


if __name__ == "__main__":
    main()
