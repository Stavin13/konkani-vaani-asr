#!/usr/bin/env python3
"""
Retrain translation model with clean data from pre-trained models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkani_custom_translator import create_custom_translation_model


class CleanTranslationDataset(Dataset):
    """Dataset using clean pre-translated data"""
    def __init__(self, data_path, split='train', max_len=150):
        self.max_len = max_len
        
        # Load clean data
        print(f"Loading data from: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        # Filter by confidence if available
        all_data = [d for d in all_data if d.get('confidence', 1.0) > 0.5]
        
        # Filter out empty translations
        all_data = [d for d in all_data if d['english'].strip() and d['konkani'].strip()]
        
        print(f"✓ Loaded {len(all_data)} clean translation pairs")
        
        # Split data (80/10/10)
        n_train = int(len(all_data) * 0.8)
        n_val = int(len(all_data) * 0.1)
        
        if split == 'train':
            self.data = all_data[:n_train]
        elif split == 'val':
            self.data = all_data[n_train:n_train+n_val]
        else:
            self.data = all_data[n_train+n_val:]
        
        # Build vocabularies from ALL data (not just split)
        self.src_vocab = self._build_vocab('konkani', all_data)
        self.tgt_vocab = self._build_vocab('english', all_data)
        
        print(f"✓ {split} set: {len(self.data)} pairs")
        print(f"✓ Konkani vocab: {len(self.src_vocab)} chars")
        print(f"✓ English vocab: {len(self.tgt_vocab)} chars")
    
    def _build_vocab(self, lang, all_data):
        """Build character-level vocabulary"""
        vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        
        for item in all_data:
            text = item[lang]
            for char in text:
                if char not in vocab:
                    vocab[char] = len(vocab)
        
        return vocab
    
    def _tokenize(self, text, vocab, add_special=False):
        """Character-level tokenization"""
        if add_special:
            tokens = [vocab['<SOS>']]
            tokens.extend([vocab.get(char, vocab['<UNK>']) for char in text])
            tokens.append(vocab['<EOS>'])
        else:
            tokens = [vocab.get(char, vocab['<UNK>']) for char in text]
        
        # Pad or truncate
        if len(tokens) < self.max_len:
            tokens = tokens + [vocab['<PAD>']] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        
        return torch.tensor(tokens)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        src = self._tokenize(item['konkani'], self.src_vocab)
        tgt = self._tokenize(item['english'], self.tgt_vocab, add_special=True)
        return src, tgt


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for src, tgt in tqdm(dataloader, desc="Training"):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Prepare target
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        
        # Forward
        optimizer.zero_grad()
        output = model(src, tgt_input, tgt_mask=tgt_mask)
        
        # Loss
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        predictions = torch.argmax(output, dim=-1)
        correct += (predictions == tgt_output).sum().item()
        total += tgt_output.numel()
    
    return total_loss / len(dataloader), 100 * correct / total


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            output = model(src, tgt_input, tgt_mask=tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            total_loss += loss.item()
            predictions = torch.argmax(output, dim=-1)
            correct += (predictions == tgt_output).sum().item()
            total += tgt_output.numel()
    
    return total_loss / len(dataloader), 100 * correct / total


def main():
    print("\n" + "="*70)
    print("RETRAIN TRANSLATION MODEL WITH CLEAN DATA")
    print("="*70)
    
    # Check for clean data
    data_paths = [
        'data/translation_data/konkani_english_google.json',
        'data/translation_data/konkani_english_pretrained.json',
        'data/translation_data/konkani_english_indictrans.json'
    ]
    
    data_path = None
    for path in data_paths:
        if Path(path).exists():
            data_path = path
            break
    
    if not data_path:
        print("\n❌ No clean translation data found!")
        print("\nGenerate data first with:")
        print("  python scripts/quick_translate_with_google.py")
        print("\nOr:")
        print("  python scripts/generate_translation_data_with_pretrained.py")
        return
    
    print(f"\n✓ Using data: {data_path}")
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✓ Using Mac GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print(f"⚠️  Using CPU (will be slow)")
    
    # Load datasets
    train_dataset = CleanTranslationDataset(data_path, split='train')
    val_dataset = CleanTranslationDataset(data_path, split='val')
    
    # Create model
    src_vocab_size = len(train_dataset.src_vocab)
    tgt_vocab_size = len(train_dataset.tgt_vocab)
    model = create_custom_translation_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model: {num_params:,} parameters")
    
    # Dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 0
    max_patience = 15
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print(f"   Batch size: {batch_size}")
    print(f"   Early stopping patience: {max_patience}\n")
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch:2d}: Train Loss={train_loss:.4f} Acc={train_acc:.1f}% | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.1f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            
            checkpoint_dir = Path('checkpoints/translation_model')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'src_vocab': train_dataset.src_vocab,
                'tgt_vocab': train_dataset.tgt_vocab,
                'config': {
                    'src_vocab_size': src_vocab_size,
                    'tgt_vocab_size': tgt_vocab_size,
                    'num_params': num_params
                },
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, checkpoint_dir / 'translation_model_best.pt')
            
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE! 🎉")
    print("="*70)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {max(history['val_acc']):.1f}%")
    print(f"\nModel saved at: checkpoints/translation_model/translation_model_best.pt")
    print(f"\nTest it with:")
    print(f"  python scripts/test_translation_best.py")


if __name__ == '__main__':
    main()
