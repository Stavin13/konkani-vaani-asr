#!/usr/bin/env python3
"""
Train Translation model with CLEAN curriculum data only
Excludes noisy augmented data
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import sys
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkani_custom_translator import create_custom_translation_model


def check_mac_gpu():
    """Check Mac GPU availability"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Using Mac GPU (MPS)")
        return device
    else:
        print("⚠️  Using CPU")
        return torch.device("cpu")


class TranslationDataset(Dataset):
    """Translation dataset"""
    def __init__(self, data, max_len=150, src_vocab=None, tgt_vocab=None):
        self.data = data
        self.max_len = max_len
        
        if src_vocab is None or tgt_vocab is None:
            self.src_vocab = self._build_vocab('konkani')
            self.tgt_vocab = self._build_vocab('english')
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
    
    def _build_vocab(self, lang):
        vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        for item in self.data:
            text = item[lang]
            for char in text:
                if char not in vocab:
                    vocab[char] = len(vocab)
        return vocab
    
    def _tokenize(self, text, vocab, add_special=False):
        if add_special:
            tokens = [vocab['<SOS>']]
            tokens.extend([vocab.get(char, vocab['<UNK>']) for char in text])
            tokens.append(vocab['<EOS>'])
        else:
            tokens = [vocab.get(char, vocab['<UNK>']) for char in text]
        
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


def train_model(device, train_data, val_data, num_epochs=30, batch_size=32):
    """Train translation model"""
    print("\n" + "="*70)
    print("TRAINING CLEAN TRANSLATION MODEL")
    print("="*70)
    
    # Build vocabularies
    print("\nBuilding vocabularies...")
    all_data = train_data + val_data
    temp_dataset = TranslationDataset(all_data)
    src_vocab = temp_dataset.src_vocab
    tgt_vocab = temp_dataset.tgt_vocab
    
    print(f"Konkani vocab: {len(src_vocab)} chars")
    print(f"English vocab: {len(tgt_vocab)} chars")
    
    # Create datasets
    train_dataset = TranslationDataset(train_data, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    val_dataset = TranslationDataset(val_data, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    model = create_custom_translation_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab)
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    pad_idx = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt_input, tgt_mask=tgt_mask)
            
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            predictions = torch.argmax(output, dim=-1)
            mask = tgt_output != pad_idx
            train_correct += (predictions[mask] == tgt_output[mask]).sum().item()
            train_total += mask.sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
                
                output = model(src, tgt_input, tgt_mask=tgt_mask)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                
                val_loss += loss.item()
                
                predictions = torch.argmax(output, dim=-1)
                mask = tgt_output != pad_idx
                val_correct += (predictions[mask] == tgt_output[mask]).sum().item()
                val_total += mask.sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
              f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            checkpoint_dir = Path('checkpoints/translation_model')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'config': {
                    'src_vocab_size': len(src_vocab),
                    'tgt_vocab_size': len(tgt_vocab),
                },
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'val_acc': val_acc
            }, checkpoint_dir / 'translation_model_clean_best.pt')
            
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f}, val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break
    
    print(f"\n✅ Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")


def main():
    print("="*70)
    print("TRAIN CLEAN TRANSLATION MODEL")
    print("="*70)
    
    device = check_mac_gpu()
    
    # Load ONLY clean curriculum data (exclude complex/noisy data)
    data_path = Path('data/translation_data/konkani_english_curriculum_sorted.json')
    if not data_path.exists():
        print(f"\n❌ Data not found: {data_path}")
        print("Run: python scripts/generate_progressive_translation_data.py")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # Filter out complex/noisy data - keep only clean examples
    clean_data = [d for d in all_data if d['level'] != 'complex']
    
    print(f"\n✓ Loaded {len(clean_data)} clean examples")
    print(f"  (Excluded {len(all_data) - len(clean_data)} complex/noisy examples)")
    
    # Show distribution
    by_level = {}
    for item in clean_data:
        level = item['level']
        by_level[level] = by_level.get(level, 0) + 1
    
    print("\nData distribution:")
    for level, count in sorted(by_level.items()):
        print(f"  {level:20s}: {count:4d} examples")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(clean_data)
    
    n_train = int(len(clean_data) * 0.85)
    train_data = clean_data[:n_train]
    val_data = clean_data[n_train:]
    
    print(f"\nSplit: {len(train_data)} train, {len(val_data)} val")
    
    # Train
    train_model(device, train_data, val_data, num_epochs=30, batch_size=32)
    
    print("\n" + "="*70)
    print("DONE! Test with:")
    print("  python scripts/test_translation_model.py --checkpoint checkpoints/translation_model/translation_model_clean_best.pt")
    print("="*70)


if __name__ == '__main__':
    main()
