#!/usr/bin/env python3
"""
Train word-level translation model
Much faster and more effective than character-level
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

from models.word_level_translator import create_word_level_translator


class WordLevelDataset(Dataset):
    """Word-level tokenization dataset"""
    def __init__(self, data, max_len=50, src_vocab=None, tgt_vocab=None):
        self.data = data
        self.max_len = max_len
        
        if src_vocab is None or tgt_vocab is None:
            self.src_vocab = self._build_vocab('konkani')
            self.tgt_vocab = self._build_vocab('english')
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
    
    def _build_vocab(self, lang):
        """Build word-level vocabulary"""
        vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        
        for item in self.data:
            text = item[lang]
            # Simple word tokenization (split on spaces and punctuation)
            words = text.replace(',', ' ,').replace('.', ' .').replace('!', ' !').replace('?', ' ?').split()
            for word in words:
                word = word.strip()
                if word and word not in vocab:
                    vocab[word] = len(vocab)
        
        return vocab
    
    def _tokenize(self, text, vocab, add_special=False):
        """Word-level tokenization"""
        words = text.replace(',', ' ,').replace('.', ' .').replace('!', ' !').replace('?', ' ?').split()
        words = [w.strip() for w in words if w.strip()]
        
        if add_special:
            tokens = [vocab['<SOS>']]
            tokens.extend([vocab.get(word, vocab['<UNK>']) for word in words])
            tokens.append(vocab['<EOS>'])
        else:
            tokens = [vocab.get(word, vocab['<UNK>']) for word in words]
        
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


def train_model(device, train_data, val_data, num_epochs=100, batch_size=64):
    """Train word-level translation model"""
    print("\n" + "="*70)
    print("TRAINING WORD-LEVEL TRANSLATION MODEL")
    print("="*70)
    
    # Build vocabularies
    print("\nBuilding word-level vocabularies...")
    all_data = train_data + val_data
    temp_dataset = WordLevelDataset(all_data)
    src_vocab = temp_dataset.src_vocab
    tgt_vocab = temp_dataset.tgt_vocab
    
    print(f"Konkani vocab: {len(src_vocab)} words")
    print(f"English vocab: {len(tgt_vocab)} words")
    
    # Create datasets
    train_dataset = WordLevelDataset(train_data, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    val_dataset = WordLevelDataset(val_data, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    
    # Create model
    model = create_word_level_translator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Training setup
    pad_idx = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nTraining for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    
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
            
            # Create masks
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            src_padding_mask = (src == pad_idx)
            tgt_padding_mask = (tgt_input == pad_idx)
            
            optimizer.zero_grad()
            output = model(src, tgt_input, tgt_mask=tgt_mask,
                          src_padding_mask=src_padding_mask,
                          tgt_padding_mask=tgt_padding_mask)
            
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy
            predictions = torch.argmax(output, dim=-1)
            mask = tgt_output != pad_idx
            train_correct += (predictions[mask] == tgt_output[mask]).sum().item()
            train_total += mask.sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        
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
                src_padding_mask = (src == pad_idx)
                tgt_padding_mask = (tgt_input == pad_idx)
                
                output = model(src, tgt_input, tgt_mask=tgt_mask,
                              src_padding_mask=src_padding_mask,
                              tgt_padding_mask=tgt_padding_mask)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                
                val_loss += loss.item()
                
                predictions = torch.argmax(output, dim=-1)
                mask = tgt_output != pad_idx
                val_correct += (predictions[mask] == tgt_output[mask]).sum().item()
                val_total += mask.sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
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
                    'd_model': 256
                },
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'val_acc': val_acc
            }, checkpoint_dir / 'translation_word_level_best.pt')
            
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f}, val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break
    
    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")


def main():
    print("="*70)
    print("TRAIN WORD-LEVEL TRANSLATION MODEL")
    print("="*70)
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    
    # Load data
    data_path = Path('data/translation_data/konkani_english_10k.json')
    if not data_path.exists():
        print(f"\n❌ Data not found: {data_path}")
        print("\nRun first: python scripts/generate_10k_translations.py")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    print(f"\n✓ Loaded {len(all_data)} translation pairs")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(all_data)
    
    n_train = int(len(all_data) * 0.85)
    n_val = int(len(all_data) * 0.10)
    
    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train+n_val]
    test_data = all_data[n_train+n_val:]
    
    print(f"\nSplit: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Train
    train_model(device, train_data, val_data, num_epochs=100, batch_size=64)
    
    print("\n" + "="*70)
    print("Test with:")
    print("  python scripts/test_word_level_translator.py")
    print("="*70)


if __name__ == '__main__':
    main()
