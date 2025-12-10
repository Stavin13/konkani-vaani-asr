#!/usr/bin/env python3
"""
Train Translation model with ALL available data combined
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import time
import sys
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkani_custom_translator import create_custom_translation_model


def check_mac_gpu():
    """Check Mac GPU availability"""
    print("="*70)
    print("MAC GPU CHECK")
    print("="*70)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"MPS (Metal) available: {torch.backends.mps.is_available()}")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n✅ Mac GPU is ready yaar! Using device: {device}")
        
        # Test GPU
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        start = time.time()
        z = torch.matmul(x, y)
        torch.mps.synchronize()
        elapsed = time.time() - start
        print(f"✅ GPU test done! Matrix multiply: {elapsed*1000:.2f}ms")
        
        return device
    else:
        print("\n⚠️  MPS not available yaar, using CPU only")
        return torch.device("cpu")


def load_all_translation_data():
    """Load and combine all translation data sources"""
    all_data = []
    
    # 1. Load augmented JSON data
    augmented_path = Path('data/translation_data/konkani_english_augmented.json')
    if augmented_path.exists():
        with open(augmented_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✓ Loaded {len(data)} pairs from augmented JSON")
            all_data.extend(data)
    
    # 2. Load translated JSON data
    translated_path = Path('data/translation_data/konkani_english_translated.json')
    if translated_path.exists():
        with open(translated_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✓ Loaded {len(data)} pairs from translated JSON")
            all_data.extend(data)
    
    # 3. Load JSONL data (clean up English translations)
    jsonl_path = Path('data/kok_eng_dataset.jsonl')
    if jsonl_path.exists():
        count = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # Clean up English translation (remove "Here's the translation..." prefix)
                eng = item['english']
                if 'Here\'s the translation' in eng:
                    # Extract actual translation
                    parts = eng.split('\n\n')
                    if len(parts) > 1:
                        eng = parts[1].strip('"')
                    # Remove trailing notes
                    if '\n\nNote:' in eng:
                        eng = eng.split('\n\nNote:')[0]
                
                all_data.append({
                    'konkani': item['konkani'],
                    'english': eng,
                    'source': 'jsonl_' + item.get('source', 'unknown')
                })
                count += 1
        print(f"✓ Loaded {count} pairs from JSONL (cleaned)")
    
    print(f"\n📊 Total translation pairs: {len(all_data)}")
    return all_data


class TranslationDataset(Dataset):
    """Combined translation dataset"""
    def __init__(self, data, split='train', max_len=150, src_vocab=None, tgt_vocab=None):
        self.max_len = max_len
        
        # Split data
        n_train = int(len(data) * 0.8)
        n_val = int(len(data) * 0.1)
        
        if split == 'train':
            self.data = data[:n_train]
        elif split == 'val':
            self.data = data[n_train:n_train+n_val]
        else:
            self.data = data[n_train+n_val:]
        
        # Build or reuse vocabularies
        if src_vocab is None or tgt_vocab is None:
            self.src_vocab = self._build_vocab('konkani', data)
            self.tgt_vocab = self._build_vocab('english', data)
            print(f"Built vocabularies - Konkani: {len(self.src_vocab)}, English: {len(self.tgt_vocab)}")
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
        
        print(f"Loaded {len(self.data)} {split} translation pairs")
    
    def _build_vocab(self, lang, all_data):
        """Build vocabulary for language"""
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
        src_text = item['konkani']
        tgt_text = item['english']
        
        src = self._tokenize(src_text, self.src_vocab)
        tgt = self._tokenize(tgt_text, self.tgt_vocab, add_special=True)
        
        return src, tgt


def train_translation_model(device, all_data, num_epochs=50, batch_size=32):
    """Train translation model on Mac GPU"""
    print("\n" + "="*70)
    print("TRAINING TRANSLATION MODEL")
    print("="*70)
    
    # Build vocabularies once from all data
    print("\nBuilding vocabularies from all data...")
    temp_dataset = TranslationDataset(all_data, split='train')
    src_vocab = temp_dataset.src_vocab
    tgt_vocab = temp_dataset.tgt_vocab
    
    # Create datasets with shared vocabularies
    train_dataset = TranslationDataset(all_data, split='train', src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    val_dataset = TranslationDataset(all_data, split='val', src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    
    # Create model with actual vocab sizes
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    model = create_custom_translation_model(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Loss and optimizer (lower LR for stability)
    pad_idx = 0  # <PAD> token index
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15
    
    # Training loop
    print("\nStarting training yaar...")
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
            
            # Create target mask
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
            
            # Accuracy: ignore PAD tokens
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
                
                # Accuracy: ignore PAD tokens
                predictions = torch.argmax(output, dim=-1)
                mask = tgt_output != pad_idx
                val_correct += (predictions[mask] == tgt_output[mask]).sum().item()
                val_total += mask.sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
              f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint_dir = Path('checkpoints/translation_model')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
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
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n⚠️  Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                break
    
    # Save final model
    checkpoint_dir = Path('checkpoints/translation_model')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'config': {
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size,
            'num_params': num_params
        }
    }, checkpoint_dir / 'translation_model_final.pt')
    
    print(f"\n✓ Final model saved: {checkpoint_dir / 'translation_model_final.pt'}")
    print(f"✓ Best model saved: {checkpoint_dir / 'translation_model_best.pt'}")
    
    return history


def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Translation Model - Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Translation Model - Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path('outputs/translation_training_combined.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training graph saved to: {output_path}")


def main():
    print("\n" + "="*70)
    print("TRAIN TRANSLATION MODEL WITH ALL COMBINED DATA")
    print("="*70)
    
    # Check GPU
    device = check_mac_gpu()
    
    # Load all data
    print("\n" + "="*70)
    print("LOADING ALL TRANSLATION DATA")
    print("="*70)
    all_data = load_all_translation_data()
    
    if len(all_data) == 0:
        print("\n❌ No translation data found!")
        return
    
    # 🔀 IMPORTANT: shuffle before splitting to avoid distribution bias
    print("\n🔀 Shuffling data for better train/val split...")
    random.seed(42)
    random.shuffle(all_data)
    
    # Training configuration
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"\nTotal data pairs: {len(all_data)}")
    print(f"Train: {int(len(all_data) * 0.8)} pairs (80%)")
    print(f"Val: {int(len(all_data) * 0.1)} pairs (10%)")
    print(f"Test: {len(all_data) - int(len(all_data) * 0.9)} pairs (10%)")
    print("\nModel:")
    print("  - Parameters: ~11.2M")
    print("  - Batch size: 32")
    print("  - Max epochs: 50")
    print("  - Early stopping: patience 15")
    print(f"  - Device: {device}")
    
    response = input("\nShall we start training? (y/n): ")
    if response.lower() != 'y':
        print("Okay boss, exiting...")
        return
    
    # Train translation model
    print("\n🚀 Training translation model...")
    history = train_translation_model(device, all_data, num_epochs=50, batch_size=32)
    
    # Plot results
    plot_training_history(history)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE! 🎉")
    print("="*70)
    print("\nModels saved at:")
    print("  - checkpoints/translation_model/translation_model_best.pt")
    print("  - checkpoints/translation_model/translation_model_final.pt")
    
    print("\nFinal Results:")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Val Loss:   {history['val_loss'][-1]:.4f}")
    print(f"  Train Acc:  {history['train_acc'][-1]:.2f}%")
    print(f"  Val Acc:    {history['val_acc'][-1]:.2f}%")
    print(f"  Best Val Acc: {max(history['val_acc']):.2f}%")


if __name__ == '__main__':
    main()
