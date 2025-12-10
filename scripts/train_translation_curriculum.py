#!/usr/bin/env python3
"""
Train Translation model using Curriculum Learning
Progressively trains on: letters → words → phrases → sentences → complex
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
        print(f"\n✅ Mac GPU ready! Using device: {device}")
        return device
    else:
        print("\n⚠️  MPS not available, using CPU")
        return torch.device("cpu")


class CurriculumDataset(Dataset):
    """Dataset that supports curriculum learning"""
    def __init__(self, data, max_len=150, src_vocab=None, tgt_vocab=None):
        self.data = data
        self.max_len = max_len
        
        # Build or reuse vocabularies
        if src_vocab is None or tgt_vocab is None:
            self.src_vocab = self._build_vocab('konkani')
            self.tgt_vocab = self._build_vocab('english')
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
    
    def _build_vocab(self, lang):
        """Build vocabulary for language"""
        vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        
        for item in self.data:
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


def train_curriculum_stage(model, device, stage_data, stage_name, num_epochs, 
                           batch_size, optimizer, criterion, history):
    """Train one curriculum stage"""
    print(f"\n{'='*70}")
    print(f"STAGE: {stage_name.upper()}")
    print(f"{'='*70}")
    print(f"Examples: {len(stage_data)}")
    print(f"Epochs: {num_epochs}")
    
    # Split into train/val
    n_train = int(len(stage_data) * 0.9)
    train_data = stage_data[:n_train]
    val_data = stage_data[n_train:]
    
    # Create datasets (reuse vocabularies from model)
    train_dataset = CurriculumDataset(train_data, 
                                     src_vocab=history['src_vocab'],
                                     tgt_vocab=history['tgt_vocab'])
    val_dataset = CurriculumDataset(val_data,
                                   src_vocab=history['src_vocab'],
                                   tgt_vocab=history['tgt_vocab'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    pad_idx = 0
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"{stage_name} Epoch {epoch}/{num_epochs}")
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
                
                output = model(src, tgt_input, tgt_mask=tgt_mask)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                
                val_loss += loss.item()
                
                predictions = torch.argmax(output, dim=-1)
                mask = tgt_output != pad_idx
                val_correct += (predictions[mask] == tgt_output[mask]).sum().item()
                val_total += mask.sum().item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        print(f"  Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
              f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        # Save history
        history['stages'][stage_name]['train_loss'].append(avg_train_loss)
        history['stages'][stage_name]['val_loss'].append(avg_val_loss)
        history['stages'][stage_name]['train_acc'].append(train_acc)
        history['stages'][stage_name]['val_acc'].append(val_acc)
        
        # Save best for this stage
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_dir = Path('checkpoints/translation_model')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'stage': stage_name,
                'best_val_loss': best_val_loss
            }, checkpoint_dir / f'curriculum_{stage_name}_best.pt')


def train_with_curriculum(device, all_data, batch_size=32):
    """Train using curriculum learning"""
    print("\n" + "="*70)
    print("CURRICULUM LEARNING TRAINING")
    print("="*70)
    
    # Build vocabularies from ALL data first
    print("\nBuilding vocabularies from all data...")
    temp_dataset = CurriculumDataset(all_data)
    src_vocab = temp_dataset.src_vocab
    tgt_vocab = temp_dataset.tgt_vocab
    
    print(f"Konkani vocab size: {len(src_vocab)}")
    print(f"English vocab size: {len(tgt_vocab)}")
    
    # Create model
    model = create_custom_translation_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab)
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer and criterion
    pad_idx = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # History
    history = {
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'stages': {
            'letters': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []},
            'words': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []},
            'phrases': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []},
            'sentences': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []},
            'complex': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        }
    }
    
    # Organize data by difficulty
    stages = {
        'letters': [],
        'words': [],
        'phrases': [],
        'sentences': [],
        'complex': []
    }
    
    for item in all_data:
        level = item['level']
        if 'letter' in level:
            stages['letters'].append(item)
        elif 'word' in level:
            stages['words'].append(item)
        elif 'phrase' in level:
            stages['phrases'].append(item)
        elif 'sentence' in level:
            stages['sentences'].append(item)
        else:
            stages['complex'].append(item)
    
    # Train each stage progressively
    stage_configs = [
        ('letters', 10, 16),      # 10 epochs, batch 16
        ('words', 15, 24),        # 15 epochs, batch 24
        ('phrases', 15, 32),      # 15 epochs, batch 32
        ('sentences', 20, 32),    # 20 epochs, batch 32
        ('complex', 30, 32)       # 30 epochs, batch 32
    ]
    
    for stage_name, num_epochs, stage_batch_size in stage_configs:
        stage_data = stages[stage_name]
        if len(stage_data) == 0:
            print(f"\n⚠️  Skipping {stage_name} - no data")
            continue
        
        train_curriculum_stage(
            model, device, stage_data, stage_name,
            num_epochs, stage_batch_size, optimizer, criterion, history
        )
    
    # Save final model
    checkpoint_dir = Path('checkpoints/translation_model')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'config': {
            'src_vocab_size': len(src_vocab),
            'tgt_vocab_size': len(tgt_vocab),
            'num_params': num_params
        }
    }, checkpoint_dir / 'translation_model_curriculum_final.pt')
    
    print(f"\n✅ Final curriculum model saved!")
    return history


def plot_curriculum_history(history):
    """Plot training history for all stages"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    stages = ['letters', 'words', 'phrases', 'sentences', 'complex']
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for idx, (stage, color) in enumerate(zip(stages, colors)):
        if stage not in history['stages'] or len(history['stages'][stage]['train_loss']) == 0:
            continue
        
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        stage_hist = history['stages'][stage]
        epochs = range(1, len(stage_hist['train_loss']) + 1)
        
        ax.plot(epochs, stage_hist['train_loss'], f'{color[0]}-', label='Train Loss', linewidth=2)
        ax.plot(epochs, stage_hist['val_loss'], f'{color[0]}--', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{stage.capitalize()} Stage', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    if len(stages) < 6:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    output_path = Path('outputs/translation_curriculum_training.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training graph saved to: {output_path}")


def main():
    print("\n" + "="*70)
    print("TRAIN TRANSLATION WITH CURRICULUM LEARNING")
    print("="*70)
    
    # Check GPU
    device = check_mac_gpu()
    
    # Load curriculum data
    data_path = Path('data/translation_data/konkani_english_curriculum_sorted.json')
    if not data_path.exists():
        print(f"\n❌ Curriculum data not found: {data_path}")
        print("\nRun first: python scripts/generate_progressive_translation_data.py")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    print(f"\n✓ Loaded {len(all_data)} training examples")
    
    # Train
    history = train_with_curriculum(device, all_data)
    
    # Plot
    plot_curriculum_history(history)
    
    print("\n" + "="*70)
    print("CURRICULUM TRAINING COMPLETE! 🎉")
    print("="*70)
    print("\nModels saved at:")
    print("  - checkpoints/translation_model/curriculum_*_best.pt")
    print("  - checkpoints/translation_model/translation_model_curriculum_final.pt")


if __name__ == '__main__':
    main()
