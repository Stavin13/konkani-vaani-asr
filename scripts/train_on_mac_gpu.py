#!/usr/bin/env python3
"""
Train Translation and Emotion models on Mac GPU (Apple Silicon)
Optimized for M1/M2/M3 chips using MPS backend
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
from models.konkani_custom_emotion import create_custom_emotion_model, EmotionLoss


def check_mac_gpu():
    """Check Mac GPU availability"""
    print("="*70)
    print("MAC GPU CHECK")
    print("="*70)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"MPS (Metal) available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n✅ Mac GPU is ready yaar! Using device: {device}")
        
        # Test GPU
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        start = time.time()
        z = torch.matmul(x, y)
        torch.mps.synchronize()  # Wait for GPU to finish
        elapsed = time.time() - start
        print(f"✅ GPU test done! Matrix multiply: {elapsed*1000:.2f}ms")
        
        return device
    else:
        print("\n⚠️  MPS not available yaar, using CPU only")
        return torch.device("cpu")


# ============================================================================
# REAL DATASETS
# ============================================================================

class EmotionDataset(Dataset):
    """Real emotion detection dataset"""
    def __init__(self, split='train', max_len=100):
        self.max_len = max_len
        
        # Load data
        data_path = Path(f'data/emotion_data/splits/{split}.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Build vocabulary from all splits
        self.vocab = self._build_vocab()
        self.emotion_map = {
            'joy': 0, 'sadness': 1, 'anger': 2, 
            'fear': 3, 'surprise': 4, 'disgust': 5, 'neutral': 6
        }
        
        print(f"Loaded {len(self.data)} {split} samples yaar")
        print(f"Vocab size: {len(self.vocab)}")
    
    def _build_vocab(self):
        """Build vocabulary from all splits"""
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        for split in ['train', 'val', 'test']:
            data_path = Path(f'data/emotion_data/splits/{split}.json')
            if not data_path.exists():
                continue
            
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                text = item['text']
                for char in text:
                    if char not in vocab:
                        vocab[char] = len(vocab)
        
        return vocab
    
    def _tokenize(self, text):
        """Simple character-level tokenization"""
        tokens = [self.vocab.get(char, self.vocab['<UNK>']) for char in text]
        
        # Pad or truncate
        if len(tokens) < self.max_len:
            attention_mask = [1] * len(tokens) + [0] * (self.max_len - len(tokens))
            tokens = tokens + [self.vocab['<PAD>']] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
            attention_mask = [1] * self.max_len
        
        return torch.tensor(tokens), torch.tensor(attention_mask)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        emotion = item['emotion']
        
        input_ids, attention_mask = self._tokenize(text)
        label = self.emotion_map[emotion]
        
        return input_ids, attention_mask, label


class TranslationDataset(Dataset):
    """Real translation dataset"""
    def __init__(self, split='train', max_len=150):
        self.max_len = max_len
        
        # Load data (use augmented if available)
        augmented_path = Path('data/translation_data/konkani_english_augmented.json')
        original_path = Path('data/translation_data/konkani_english_translated.json')
        
        if augmented_path.exists():
            data_path = augmented_path
        else:
            data_path = original_path
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        # Split data
        n_train = int(len(all_data) * 0.8)
        n_val = int(len(all_data) * 0.1)
        
        if split == 'train':
            self.data = all_data[:n_train]
        elif split == 'val':
            self.data = all_data[n_train:n_train+n_val]
        else:
            self.data = all_data[n_train+n_val:]
        
        # Build vocabularies
        self.src_vocab = self._build_vocab('konkani')
        self.tgt_vocab = self._build_vocab('english')
        
        print(f"Loaded {len(self.data)} {split} translation pairs yaar")
        print(f"Konkani vocab: {len(self.src_vocab)}")
        print(f"English vocab: {len(self.tgt_vocab)}")
    
    def _build_vocab(self, lang):
        """Build vocabulary for language"""
        vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        
        # Use augmented data if available
        augmented_path = Path('data/translation_data/konkani_english_augmented.json')
        original_path = Path('data/translation_data/konkani_english_translated.json')
        
        if augmented_path.exists():
            data_path = augmented_path
        else:
            data_path = original_path
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
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


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_emotion_model(device, num_epochs=10, batch_size=32):
    """Train emotion detection model on Mac GPU"""
    print("\n" + "="*70)
    print("TRAINING EMOTION MODEL ON MAC GPU")
    print("="*70)
    
    # Create dataset first to get vocab size
    train_dataset = EmotionDataset(split='train')
    val_dataset = EmotionDataset(split='val')
    
    # Create model with actual vocab size
    vocab_size = len(train_dataset.vocab)
    model = create_custom_emotion_model(vocab_size=vocab_size, num_emotions=7)
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
    
    # Loss and optimizer
    criterion = EmotionLoss(num_emotions=7)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Training loop
    print("\nStarting training yaar...")
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for input_ids, attention_mask, labels in pbar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                logits, _ = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
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
    
    # Save model
    checkpoint_dir = Path('checkpoints/emotion_model')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'vocab': train_dataset.vocab,
        'emotion_map': train_dataset.emotion_map,
        'config': {
            'vocab_size': vocab_size,
            'num_emotions': 7,
            'num_params': num_params
        }
    }, checkpoint_dir / 'emotion_model_mac.pt')
    
    print(f"\n✓ Model saved boss: {checkpoint_dir / 'emotion_model_mac.pt'}")
    
    return history


def train_translation_model(device, num_epochs=10, batch_size=16):
    """Train translation model on Mac GPU"""
    print("\n" + "="*70)
    print("TRAINING TRANSLATION MODEL ON MAC GPU")
    print("="*70)
    
    # Create dataset first to get vocab sizes
    train_dataset = TranslationDataset(split='train')
    val_dataset = TranslationDataset(split='val')
    
    # Create model with actual vocab sizes
    src_vocab_size = len(train_dataset.src_vocab)
    tgt_vocab_size = len(train_dataset.tgt_vocab)
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
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
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
            predictions = torch.argmax(output, dim=-1)
            train_correct += (predictions == tgt_output).sum().item()
            train_total += tgt_output.numel()
            
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
                val_correct += (predictions == tgt_output).sum().item()
                val_total += tgt_output.numel()
        
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
    
    # Save model
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
        }
    }, checkpoint_dir / 'translation_model_mac.pt')
    
    print(f"\n✓ Model saved boss: {checkpoint_dir / 'translation_model_mac.pt'}")
    
    return history


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("TRAIN TRANSLATION & EMOTION MODELS ON MAC GPU")
    print("="*70)
    
    # Check GPU
    device = check_mac_gpu()
    
    if device.type != 'mps':
        print("\n⚠️  Warning yaar: MPS not available. Training will be slower on CPU.")
        response = input("Continue with CPU only? (y/n): ")
        if response.lower() != 'y':
            print("Okay boss, exiting...")
            return
    
    # Training configuration
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print("\nEmotion Model:")
    print("  - Real data: 2,800 training samples (7 emotions)")
    print("  - Parameters: ~3.1M")
    print("  - Batch size: 32")
    print("  - Epochs: 10")
    print("  - Estimated time: 5-10 minutes")
    
    print("\nTranslation Model:")
    print("  - Real data: 80 Konkani-English pairs")
    print("  - Parameters: ~17.5M")
    print("  - Batch size: 16")
    print("  - Epochs: 10")
    print("  - Estimated time: 15-25 minutes")
    
    print("\nTotal estimated time: 20-35 minutes")
    
    response = input("\nShall we start training? (y/n): ")
    if response.lower() != 'y':
        print("Okay boss, exiting...")
        return
    
    # Train emotion model
    print("\n🚀 Training emotion model first...")
    emotion_history = train_emotion_model(device, num_epochs=10, batch_size=32)
    
    # Train translation model
    print("\n🚀 Now training translation model...")
    translation_history = train_translation_model(device, num_epochs=10, batch_size=16)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE BOSS! 🎉")
    print("="*70)
    print("\nModels saved at:")
    print("  - checkpoints/emotion_model/emotion_model_mac.pt")
    print("  - checkpoints/translation_model/translation_model_mac.pt")
    
    print("\nWhat's next:")
    print("  1. Test models on test set")
    print("  2. Create visualizations")
    print("  3. Fine-tune if needed")
    print("  4. Deploy for inference")


if __name__ == '__main__':
    main()
