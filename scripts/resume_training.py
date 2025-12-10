#!/usr/bin/env python3
"""
Resume ASR training from best checkpoint
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import json
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkanivani_asr import KonkaniVaniASR
from data.audio_processing.audio_processor import AudioProcessor
from data.audio_processing.dataset import KonkaniASRDataset

def resume_training():
    """Resume training from best checkpoint"""
    
    # Configuration
    config = {
        'batch_size': 8,
        'learning_rate': 0.0001,
        'num_epochs': 50,
        'save_every': 5,
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'checkpoint_dir': 'checkpoints/asr_resumed',
        'resume_from': '/Volumes/data&proj/konkani/kaggle_downloads/20251210_060024/checkpoints/best_model.pt'
    }
    
    print("="*70)
    print("RESUME ASR TRAINING")
    print("="*70)
    print(f"Device: {config['device']}")
    print(f"Resume from: {config['resume_from']}")
    
    # Create checkpoint directory
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {config['resume_from']}")
    checkpoint = torch.load(config['resume_from'], map_location='cpu')
    
    # Infer model architecture from checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    vocab_size = state_dict['ctc_head.weight'].shape[0]
    d_model = state_dict['encoder.input_proj.weight'].shape[0]
    
    print(f"Inferred vocab size: {vocab_size}")
    print(f"Inferred d_model: {d_model}")
    
    # Create model
    model = KonkaniVaniASR(
        vocab_size=vocab_size,
        d_model=d_model,
        encoder_layers=12,
        dropout=0.1
    )
    
    # Load state dict
    model.load_state_dict(state_dict)
    model = model.to(config['device'])
    
    print(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Load vocabulary
    vocab = load_vocab()
    print(f"✓ Vocabulary loaded ({len(vocab)} tokens)")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config['batch_size'])
    print(f"✓ Data loaders created")
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Resume from checkpoint epoch
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    print(f"\nStarting training from epoch {start_epoch}")
    print(f"Best validation loss so far: {best_val_loss:.4f}")
    
    # Training loop
    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        print(f"\nEpoch {epoch}/{start_epoch + config['num_epochs'] - 1}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, ctc_loss, config['device'])
        
        # Validate
        val_loss = validate_epoch(model, val_loader, ctc_loss, config['device'])
        
        # Update scheduler
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if epoch % config['save_every'] == 0 or val_loss < best_val_loss:
            checkpoint_path = Path(config['checkpoint_dir']) / f'checkpoint_epoch_{epoch}.pt'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'vocab': vocab
            }, checkpoint_path)
            
            print(f"✓ Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = Path(config['checkpoint_dir']) / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'config': config,
                    'vocab': vocab
                }, best_path)
                print(f"✓ New best model saved: {best_path}")

def load_vocab():
    """Load vocabulary from manifest"""
    manifest_path = Path('data/konkani-asr-v0/splits/manifests/train.json')
    
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return None
    
    vocab = {'<blank>': 0, '<unk>': 1}
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '')
            for char in text:
                if char not in vocab:
                    vocab[char] = len(vocab)
    
    return vocab

def create_data_loaders(batch_size):
    """Create train and validation data loaders"""
    # This is a simplified version - you'll need to implement proper data loading
    # based on your existing dataset structure
    
    train_manifest = 'data/konkani-asr-v0/splits/manifests/train.json'
    val_manifest = 'data/konkani-asr-v0/splits/manifests/val.json'
    
    # Create datasets (you'll need to implement KonkaniASRDataset)
    # train_dataset = KonkaniASRDataset(train_manifest)
    # val_dataset = KonkaniASRDataset(val_manifest)
    
    # For now, return None - you'll need to implement this
    print("⚠️  Data loaders not implemented - you'll need to add your dataset code")
    return None, None

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    if dataloader is None:
        print("⚠️  No dataloader - skipping training")
        return 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Implement training step
        pass
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    
    if dataloader is None:
        print("⚠️  No dataloader - skipping validation")
        return 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Implement validation step
            pass
    
    return total_loss / len(dataloader)

if __name__ == '__main__':
    resume_training()