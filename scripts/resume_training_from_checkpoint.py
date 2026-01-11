#!/usr/bin/env python3
"""
Resume training from a checkpoint
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from pathlib import Path
import sys
import os
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.load_checkpoint_for_training import load_checkpoint_for_training, setup_optimizer_and_scheduler


def create_dummy_dataloader(vocab_info, batch_size=4, num_samples=100):
    """
    Create a dummy dataloader for testing
    Replace this with your actual data loading logic
    """
    
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, vocab_info, num_samples):
            self.vocab_info = vocab_info
            self.num_samples = num_samples
            self.vocab_size = len(vocab_info['char_to_idx'])
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Dummy mel spectrogram (time_steps, mel_features)
            time_steps = torch.randint(50, 200, (1,)).item()
            mel_spec = torch.randn(time_steps, 80)
            
            # Dummy target sequence
            target_length = torch.randint(10, 30, (1,)).item()
            target = torch.randint(1, self.vocab_size, (target_length,))
            
            return mel_spec, target
    
    dataset = DummyDataset(vocab_info, num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def collate_fn(batch):
    """Collate function for batching variable length sequences"""
    mel_specs, targets = zip(*batch)
    
    # Pad mel spectrograms
    max_mel_len = max(mel.size(0) for mel in mel_specs)
    batch_size = len(mel_specs)
    
    padded_mels = torch.zeros(batch_size, max_mel_len, 80)
    mel_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, mel in enumerate(mel_specs):
        mel_len = mel.size(0)
        padded_mels[i, :mel_len] = mel
        mel_lengths[i] = mel_len
    
    # Pad targets
    max_target_len = max(target.size(0) for target in targets)
    padded_targets = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, target in enumerate(targets):
        target_len = target.size(0)
        padded_targets[i, :target_len] = target
        target_lengths[i] = target_len
    
    return padded_mels, padded_targets, mel_lengths, target_lengths


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (mel_specs, targets, mel_lengths, target_lengths) in enumerate(pbar):
        # Move to device
        mel_specs = mel_specs.to(device)
        targets = targets.to(device)
        mel_lengths = mel_lengths.to(device)
        target_lengths = target_lengths.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        try:
            # Get model outputs
            outputs = model(mel_specs)  # Shape: (batch, time, vocab_size)
            
            # Prepare for CTC loss
            # CTC expects (time, batch, vocab_size)
            log_probs = torch.log_softmax(outputs, dim=-1)
            log_probs = log_probs.transpose(0, 1)  # (time, batch, vocab_size)
            
            # Input lengths (time dimension after any downsampling)
            input_lengths = mel_lengths // 4  # Assuming 4x downsampling in encoder
            input_lengths = torch.clamp(input_lengths, min=1)
            
            # CTC Loss
            loss = criterion(
                log_probs=log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
        except Exception as e:
            print(f"❌ Error in batch {batch_idx}: {e}")
            continue
    
    return total_loss / max(num_batches, 1)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir):
    """Save training checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save as latest
    latest_path = checkpoint_dir / "latest_checkpoint.pt"
    torch.save(checkpoint, latest_path)
    
    print(f"💾 Saved checkpoint: {checkpoint_path}")


def main():
    """Main training function"""
    
    # Configuration
    checkpoint_path = "kaggle_asr_outputs/checkpoints/best_model.pt"  # Change this to your desired checkpoint
    num_epochs = 5
    batch_size = 4
    checkpoint_dir = "checkpoints/resumed_training"
    
    print(f"🚀 Resuming training from: {checkpoint_path}")
    
    # Load checkpoint for training
    training_setup = load_checkpoint_for_training(checkpoint_path)
    
    model = training_setup['model']
    optimizer_state = training_setup['optimizer_state']
    scheduler_state = training_setup['scheduler_state']
    start_epoch = training_setup['epoch']
    vocab_info = training_setup['vocab_info']
    device = training_setup['device']
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(
        model, optimizer_state, scheduler_state
    )
    
    # Setup loss function
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # Create dataloader (replace with your actual data)
    print("📊 Creating dataloader...")
    dataloader = create_dummy_dataloader(vocab_info, batch_size=batch_size)
    
    print(f"✅ Setup complete!")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📈 Starting from epoch: {start_epoch + 1}")
    print(f"🎯 Training for {num_epochs} epochs")
    
    # Training loop
    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        print(f"\n🔄 Epoch {epoch}/{start_epoch + num_epochs}")
        
        # Train one epoch
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device, epoch)
        
        print(f"📊 Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        
        # Update scheduler
        scheduler.step(avg_loss)
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, checkpoint_dir)
        
        # Early stopping or other logic can go here
        if avg_loss < 0.1:  # Example early stopping
            print("🎉 Early stopping - loss threshold reached!")
            break
    
    print("✅ Training completed!")


if __name__ == "__main__":
    main()