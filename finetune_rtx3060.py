#!/usr/bin/env python3
"""
RTX 3060 6GB VRAM Optimized Fine-tuning Script for Konkani ASR Model
Optimized for memory efficiency with gradient accumulation and mixed precision
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchaudio
import json
import argparse
from pathlib import Path
import os
import sys
from tqdm import tqdm
import logging
from datetime import datetime
import librosa
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import gc

# Add current directory to path for imports
sys.path.insert(0, '.')

# Memory optimization settings for RTX 3060
MEMORY_OPTIMIZED_SETTINGS = {
    'batch_size': 2,  # Small batch size for 6GB VRAM
    'gradient_accumulation_steps': 8,  # Effective batch size = 2 * 8 = 16
    'max_audio_length': 16000 * 8,  # 8 seconds max (reduced from 10)
    'num_workers': 1,  # Reduced workers to save memory
    'pin_memory': False,  # Disable pin memory to save VRAM
    'use_mixed_precision': True,
    'gradient_checkpointing': True,
    'max_mel_length': 800,  # Limit mel spectrogram length
}

class MemoryOptimizedAudioDataset(Dataset):
    """Memory-optimized audio dataset for RTX 3060"""
    
    def __init__(self, manifest_path, vocab_path, max_length=None):
        self.manifest_path = manifest_path
        self.max_length = max_length or MEMORY_OPTIMIZED_SETTINGS['max_audio_length']
        
        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.char2idx = vocab_data['char2idx']
        self.idx2char = vocab_data['idx2char']
        self.vocab_size = len(self.char2idx)
        
        # Load and filter manifest for shorter audio files
        self.samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        sample = json.loads(line.strip())
                        # Filter out very long audio files to save memory
                        if sample.get('duration', 0) <= 10.0:  # Max 10 seconds
                            self.samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Loaded {len(self.samples)} samples from {manifest_path} (filtered for memory)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio with path correction for Windows
        audio_path = sample['audio_filepath']
        
        # Fix path for Windows if it's a Unix/Mac path
        if audio_path.startswith('/Volumes/') or audio_path.startswith('/'):
            # Try to find the audio file in local directories
            filename = os.path.basename(audio_path)
            possible_paths = [
                f"data/audio/synthetic/{filename}",
                f"KonkaniRawSpeechCorpus/{filename}",
                f"data/{filename}",
                filename
            ]
            
            audio_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    audio_path = path
                    break
            
            if audio_path is None:
                # If no file found, create silence
                print(f"Audio file not found, using silence for: {sample['audio_filepath']}")
                audio = torch.zeros(16000, dtype=torch.float32)
                return {
                    'audio': F.pad(audio, (0, self.max_length - len(audio))) if len(audio) < self.max_length else audio[:self.max_length],
                    'text': torch.LongTensor([self.char2idx.get('<unk>', 4)]),
                    'text_length': 1,
                    'audio_length': len(audio)
                }
        
        try:
            # Use librosa with lower precision to save memory
            audio, sr = librosa.load(audio_path, sr=16000, dtype=np.float32)
            audio = torch.FloatTensor(audio)
        except:
            try:
                audio, sr = torchaudio.load(audio_path)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    audio = resampler(audio)
                audio = audio.squeeze(0).float()
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                audio = torch.zeros(16000, dtype=torch.float32)
        
        # Aggressive truncation for memory
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            # Minimal padding
            padding = self.max_length - len(audio)
            if padding > 0:
                audio = F.pad(audio, (0, padding))
        
        # Convert text to indices with length limit
        text = sample['text'][:100]  # Limit text length
        text_indices = []
        for char in text:
            if char in self.char2idx:
                text_indices.append(self.char2idx[char])
            else:
                text_indices.append(self.char2idx.get('<unk>', 4))
        
        # Limit text length for memory
        if len(text_indices) > 50:
            text_indices = text_indices[:50]
        
        return {
            'audio': audio,
            'text': torch.LongTensor(text_indices),
            'text_length': len(text_indices),
            'audio_length': len(audio)
        }

def memory_optimized_collate_fn(batch):
    """Memory-optimized collate function"""
    # Sort by audio length and limit batch diversity
    batch = sorted(batch, key=lambda x: x['audio_length'])
    
    # Get reasonable max lengths (not the absolute max)
    audio_lengths = [item['audio_length'] for item in batch]
    text_lengths = [item['text_length'] for item in batch]
    
    # Use 95th percentile instead of max to save memory
    max_audio_len = int(np.percentile(audio_lengths, 95))
    max_text_len = min(max(text_lengths), 50)  # Cap at 50 tokens
    
    audios = []
    texts = []
    audio_lengths_out = []
    text_lengths_out = []
    
    for item in batch:
        # Truncate audio if needed
        audio = item['audio']
        if len(audio) > max_audio_len:
            audio = audio[:max_audio_len]
        elif len(audio) < max_audio_len:
            audio = F.pad(audio, (0, max_audio_len - len(audio)))
        
        audios.append(audio)
        audio_lengths_out.append(min(item['audio_length'], max_audio_len))
        
        # Truncate text if needed
        text = item['text']
        if len(text) > max_text_len:
            text = text[:max_text_len]
        elif len(text) < max_text_len:
            text = F.pad(text, (0, max_text_len - len(text)))
        
        texts.append(text)
        text_lengths_out.append(min(item['text_length'], max_text_len))
    
    return {
        'audio': torch.stack(audios),
        'text': torch.stack(texts),
        'audio_lengths': torch.LongTensor(audio_lengths_out),
        'text_lengths': torch.LongTensor(text_lengths_out)
    }

def compute_mel_features_optimized(audio, n_mels=80):
    """Memory-optimized mel feature computation"""
    # Use smaller FFT for memory efficiency
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=n_mels,
        n_fft=512,  # Reduced from 400
        hop_length=256,  # Increased hop length
        win_length=512
    ).to(audio.device)
    
    with torch.no_grad():
        mel_spec = mel_transform(audio)
        mel_spec = torch.log(mel_spec + 1e-8)
    
    # Limit sequence length for memory
    max_len = MEMORY_OPTIMIZED_SETTINGS['max_mel_length']
    if mel_spec.size(-1) > max_len:
        mel_spec = mel_spec[..., :max_len]
    
    return mel_spec.transpose(1, 2)  # (batch, time, features)

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def train_epoch_optimized(model, train_loader, optimizer, device, scaler, grad_accumulation_steps=8):
    """Memory-optimized training epoch"""
    model.train()
    total_loss = 0.0
    total_ctc_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        audio = batch['audio'].to(device, non_blocking=True)
        text = batch['text'].to(device, non_blocking=True)
        audio_lengths = batch['audio_lengths'].to(device, non_blocking=True)
        text_lengths = batch['text_lengths'].to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu'):
            # Compute mel features
            mel_features = compute_mel_features_optimized(audio)
            
            # Forward pass with gradient checkpointing
            if hasattr(model, 'gradient_checkpointing') and model.gradient_checkpointing:
                outputs = torch.utils.checkpoint.checkpoint(
                    model, mel_features, audio_lengths, text[:, :-1]
                )
            else:
                outputs = model(mel_features, audio_lengths, text[:, :-1])
            
            # CTC loss computation
            if isinstance(outputs, dict):
                encoder_outputs = outputs.get('encoder_outputs', outputs.get('ctc_logits'))
            elif isinstance(outputs, tuple):
                # Handle tuple output (encoder_outputs, decoder_outputs)
                encoder_outputs = outputs[0] if len(outputs) > 0 else None
            else:
                encoder_outputs = outputs
            
            if encoder_outputs is not None and hasattr(encoder_outputs, 'log_softmax'):
                log_probs = F.log_softmax(encoder_outputs, dim=-1)
                input_lengths = torch.full(
                    (audio.size(0),), 
                    log_probs.size(1), 
                    dtype=torch.long, 
                    device=device
                )
                
                ctc_loss = F.ctc_loss(
                    log_probs.transpose(0, 1),
                    text,
                    input_lengths,
                    text_lengths,
                    blank=1,
                    reduction='mean',
                    zero_infinity=True  # Handle infinite losses
                )
            else:
                ctc_loss = torch.tensor(0.0, device=device)
            
            # Scale loss for gradient accumulation
            loss = ctc_loss / grad_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Clear memory periodically
            if (batch_idx + 1) % (grad_accumulation_steps * 4) == 0:
                clear_memory()
        
        total_loss += loss.item() * grad_accumulation_steps
        total_ctc_loss += ctc_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{ctc_loss.item():.4f}',
            'mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A'
        })
        
        # Delete tensors to free memory
        del audio, text, mel_features, outputs, ctc_loss, loss
        
    return total_loss / num_batches, total_ctc_loss / num_batches

def validate_epoch_optimized(model, val_loader, device):
    """Memory-optimized validation epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        
        for batch in pbar:
            audio = batch['audio'].to(device, non_blocking=True)
            text = batch['text'].to(device, non_blocking=True)
            audio_lengths = batch['audio_lengths'].to(device, non_blocking=True)
            text_lengths = batch['text_lengths'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu'):
                mel_features = compute_mel_features_optimized(audio)
                outputs = model(mel_features, audio_lengths, text[:, :-1])
                
            if isinstance(outputs, dict):
                encoder_outputs = outputs.get('encoder_outputs', outputs.get('ctc_logits'))
            elif isinstance(outputs, tuple):
                encoder_outputs = outputs[0] if len(outputs) > 0 else None
            else:
                encoder_outputs = outputs
            
            if encoder_outputs is not None and hasattr(encoder_outputs, 'log_softmax'):
                log_probs = F.log_softmax(encoder_outputs, dim=-1)
                input_lengths = torch.full(
                    (audio.size(0),), 
                    log_probs.size(1), 
                    dtype=torch.long, 
                    device=device
                )
                
                ctc_loss = F.ctc_loss(
                    log_probs.transpose(0, 1),
                    text,
                    input_lengths,
                    text_lengths,
                    blank=1,
                    reduction='mean',
                    zero_infinity=True
                )
            else:
                ctc_loss = torch.tensor(0.0, device=device)
            
            total_loss += ctc_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'val_loss': f'{ctc_loss.item():.4f}',
                'mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A'
            })
            
            # Clean up
            del audio, text, mel_features, outputs, ctc_loss
            
        clear_memory()
    
    return total_loss / num_batches

def main():
    parser = argparse.ArgumentParser(description='RTX 3060 Optimized Fine-tuning')
    parser.add_argument('--checkpoint', type=str, default='best_model (1).pt',
                       help='Path to checkpoint file')
    parser.add_argument('--train_manifest', type=str, default='data/konkani-10k/train_manifest.json',
                       help='Path to training manifest')
    parser.add_argument('--val_manifest', type=str, default='data/konkani-10k/val_manifest.json',
                       help='Path to validation manifest')
    parser.add_argument('--vocab_file', type=str, default='data/konkani-10k/vocab.json',
                       help='Path to vocabulary file')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.00003,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='rtx3060_finetuned',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Setup device and memory
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        # Set memory fraction for RTX 3060
        torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Import model
    try:
        sys.path.append('archives/kaggle_minimal')
        from models.konkanivani_asr import create_konkanivani_model, KonkaniVaniASR
        
        # Get model config
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        # Create model matching checkpoint architecture first
        model = create_konkanivani_model(
            vocab_size=model_config.get('vocab_size', 81),
            config={
                'input_dim': model_config.get('input_dim', 80),
                'd_model': model_config.get('d_model', 256),
                'encoder_layers': model_config.get('encoder_layers', 12),
                'decoder_layers': model_config.get('decoder_layers', 6),
                'num_heads': model_config.get('num_heads', 4),
                'dropout': model_config.get('dropout', 0.1),
                'conv_kernel_size': model_config.get('conv_kernel_size', 31)
            }
        )
        
        # Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing'):
            model.gradient_checkpointing = True
        
        # Load weights with error handling
        if 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("✓ Model weights loaded")
            except Exception as e:
                print(f"Warning: Could not load all weights: {e}")
        
        model = model.to(device)
        
    except ImportError as e:
        print(f"Error importing model: {e}")
        return
    
    # Create datasets
    print("Creating memory-optimized datasets...")
    train_dataset = MemoryOptimizedAudioDataset(args.train_manifest, args.vocab_file)
    val_dataset = MemoryOptimizedAudioDataset(args.val_manifest, args.vocab_file)
    
    # Create data loaders with memory optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=MEMORY_OPTIMIZED_SETTINGS['batch_size'],
        shuffle=True,
        collate_fn=memory_optimized_collate_fn,
        num_workers=MEMORY_OPTIMIZED_SETTINGS['num_workers'],
        pin_memory=MEMORY_OPTIMIZED_SETTINGS['pin_memory'],
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=MEMORY_OPTIMIZED_SETTINGS['batch_size'],
        shuffle=False,
        collate_fn=memory_optimized_collate_fn,
        num_workers=MEMORY_OPTIMIZED_SETTINGS['num_workers'],
        pin_memory=MEMORY_OPTIMIZED_SETTINGS['pin_memory']
    )
    
    # Create optimizer with memory-efficient settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        eps=1e-6  # Smaller epsilon for stability
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader) // MEMORY_OPTIMIZED_SETTINGS['gradient_accumulation_steps']
    )
    
    # Mixed precision scaler (fix deprecation warning)
    if device == 'cuda':
        try:
            from torch.amp import GradScaler
            scaler = GradScaler('cuda')
        except ImportError:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
    else:
        scaler = None
    
    print(f"""
=================================================================
RTX 3060 OPTIMIZED FINE-TUNING
=================================================================
Model: Konkani ASR (Memory Optimized)
Epochs: {args.epochs}
Batch size: {MEMORY_OPTIMIZED_SETTINGS['batch_size']} (effective: {MEMORY_OPTIMIZED_SETTINGS['batch_size'] * MEMORY_OPTIMIZED_SETTINGS['gradient_accumulation_steps']})
Learning rate: {args.learning_rate}
Device: {device}
Mixed precision: {MEMORY_OPTIMIZED_SETTINGS['use_mixed_precision']}
Gradient accumulation: {MEMORY_OPTIMIZED_SETTINGS['gradient_accumulation_steps']}
=================================================================
""")
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_ctc_loss = train_epoch_optimized(
            model, train_loader, optimizer, device, scaler,
            MEMORY_OPTIMIZED_SETTINGS['gradient_accumulation_steps']
        )
        
        # Validate
        val_loss = validate_epoch_optimized(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            
            best_model_path = output_dir / 'best_model_rtx3060.pt'
            torch.save(checkpoint_data, best_model_path)
            print(f"✓ Saved new best model with val_loss: {val_loss:.4f}")
        
        # Clear memory after each epoch
        clear_memory()
    
    print(f"\nFine-tuning complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved in: {output_dir}")

if __name__ == "__main__":
    main()