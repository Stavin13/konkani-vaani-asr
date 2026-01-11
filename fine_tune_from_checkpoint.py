#!/usr/bin/env python3
"""
Complete fine-tuning script that resumes from checkpoint and continues training
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchaudio
import json
import yaml
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

# Add current directory to path for imports
sys.path.insert(0, '.')

class AudioDataset(Dataset):
    """Simple audio dataset for ASR training"""
    
    def __init__(self, manifest_path, vocab_path, max_length=16000*10):
        self.manifest_path = manifest_path
        self.max_length = max_length
        
        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.char2idx = vocab_data['char2idx']
        self.idx2char = vocab_data['idx2char']
        self.vocab_size = len(self.char2idx)
        
        # Load manifest
        self.samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        sample = json.loads(line.strip())
                        self.samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Loaded {len(self.samples)} samples from {manifest_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        audio_path = sample['audio_filepath']
        try:
            # Try with librosa first
            audio, sr = librosa.load(audio_path, sr=16000)
            audio = torch.FloatTensor(audio)
        except:
            try:
                # Fallback to torchaudio
                audio, sr = torchaudio.load(audio_path)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    audio = resampler(audio)
                audio = audio.squeeze(0)  # Remove channel dimension
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                # Return silence as fallback
                audio = torch.zeros(16000)
        
        # Truncate or pad audio
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            padding = self.max_length - len(audio)
            audio = F.pad(audio, (0, padding))
        
        # Convert text to indices
        text = sample['text']
        text_indices = []
        for char in text:
            if char in self.char2idx:
                text_indices.append(self.char2idx[char])
            else:
                text_indices.append(self.char2idx.get('<unk>', 0))
        
        return {
            'audio': audio,
            'text': torch.LongTensor(text_indices),
            'text_length': len(text_indices),
            'audio_length': len(audio)
        }

def collate_fn(batch):
    """Collate function for DataLoader"""
    # Sort by audio length (descending)
    batch = sorted(batch, key=lambda x: x['audio_length'], reverse=True)
    
    # Get max lengths
    max_audio_len = max(item['audio_length'] for item in batch)
    max_text_len = max(item['text_length'] for item in batch)
    
    # Pad sequences
    audios = []
    texts = []
    audio_lengths = []
    text_lengths = []
    
    for item in batch:
        # Pad audio
        audio = item['audio']
        if len(audio) < max_audio_len:
            audio = F.pad(audio, (0, max_audio_len - len(audio)))
        audios.append(audio)
        audio_lengths.append(item['audio_length'])
        
        # Pad text
        text = item['text']
        if len(text) < max_text_len:
            text = F.pad(text, (0, max_text_len - len(text)))
        texts.append(text)
        text_lengths.append(item['text_length'])
    
    return {
        'audio': torch.stack(audios),
        'text': torch.stack(texts),
        'audio_lengths': torch.LongTensor(audio_lengths),
        'text_lengths': torch.LongTensor(text_lengths)
    }

def compute_mel_features(audio, n_mels=80):
    """Compute mel-scale features from audio"""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=n_mels,
        n_fft=400,
        hop_length=160
    )
    
    mel_spec = mel_transform(audio)
    mel_spec = torch.log(mel_spec + 1e-8)  # Log mel spectrogram
    
    return mel_spec.transpose(1, 2)  # (batch, time, features)

def train_epoch(model, train_loader, optimizer, criterion, device, scaler=None, grad_clip=5.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_ctc_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch in pbar:
        audio = batch['audio'].to(device)
        text = batch['text'].to(device)
        audio_lengths = batch['audio_lengths'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        
        # Compute mel features
        mel_features = compute_mel_features(audio)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                # Forward pass
                outputs = model(mel_features, text[:, :-1])  # Exclude last token for input
                
                # CTC loss (using encoder outputs)
                encoder_outputs = outputs.get('encoder_outputs', outputs.get('ctc_logits'))
                if encoder_outputs is not None:
                    log_probs = F.log_softmax(encoder_outputs, dim=-1)
                    input_lengths = torch.full((audio.size(0),), log_probs.size(1), dtype=torch.long)
                    ctc_loss = F.ctc_loss(
                        log_probs.transpose(0, 1),  # (T, N, C)
                        text,
                        input_lengths,
                        text_lengths,
                        blank=1,  # Assuming blank token is at index 1
                        reduction='mean'
                    )
                else:
                    ctc_loss = 0
                
                # Decoder loss (if available)
                decoder_outputs = outputs.get('decoder_outputs', outputs.get('logits'))
                if decoder_outputs is not None:
                    # Cross-entropy loss for decoder
                    decoder_loss = F.cross_entropy(
                        decoder_outputs.reshape(-1, decoder_outputs.size(-1)),
                        text[:, 1:].reshape(-1),  # Exclude first token (SOS)
                        ignore_index=0  # Assuming PAD token is at index 0
                    )
                else:
                    decoder_loss = 0
                
                # Combined loss
                total_batch_loss = 0.9 * ctc_loss + 0.1 * decoder_loss
            
            scaler.scale(total_batch_loss).backward()
            
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass without mixed precision
            outputs = model(mel_features, text[:, :-1])
            
            # CTC loss
            encoder_outputs = outputs.get('encoder_outputs', outputs.get('ctc_logits'))
            if encoder_outputs is not None:
                log_probs = F.log_softmax(encoder_outputs, dim=-1)
                input_lengths = torch.full((audio.size(0),), log_probs.size(1), dtype=torch.long)
                ctc_loss = F.ctc_loss(
                    log_probs.transpose(0, 1),
                    text,
                    input_lengths,
                    text_lengths,
                    blank=1,
                    reduction='mean'
                )
            else:
                ctc_loss = 0
            
            # Decoder loss
            decoder_outputs = outputs.get('decoder_outputs', outputs.get('logits'))
            if decoder_outputs is not None:
                decoder_loss = F.cross_entropy(
                    decoder_outputs.reshape(-1, decoder_outputs.size(-1)),
                    text[:, 1:].reshape(-1),
                    ignore_index=0
                )
            else:
                decoder_loss = 0
            
            total_batch_loss = 0.9 * ctc_loss + 0.1 * decoder_loss
            total_batch_loss.backward()
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
        
        total_loss += total_batch_loss.item()
        total_ctc_loss += ctc_loss.item() if isinstance(ctc_loss, torch.Tensor) else ctc_loss
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{total_batch_loss.item():.4f}',
            'ctc': f'{ctc_loss.item() if isinstance(ctc_loss, torch.Tensor) else ctc_loss:.4f}'
        })
    
    return total_loss / num_batches, total_ctc_loss / num_batches

def validate_epoch(model, val_loader, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_ctc_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        
        for batch in pbar:
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            audio_lengths = batch['audio_lengths'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            
            # Compute mel features
            mel_features = compute_mel_features(audio)
            
            # Forward pass
            outputs = model(mel_features, text[:, :-1])
            
            # CTC loss
            encoder_outputs = outputs.get('encoder_outputs', outputs.get('ctc_logits'))
            if encoder_outputs is not None:
                log_probs = F.log_softmax(encoder_outputs, dim=-1)
                input_lengths = torch.full((audio.size(0),), log_probs.size(1), dtype=torch.long)
                ctc_loss = F.ctc_loss(
                    log_probs.transpose(0, 1),
                    text,
                    input_lengths,
                    text_lengths,
                    blank=1,
                    reduction='mean'
                )
            else:
                ctc_loss = 0
            
            # Decoder loss
            decoder_outputs = outputs.get('decoder_outputs', outputs.get('logits'))
            if decoder_outputs is not None:
                decoder_loss = F.cross_entropy(
                    decoder_outputs.reshape(-1, decoder_outputs.size(-1)),
                    text[:, 1:].reshape(-1),
                    ignore_index=0
                )
            else:
                decoder_loss = 0
            
            total_batch_loss = 0.9 * ctc_loss + 0.1 * decoder_loss
            
            total_loss += total_batch_loss.item()
            total_ctc_loss += ctc_loss.item() if isinstance(ctc_loss, torch.Tensor) else ctc_loss
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}',
                'ctc': f'{ctc_loss.item() if isinstance(ctc_loss, torch.Tensor) else ctc_loss:.4f}'
            })
    
    return total_loss / num_batches, total_ctc_loss / num_batches

def main():
    parser = argparse.ArgumentParser(description='Fine-tune ASR model from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--train_manifest', type=str, required=True,
                       help='Path to training manifest')
    parser.add_argument('--val_manifest', type=str, required=True,
                       help='Path to validation manifest')
    parser.add_argument('--vocab_file', type=str, required=True,
                       help='Path to vocabulary file')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of additional epochs')
    parser.add_argument('--learning_rate', type=float, default=0.00005,
                       help='Learning rate for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--output_dir', type=str, default='fine_tuned_model',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Create model (you'll need to implement this based on your model architecture)
    try:
        from models.konkanivani_asr import create_konkanivani_model
        
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        model = create_konkanivani_model(
            vocab_size=model_config.get('vocab_size', 81),
            d_model=model_config.get('d_model', 128),
            encoder_layers=model_config.get('encoder_layers', 8),
            decoder_layers=model_config.get('decoder_layers', 6),
            num_heads=model_config.get('num_heads', 4),
            dropout=model_config.get('dropout', 0.3),
            conv_kernel_size=model_config.get('conv_kernel_size', 31)
        )
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Warning: No model state dict found")
        
        model = model.to(device)
        print("✓ Model loaded successfully")
        
    except ImportError:
        print("Error: Could not import model. Make sure models/konkanivani_asr.py is available")
        return
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = AudioDataset(args.train_manifest, args.vocab_file)
    val_dataset = AudioDataset(args.val_manifest, args.vocab_file)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0001
    )
    
    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state loaded")
        except:
            print("Warning: Could not load optimizer state")
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if device == 'cuda' else None
    
    # Get starting epoch
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    print(f"""
=================================================================
FINE-TUNING STARTED
=================================================================
Starting epoch: {start_epoch}
Additional epochs: {args.epochs}
Learning rate: {args.learning_rate}
Batch size: {args.batch_size}
Device: {device}
Output directory: {output_dir}
=================================================================
""")
    
    # Training loop
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f"\nEpoch {epoch}/{start_epoch + args.epochs - 1}")
        
        # Train
        train_loss, train_ctc_loss = train_epoch(
            model, train_loader, optimizer, None, device, scaler
        )
        
        # Validate
        val_loss, val_ctc_loss = validate_epoch(model, val_loader, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} (CTC: {train_ctc_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (CTC: {val_ctc_loss:.4f})")
        
        # Save checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': checkpoint.get('config', {})
        }
        
        # Save regular checkpoint
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / 'best_model_finetuned.pt'
            torch.save(checkpoint_data, best_model_path)
            print(f"✓ Saved new best model with val_loss: {val_loss:.4f}")
    
    print(f"\nFine-tuning complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved in: {output_dir}")

if __name__ == "__main__":
    main()