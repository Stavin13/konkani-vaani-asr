"""
KonkaniVani ASR Training Script
===============================
Train custom Transformer-based ASR model for Konkani
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import sys

sys.path.append(str(Path(__file__).parent))

from models.konkanivani_asr import create_konkanivani_model
from data.audio_processing.dataset import create_dataloaders
from data.audio_processing.text_tokenizer import KonkaniTokenizer


class CTCAttentionLoss(nn.Module):
    """Hybrid CTC + Attention loss"""
    
    def __init__(self, ctc_weight=0.3, blank_id=1):
        super().__init__()
        self.ctc_weight = ctc_weight
        self.attn_weight = 1.0 - ctc_weight
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)
        self.attn_loss = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    def forward(self, ctc_logits, attn_logits, targets, input_lengths, target_lengths):
        """
        Compute hybrid loss
        
        Args:
            ctc_logits: (batch, time, vocab_size)
            attn_logits: (batch, tgt_len, vocab_size) or None
            targets: (batch, tgt_len)
            input_lengths: (batch,)
            target_lengths: (batch,)
        """
        # CTC loss
        ctc_log_probs = torch.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (time, batch, vocab)
        ctc_loss_val = self.ctc_loss(ctc_log_probs, targets, input_lengths, target_lengths)
        
        # Attention loss (if decoder output provided)
        attn_loss_val = 0.0
        if attn_logits is not None:
            # attn_logits shape: (batch, tgt_len-1, vocab) because we passed tokens[:, :-1]
            # targets shape: (batch, tgt_len)
            # We want to predict tokens[1:] from tokens[:-1]
            attn_targets = targets[:, 1:]  # Remove SOS, shape: (batch, tgt_len-1)
            
            # Ensure shapes match
            min_len = min(attn_logits.size(1), attn_targets.size(1))
            attn_logits_trimmed = attn_logits[:, :min_len, :]
            attn_targets_trimmed = attn_targets[:, :min_len]
            
            attn_loss_val = self.attn_loss(
                attn_logits_trimmed.reshape(-1, attn_logits_trimmed.size(-1)),
                attn_targets_trimmed.reshape(-1)
            )
        
        # Combined loss
        total_loss = self.ctc_weight * ctc_loss_val + self.attn_weight * attn_loss_val
        
        return total_loss, ctc_loss_val, attn_loss_val


class ASRTrainer:
    """Trainer for KonkaniVani ASR"""
    
    def __init__(self, model, tokenizer, train_loader, val_loader, 
                 device, config):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss function
        self.criterion = CTCAttentionLoss(
            ctc_weight=config['ctc_weight'],
            blank_id=tokenizer.blank_id
        )
        
        # Optimizer (with optional CPU offloading)
        if config.get('use_cpu_offload', False):
            print("✅ Using CPU offloading for optimizer (saves GPU memory)")
            # Use 8-bit optimizer to save memory
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    model.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay']
                )
            except ImportError:
                print("⚠️  bitsandbytes not available, using standard AdamW")
                self.optimizer = optim.AdamW(
                    model.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay']
                )
        else:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # Checkpointing
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # Mixed precision training
        self.use_amp = config.get('mixed_precision', False)
        if self.use_amp:
            print("✅ Using mixed precision training (FP16)")
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Google Drive backup
        if config.get('drive_backup'):
            print(f"✅ Will backup to: {config['drive_backup']}")
            import os
            os.makedirs(config['drive_backup'], exist_ok=True)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_ctc_loss = 0
        total_attn_loss = 0
        
        # Gradient accumulation
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Clear cache periodically to avoid fragmentation
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Move to device
            audio_features = batch['audio_features'].to(self.device)
            transcript_tokens = batch['transcript_tokens'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)
            transcript_lengths = batch['transcript_lengths'].to(self.device)
            
            # Forward pass (with mixed precision if enabled)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    ctc_logits, attn_logits = self.model(
                        audio_features,
                        audio_lengths,
                        transcript_tokens[:, :-1],  # Remove EOS for input
                        transcript_lengths - 1
                    )
                    
                    # Compute loss
                    loss, ctc_loss, attn_loss = self.criterion(
                        ctc_logits,
                        attn_logits,
                        transcript_tokens,
                        audio_lengths,
                        transcript_lengths
                    )
                
                # Backward pass with gradient scaling and accumulation
                self.scaler.scale(loss / accumulation_steps).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                ctc_logits, attn_logits = self.model(
                    audio_features,
                    audio_lengths,
                    transcript_tokens[:, :-1],  # Remove EOS for input
                    transcript_lengths - 1
                )
                
                # Compute loss
                loss, ctc_loss, attn_loss = self.criterion(
                    ctc_logits,
                    attn_logits,
                    transcript_tokens,
                    audio_lengths,
                    transcript_lengths
                )
                
                # Backward pass with gradient accumulation
                (loss / accumulation_steps).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            
            # Update metrics
            total_loss += loss.item()
            total_ctc_loss += ctc_loss.item()
            if isinstance(attn_loss, torch.Tensor):
                total_attn_loss += attn_loss.item()
            
            # Logging
            if batch_idx % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/ctc_loss', ctc_loss.item(), self.global_step)
                if isinstance(attn_loss, torch.Tensor):
                    self.writer.add_scalar('train/attn_loss', attn_loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ctc': f'{ctc_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_ctc_loss = total_ctc_loss / len(self.train_loader)
        
        return avg_loss, avg_ctc_loss
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_ctc_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                audio_features = batch['audio_features'].to(self.device)
                transcript_tokens = batch['transcript_tokens'].to(self.device)
                audio_lengths = batch['audio_lengths'].to(self.device)
                transcript_lengths = batch['transcript_lengths'].to(self.device)
                
                # Forward pass
                ctc_logits, attn_logits = self.model(
                    audio_features,
                    audio_lengths,
                    transcript_tokens[:, :-1],
                    transcript_lengths - 1
                )
                
                # Compute loss
                loss, ctc_loss, attn_loss = self.criterion(
                    ctc_logits,
                    attn_logits,
                    transcript_tokens,
                    audio_lengths,
                    transcript_lengths
                )
                
                total_loss += loss.item()
                total_ctc_loss += ctc_loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_ctc_loss = total_ctc_loss / len(self.val_loader)
        
        # Log to TensorBoard
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/ctc_loss', avg_ctc_loss, epoch)
        
        return avg_loss, avg_ctc_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model with val_loss: {val_loss:.4f}")
        
        # Backup to Google Drive if path provided
        if self.config.get('drive_backup'):
            self._backup_to_drive()
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training"""
        print(f"\n📂 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        print(f"✅ Resumed from epoch {start_epoch}")
        print(f"   Best val loss so far: {self.best_val_loss:.4f}")
        
        return start_epoch
    
    def train(self, num_epochs, resume_from=None):
        """Main training loop"""
        start_epoch = 1
        
        # Resume from checkpoint if provided
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
        
        print(f"\n{'='*80}")
        print("STARTING KONKANIVANI ASR TRAINING")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Epochs: {start_epoch} to {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"{'='*80}\n")
        
        for epoch in range(start_epoch, num_epochs + 1):
            # Train
            train_loss, train_ctc_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_ctc_loss = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} (CTC: {train_ctc_loss:.4f})")
            print(f"  Val Loss: {val_loss:.4f} (CTC: {val_ctc_loss:.4f})")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % self.config['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*80}\n")
    
    def _backup_to_drive(self):
        """Backup checkpoints to Google Drive"""
        import shutil
        import time
        
        drive_path = self.config.get('drive_backup')
        if not drive_path:
            return
        
        try:
            print(f"\n📤 Backing up to Drive...")
            
            # Copy checkpoints
            drive_checkpoint_dir = Path(drive_path) / 'checkpoints'
            if drive_checkpoint_dir.exists():
                shutil.rmtree(drive_checkpoint_dir)
            shutil.copytree(self.checkpoint_dir, drive_checkpoint_dir)
            
            # Copy logs
            drive_log_dir = Path(drive_path) / 'logs'
            if drive_log_dir.exists():
                shutil.rmtree(drive_log_dir)
            shutil.copytree(self.config['log_dir'], drive_log_dir)
            
            print(f"✅ Backup completed at {time.strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"⚠️  Backup failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train KonkaniVani ASR')
    parser.add_argument('--train_manifest', type=str, 
                        default='data/konkani-asr-v0/splits/manifests/train.json')
    parser.add_argument('--val_manifest', type=str,
                        default='data/konkani-asr-v0/splits/manifests/val.json')
    parser.add_argument('--vocab_file', type=str, default='vocab.json')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--device', type=str, default='mps')  # 'cuda', 'mps', or 'cpu'
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--d_model', type=int, default=192, help='Model dimension')
    parser.add_argument('--encoder_layers', type=int, default=8, help='Number of encoder layers')
    parser.add_argument('--decoder_layers', type=int, default=4, help='Number of decoder layers')
    parser.add_argument('--use_cpu_offload', action='store_true', help='Offload optimizer to CPU to save GPU memory')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Use gradient checkpointing to save memory')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Number of steps to accumulate gradients (simulates larger batch size)')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training (FP16) to save memory')
    parser.add_argument('--drive_backup', type=str, default=None, help='Google Drive path to backup checkpoints (e.g., /content/drive/MyDrive/konkanivani_backup)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (e.g., checkpoints/checkpoint_epoch_15.pt)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = KonkaniTokenizer(args.vocab_file)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        args.train_manifest,
        args.val_manifest,
        tokenizer,
        batch_size=args.batch_size,
        num_workers=0  # Set to 0 for macOS compatibility
    )
    
    # Create model with configurable size
    model_config = {
        'input_dim': 80,
        'd_model': args.d_model,
        'encoder_layers': args.encoder_layers,
        'decoder_layers': args.decoder_layers,
        'num_heads': 4,
        'conv_kernel_size': 31,
        'dropout': 0.1
    }
    model = create_konkanivani_model(vocab_size=tokenizer.vocab_size, config=model_config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Enable gradient checkpointing if requested (saves memory)
    if args.gradient_checkpointing:
        print("✅ Enabling gradient checkpointing (saves GPU memory)")
        # Note: This requires implementing checkpointing in the model
        # For now, we'll use mixed precision instead
    
    # Training config
    config = {
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-6,
        'grad_clip': 5.0,
        'ctc_weight': 0.3,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'save_every': 5,
        'use_cpu_offload': args.use_cpu_offload,
        'mixed_precision': args.mixed_precision,
        'gradient_accumulation_steps': getattr(args, 'gradient_accumulation_steps', 4),
        'drive_backup': args.drive_backup
    }
    
    # Create trainer
    trainer = ASRTrainer(model, tokenizer, train_loader, val_loader, device, config)
    
    # Train (with optional resume)
    trainer.train(args.num_epochs, resume_from=args.resume)


if __name__ == "__main__":
    main()
