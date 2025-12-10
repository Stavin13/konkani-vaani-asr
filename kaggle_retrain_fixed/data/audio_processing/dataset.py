"""
PyTorch Dataset for KonkaniVani ASR
==================================
Load audio and transcripts from manifests for training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.audio_processing.audio_processor import AudioProcessor
from data.audio_processing.text_tokenizer import KonkaniTokenizer


class KonkaniASRDataset(Dataset):
    """Dataset for Konkani ASR training"""
    
    def __init__(self, manifest_file, tokenizer, audio_processor, 
                 apply_augment=False, max_duration=30.0):
        """
        Args:
            manifest_file: Path to JSONL manifest file
            tokenizer: KonkaniTokenizer instance
            audio_processor: AudioProcessor instance
            apply_augment: Whether to apply SpecAugment
            max_duration: Maximum audio duration in seconds
        """
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.apply_augment = apply_augment
        self.max_duration = max_duration
        
        # Load manifest
        self.samples = []
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # Filter by duration
                if data.get('duration', 0) <= max_duration:
                    self.samples.append(data)
        
        print(f"Loaded {len(self.samples)} samples from {manifest_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            audio_features: (time, n_mels) mel-spectrogram
            transcript: Konkani text
            audio_length: Actual audio length
            transcript_length: Actual transcript length
        """
        sample = self.samples[idx]
        
        # Load and process audio
        audio_path = sample['audio_filepath']
        audio_features, _ = self.audio_processor.process_audio_file(
            audio_path, 
            apply_augment=self.apply_augment
        )
        
        # Encode transcript
        transcript = sample['text']
        transcript_tokens = self.tokenizer.encode(transcript, add_sos_eos=True)
        
        return {
            'audio_features': audio_features,
            'transcript_tokens': torch.tensor(transcript_tokens, dtype=torch.long),
            'audio_length': audio_features.size(0),
            'transcript_length': len(transcript_tokens),
            'transcript_text': transcript  # For debugging
        }


def collate_fn(batch, pad_id=0):
    """
    Collate function for batching variable-length sequences
    
    Args:
        batch: List of samples from dataset
        pad_id: Padding token ID
    
    Returns:
        Batched tensors
    """
    # Sort by audio length (descending) for efficient packing
    batch = sorted(batch, key=lambda x: x['audio_length'], reverse=True)
    
    # Extract components
    audio_features = [item['audio_features'] for item in batch]
    transcript_tokens = [item['transcript_tokens'] for item in batch]
    audio_lengths = torch.tensor([item['audio_length'] for item in batch], dtype=torch.long)
    transcript_lengths = torch.tensor([item['transcript_length'] for item in batch], dtype=torch.long)
    
    # Pad sequences
    audio_features_padded = pad_sequence(audio_features, batch_first=True, padding_value=0.0)
    transcript_tokens_padded = pad_sequence(transcript_tokens, batch_first=True, padding_value=pad_id)
    
    return {
        'audio_features': audio_features_padded,
        'transcript_tokens': transcript_tokens_padded,
        'audio_lengths': audio_lengths,
        'transcript_lengths': transcript_lengths
    }


def create_dataloaders(train_manifest, val_manifest, tokenizer, 
                       batch_size=16, num_workers=4):
    """
    Create train and validation dataloaders
    
    Args:
        train_manifest: Path to training manifest
        val_manifest: Path to validation manifest
        tokenizer: KonkaniTokenizer instance
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader
    """
    # Create audio processor
    audio_processor = AudioProcessor(
        sample_rate=16000,
        n_mels=80,
        spec_augment=True
    )
    
    # Create datasets
    train_dataset = KonkaniASRDataset(
        train_manifest,
        tokenizer,
        audio_processor,
        apply_augment=True  # Apply augmentation for training
    )
    
    val_dataset = KonkaniASRDataset(
        val_manifest,
        tokenizer,
        audio_processor,
        apply_augment=False  # No augmentation for validation
    )
    
    # Create dataloaders with picklable collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, pad_id=tokenizer.pad_id)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, pad_id=tokenizer.pad_id)
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    from text_tokenizer import KonkaniTokenizer
    
    # Load tokenizer
    tokenizer = KonkaniTokenizer('../../vocab.json')
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_manifest='../../data/konkani-asr-v0/splits/manifests/train.json',
        val_manifest='../../data/konkani-asr-v0/splits/manifests/val.json',
        tokenizer=tokenizer,
        batch_size=4,
        num_workers=0
    )
    
    # Test batch
    print("\n" + "="*60)
    print("TESTING DATALOADER")
    print("="*60)
    
    for batch in train_loader:
        print(f"Audio features shape: {batch['audio_features'].shape}")
        print(f"Transcript tokens shape: {batch['transcript_tokens'].shape}")
        print(f"Audio lengths: {batch['audio_lengths']}")
        print(f"Transcript lengths: {batch['transcript_lengths']}")
        break
    
    print("\n✅ Dataset test passed!")
