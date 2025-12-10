"""
Audio Feature Extraction for KonkaniVani ASR
===========================================
Mel-spectrogram computation with SpecAugment data augmentation
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import soundfile as sf


class AudioProcessor:
    """Process audio files into mel-spectrogram features"""
    
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160,
                 spec_augment=True, time_mask_param=70, freq_mask_param=27):
        """
        Args:
            sample_rate: Target sample rate (16kHz for ASR)
            n_mels: Number of mel filterbanks
            n_fft: FFT size
            hop_length: Hop length for STFT
            spec_augment: Whether to apply SpecAugment
            time_mask_param: Max time mask width
            freq_mask_param: Max frequency mask width
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )
        
        # SpecAugment transforms
        self.spec_augment = spec_augment
        if spec_augment:
            self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
            self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
    
    def load_audio(self, audio_path):
        """
        Load audio file and resample to target sample rate
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            waveform: (1, time) tensor
        """
        # Use soundfile directly to avoid torchcodec dependency
        waveform, sr = sf.read(audio_path)
        waveform = torch.from_numpy(waveform).float()
        
        # Add channel dimension if mono
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            # Convert to mono if stereo
            waveform = waveform.mean(dim=1, keepdim=True).T
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform
    
    def compute_features(self, waveform, apply_augment=False):
        """
        Compute mel-spectrogram features
        
        Args:
            waveform: (1, time) audio tensor
            apply_augment: Whether to apply SpecAugment
        
        Returns:
            features: (time, n_mels) mel-spectrogram
        """
        # Compute mel-spectrogram
        mel_spec = self.mel_transform(waveform)  # (1, n_mels, time)
        
        # Apply SpecAugment if training
        if apply_augment and self.spec_augment:
            mel_spec = self.time_mask(mel_spec)
            mel_spec = self.freq_mask(mel_spec)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Transpose to (time, n_mels)
        mel_spec = mel_spec.squeeze(0).transpose(0, 1)
        
        return mel_spec
    
    def process_audio_file(self, audio_path, apply_augment=False):
        """
        Load and process audio file end-to-end
        
        Args:
            audio_path: Path to audio file
            apply_augment: Whether to apply SpecAugment
        
        Returns:
            features: (time, n_mels) mel-spectrogram
            duration: Audio duration in seconds
        """
        waveform = self.load_audio(audio_path)
        features = self.compute_features(waveform, apply_augment)
        duration = waveform.size(1) / self.sample_rate
        
        return features, duration
    
    def normalize_features(self, features, mean=None, std=None):
        """
        Normalize features (per-feature mean/std normalization)
        
        Args:
            features: (time, n_mels) features
            mean: Optional precomputed mean
            std: Optional precomputed std
        
        Returns:
            normalized: (time, n_mels) normalized features
            mean: Feature means
            std: Feature stds
        """
        if mean is None:
            mean = features.mean(dim=0, keepdim=True)
        if std is None:
            std = features.std(dim=0, keepdim=True)
        
        normalized = (features - mean) / (std + 1e-9)
        
        return normalized, mean, std
    
    @staticmethod
    def compute_dataset_stats(feature_list):
        """
        Compute mean and std across entire dataset
        
        Args:
            feature_list: List of (time, n_mels) feature tensors
        
        Returns:
            mean: (n_mels,) global mean
            std: (n_mels,) global std
        """
        # Concatenate all features
        all_features = torch.cat(feature_list, dim=0)
        
        mean = all_features.mean(dim=0)
        std = all_features.std(dim=0)
        
        return mean, std


def test_audio_processor():
    """Test audio processing pipeline"""
    import matplotlib.pyplot as plt
    
    processor = AudioProcessor()
    
    # Test with a sample audio file
    audio_path = "data/konkani-asr-v0/data/processed_segments_diarized/audio_segments/segment_000000.wav"
    
    try:
        features, duration = processor.process_audio_file(audio_path)
        print(f"✅ Audio processing test passed!")
        print(f"   Audio duration: {duration:.2f}s")
        print(f"   Feature shape: {features.shape}")
        print(f"   Feature range: [{features.min():.2f}, {features.max():.2f}]")
        
        # Visualize
        plt.figure(figsize=(12, 4))
        plt.imshow(features.T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Log Mel Energy')
        plt.xlabel('Time Frames')
        plt.ylabel('Mel Filterbank')
        plt.title('Mel-Spectrogram')
        plt.tight_layout()
        plt.savefig('mel_spectrogram_test.png')
        print(f"   Saved visualization to mel_spectrogram_test.png")
        
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")


if __name__ == "__main__":
    test_audio_processor()
