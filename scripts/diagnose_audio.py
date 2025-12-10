#!/usr/bin/env python3
"""
Diagnose audio file format and content issues
"""
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import json
import sys

def diagnose_audio_file(audio_path):
    """Comprehensive audio file diagnosis"""
    print(f"\n{'='*70}")
    print(f"DIAGNOSING: {Path(audio_path).name}")
    print('='*70)
    
    try:
        # Method 1: soundfile (raw file info)
        print("\n[1] SOUNDFILE INFO:")
        info = sf.info(audio_path)
        print(f"  Sample rate: {info.samplerate} Hz")
        print(f"  Channels: {info.channels}")
        print(f"  Duration: {info.duration:.2f} seconds")
        print(f"  Frames: {info.frames}")
        print(f"  Format: {info.format}")
        print(f"  Subtype: {info.subtype}")
        
        # Method 2: Load with soundfile
        print("\n[2] LOADING WITH SOUNDFILE:")
        data_sf, sr_sf = sf.read(audio_path)
        print(f"  Shape: {data_sf.shape}")
        print(f"  Dtype: {data_sf.dtype}")
        print(f"  Min value: {data_sf.min():.6f}")
        print(f"  Max value: {data_sf.max():.6f}")
        print(f"  Mean: {data_sf.mean():.6f}")
        print(f"  Std: {data_sf.std():.6f}")
        
        # Check if audio is silent
        if np.abs(data_sf).max() < 0.001:
            print("  ⚠️  WARNING: Audio appears to be silent or very quiet!")
        
        # Check for clipping
        if np.abs(data_sf).max() > 0.99:
            print("  ⚠️  WARNING: Audio may be clipped!")
        
        # Method 3: Load with librosa
        print("\n[3] LOADING WITH LIBROSA (16kHz):")
        data_lr, sr_lr = librosa.load(audio_path, sr=16000)
        print(f"  Shape: {data_lr.shape}")
        print(f"  Dtype: {data_lr.dtype}")
        print(f"  Sample rate: {sr_lr} Hz")
        print(f"  Duration: {len(data_lr)/sr_lr:.2f} seconds")
        print(f"  Min value: {data_lr.min():.6f}")
        print(f"  Max value: {data_lr.max():.6f}")
        print(f"  Mean: {data_lr.mean():.6f}")
        print(f"  Std: {data_lr.std():.6f}")
        print(f"  RMS energy: {np.sqrt(np.mean(data_lr**2)):.6f}")
        
        # Check if resampling changed things significantly
        if sr_sf != 16000:
            print(f"  ℹ️  Audio resampled from {sr_sf}Hz to 16000Hz")
        
        # Method 4: Compute mel spectrogram
        print("\n[4] MEL SPECTROGRAM FEATURES:")
        mel_spec = librosa.feature.melspectrogram(
            y=data_lr,
            sr=sr_lr,
            n_mels=80,
            n_fft=400,
            hop_length=160
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        print(f"  Shape: {mel_spec.shape}")
        print(f"  Min (dB): {mel_spec_db.min():.2f}")
        print(f"  Max (dB): {mel_spec_db.max():.2f}")
        print(f"  Mean (dB): {mel_spec_db.mean():.2f}")
        print(f"  Std (dB): {mel_spec_db.std():.2f}")
        
        # Check if features are reasonable
        if mel_spec_db.max() - mel_spec_db.min() < 10:
            print("  ⚠️  WARNING: Very low dynamic range in features!")
        
        # Method 5: Check with AudioProcessor
        print("\n[5] USING PROJECT'S AUDIOPROCESSOR:")
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.audio_processing.audio_processor import AudioProcessor
        
        processor = AudioProcessor(sample_rate=16000, n_mels=80, n_fft=400, hop_length=160)
        waveform = processor.load_audio(audio_path)
        features = processor.compute_features(waveform, apply_augment=False)
        
        print(f"  Waveform shape: {waveform.shape}")
        print(f"  Features shape: {features.shape}")
        print(f"  Features min: {features.min():.4f}")
        print(f"  Features max: {features.max():.4f}")
        print(f"  Features mean: {features.mean():.4f}")
        print(f"  Features std: {features.std():.4f}")
        
        # Summary
        print("\n[6] DIAGNOSIS SUMMARY:")
        issues = []
        
        if np.abs(data_lr).max() < 0.001:
            issues.append("❌ Audio is silent or extremely quiet")
        elif np.abs(data_lr).max() < 0.01:
            issues.append("⚠️  Audio is very quiet (may need normalization)")
        
        if mel_spec_db.max() - mel_spec_db.min() < 10:
            issues.append("❌ Features have very low dynamic range")
        
        if features.std() < 0.1:
            issues.append("⚠️  Features have low variance")
        
        if len(issues) == 0:
            print("  ✅ Audio file appears to be valid and processable")
        else:
            print("  Issues found:")
            for issue in issues:
                print(f"    {issue}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Test files from manifest
    manifest_path = 'data/konkani-asr-v0/splits/manifests/test.json'
    
    print("="*70)
    print("AUDIO FILE DIAGNOSIS")
    print("="*70)
    
    if Path(manifest_path).exists():
        print(f"\nLoading test samples from: {manifest_path}")
        with open(manifest_path, 'r') as f:
            samples = [json.loads(line) for line in f][:3]
        
        for sample in samples:
            audio_path = sample['audio_filepath']
            if Path(audio_path).exists():
                diagnose_audio_file(audio_path)
            else:
                print(f"\n❌ File not found: {audio_path}")
    else:
        print(f"\n❌ Manifest not found: {manifest_path}")
        
        # Try to find any audio files
        print("\nSearching for audio files...")
        audio_files = list(Path('data').glob('**/*.wav'))[:3]
        if audio_files:
            for audio_file in audio_files:
                diagnose_audio_file(audio_file)
        else:
            print("No audio files found!")


if __name__ == '__main__':
    main()
