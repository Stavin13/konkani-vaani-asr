#!/usr/bin/env python3
"""
Test the latest ASR model with audio samples
"""
import torch
import torchaudio
from pathlib import Path
import sys
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import librosa as fallback
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

from models.konkanivani_asr import KonkaniVaniASR
from data.audio_processing.audio_processor import AudioProcessor


def load_model(checkpoint_path):
    """Load ASR model from checkpoint"""
    print(f"\nLoading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Debug: print checkpoint keys
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    
    # Infer model architecture from state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Infer vocab size
    vocab_size = 100  # default
    if 'ctc_head.weight' in state_dict:
        vocab_size = state_dict['ctc_head.weight'].shape[0]
        print(f"  Inferred vocab size: {vocab_size}")
    
    # Infer d_model
    d_model = 256  # default
    if 'encoder.input_proj.weight' in state_dict:
        d_model = state_dict['encoder.input_proj.weight'].shape[0]
        print(f"  Inferred d_model: {d_model}")
    
    # Create model config
    config = {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'encoder_layers': 12,  # default
        'dropout': 0.1
    }
    
    # Create model
    model = KonkaniVaniASR(
        vocab_size=config.get('vocab_size', 100),
        d_model=config.get('hidden_dim', config.get('d_model', 256)),
        encoder_layers=config.get('num_layers', config.get('encoder_layers', 12)),
        dropout=config.get('dropout', 0.1)
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Get vocab
    vocab = checkpoint.get('vocab', None)
    if vocab is None:
        print("⚠️  No vocab in checkpoint, loading from manifest...")
        vocab = load_vocab_from_manifest()
    
    print(f"✓ Model loaded")
    print(f"  Vocab size: {len(vocab) if vocab else 'unknown'}")
    print(f"  D model: {config.get('d_model', config.get('hidden_dim', 'unknown'))}")
    print(f"  Encoder layers: {config.get('encoder_layers', config.get('num_layers', 'unknown'))}")
    
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    
    return model, vocab


def load_vocab_from_manifest():
    """Load vocabulary from training manifest"""
    manifest_path = Path('data/konkani-asr-v0/splits/manifests/train.json')
    
    if not manifest_path.exists():
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


def transcribe_audio(model, audio_path, vocab):
    """Transcribe audio file"""
    # Load audio
    try:
        if HAS_LIBROSA:
            # Use librosa as fallback
            audio_data, sample_rate = librosa.load(str(audio_path), sr=16000)
            waveform = torch.from_numpy(audio_data).unsqueeze(0)  # Add batch dimension
        else:
            # Try torchaudio
            waveform, sample_rate = torchaudio.load(str(audio_path))
    except Exception as e:
        print(f"    Error loading audio: {e}")
        return None
    
    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Extract features (mel spectrogram)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80
    )
    
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(mel_spec + 1e-9)
    
    # Transpose to (batch, time, features)
    mel_spec = mel_spec.transpose(1, 2)
    
    # Forward pass
    with torch.no_grad():
        output = model(mel_spec)
        # Model returns (ctc_logits, attn_logits) - we want CTC logits for inference
        if isinstance(output, tuple):
            logits = output[0]  # CTC logits
        else:
            logits = output
    
    # Decode (greedy)
    predictions = torch.argmax(logits, dim=-1)
    
    # Convert to text
    if vocab:
        reverse_vocab = {v: k for k, v in vocab.items()}
        
        # Remove blanks and duplicates
        text = []
        prev_idx = None
        for idx in predictions[0].tolist():
            if idx != 0 and idx != prev_idx:  # 0 is blank
                char = reverse_vocab.get(idx, '<unk>')
                if char not in ['<blank>', '<unk>']:
                    text.append(char)
            prev_idx = idx
        
        return ''.join(text)
    else:
        return f"Predictions: {predictions[0].tolist()[:50]}..."


def test_model(checkpoint_path):
    """Test ASR model"""
    print("\n" + "="*70)
    print("TESTING ASR MODEL")
    print("="*70)
    
    # Load model
    model, vocab = load_model(checkpoint_path)
    
    # Find test audio files
    test_manifest = Path('data/konkani-asr-v0/splits/manifests/test.json')
    
    if not test_manifest.exists():
        print(f"\n❌ Test manifest not found: {test_manifest}")
        return
    
    # Load test samples
    test_samples = []
    with open(test_manifest, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:  # Only first 10
                break
            data = json.loads(line)
            test_samples.append(data)
    
    print(f"\n✓ Loaded {len(test_samples)} test samples")
    
    print("\n" + "="*70)
    print("SAMPLE TRANSCRIPTIONS")
    print("="*70)
    
    for i, sample in enumerate(test_samples, 1):
        audio_path = Path(sample['audio_filepath'])
        true_text = sample['text']
        
        if not audio_path.exists():
            print(f"\n[{i}] ❌ Audio not found: {audio_path}")
            continue
        
        # Transcribe
        try:
            predicted_text = transcribe_audio(model, audio_path, vocab)
            
            print(f"\n[{i}]")
            print(f"  Audio: {audio_path.name}")
            print(f"  True:  {true_text[:80]}...")
            print(f"  Pred:  {predicted_text[:80]}...")
            
            # Simple accuracy check
            if predicted_text and true_text:
                # Character-level accuracy
                correct = sum(1 for a, b in zip(predicted_text, true_text) if a == b)
                total = max(len(predicted_text), len(true_text))
                accuracy = 100 * correct / total if total > 0 else 0
                print(f"  Char accuracy: {accuracy:.1f}%")
        
        except Exception as e:
            print(f"\n[{i}] ❌ Error: {e}")


def main():
    print("\n" + "="*70)
    print("TEST BEST ASR MODELS")
    print("="*70)
    
    # List of best models to try in order of preference
    best_models = [
        # Latest Kaggle best model
        ('/Volumes/data&proj/konkani/kaggle_downloads/20251210_060024/checkpoints/best_model.pt', 'LATEST KAGGLE BEST MODEL'),
        # Previous best model
        ('kaggle_asr_outputs/checkpoints/best_model.pt', 'PREVIOUS BEST MODEL'),
        # Specific good checkpoints from analysis
        ('kaggle_asr_outputs/checkpoints/checkpoint_epoch_27.pt', 'CHECKPOINT EPOCH 27 (GOOD VAL LOSS)'),
        ('/Volumes/data&proj/konkani/kaggle_downloads/20251210_060024/checkpoints/checkpoint_epoch_15.pt', 'CHECKPOINT EPOCH 15'),
        ('/Volumes/data&proj/konkani/kaggle_downloads/20251210_060024/checkpoints/checkpoint_epoch_25.pt', 'CHECKPOINT EPOCH 25'),
    ]
    
    successful_tests = 0
    
    for checkpoint_path, description in best_models:
        checkpoint = Path(checkpoint_path)
        if checkpoint.exists():
            print("\n" + "="*70)
            print(f"TESTING {description}")
            print("="*70)
            try:
                test_model(checkpoint)
                successful_tests += 1
                # Test multiple models for comparison
                if successful_tests >= 3:
                    break
            except Exception as e:
                print(f"❌ Error testing {description}: {e}")
                continue
        else:
            print(f"\n⚠️  {description} not found: {checkpoint_path}")
    
    if successful_tests == 0:
        print("\n❌ No models could be tested successfully")
    else:
        print(f"\n✓ Successfully tested {successful_tests} model(s)")
    
    print("\n✓ Testing complete boss!")


if __name__ == '__main__':
    main()
