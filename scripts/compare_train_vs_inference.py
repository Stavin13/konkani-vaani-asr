#!/usr/bin/env python3
"""
Compare training vs inference input to find mismatches
"""
import torch
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.audio_processing.audio_processor import AudioProcessor
from models.konkanivani_asr import KonkaniVaniASR


def test_training_pipeline():
    """Simulate training data pipeline"""
    print("="*70)
    print("TRAINING PIPELINE")
    print("="*70)
    
    # Load a sample from training manifest
    manifest_path = 'data/konkani-asr-v0/splits/manifests/train.json'
    with open(manifest_path, 'r') as f:
        sample = json.loads(f.readline())
    
    audio_path = sample['audio_filepath']
    print(f"\nAudio file: {Path(audio_path).name}")
    print(f"Ground truth: {sample['text'][:50]}...")
    
    # Process audio (training style)
    processor = AudioProcessor(sample_rate=16000, n_mels=80, n_fft=400, hop_length=160)
    features, duration = processor.process_audio_file(audio_path, apply_augment=False)
    
    print(f"\nFeatures shape: {features.shape}")
    print(f"Features dtype: {features.dtype}")
    print(f"Features min: {features.min():.4f}")
    print(f"Features max: {features.max():.4f}")
    print(f"Features mean: {features.mean():.4f}")
    print(f"Features std: {features.std():.4f}")
    
    # Add batch dimension (as in training)
    features_batch = features.unsqueeze(0)
    print(f"\nBatched shape: {features_batch.shape}")
    
    return features_batch, audio_path


def test_inference_pipeline(audio_path):
    """Simulate inference pipeline"""
    print("\n" + "="*70)
    print("INFERENCE PIPELINE")
    print("="*70)
    
    # Process audio (inference style - from test_best_model.py)
    processor = AudioProcessor(sample_rate=16000, n_mels=80, n_fft=400, hop_length=160)
    waveform = processor.load_audio(audio_path)
    features = processor.compute_features(waveform, apply_augment=False)
    
    print(f"\nWaveform shape: {waveform.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Features dtype: {features.dtype}")
    print(f"Features min: {features.min():.4f}")
    print(f"Features max: {features.max():.4f}")
    print(f"Features mean: {features.mean():.4f}")
    print(f"Features std: {features.std():.4f}")
    
    # Add batch dimension (as in inference)
    features_batch = features.unsqueeze(0)
    print(f"\nBatched shape: {features_batch.shape}")
    
    return features_batch


def test_model_forward(features, checkpoint_path):
    """Test model forward pass"""
    print("\n" + "="*70)
    print("MODEL FORWARD PASS")
    print("="*70)
    
    # Load model
    device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load vocab
    with open('data/vocab.json', 'r') as f:
        vocab_data = json.load(f)
    
    if 'char2idx' in vocab_data:
        vocab_size = len(vocab_data['char2idx'])
    else:
        vocab_size = len(vocab_data)
    
    # Create model
    model = KonkaniVaniASR(
        vocab_size=vocab_size,
        input_dim=80,
        d_model=256,
        encoder_layers=12,
        decoder_layers=6,
        num_heads=4,
        dropout=0.1
    )
    
    # Load weights
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"\nModel loaded from epoch {checkpoint['epoch']}")
    print(f"Input shape: {features.shape}")
    
    # Forward pass
    with torch.no_grad():
        encoder_out, mask = model.encoder(features)
        ctc_logits = model.ctc_head(encoder_out)
    
    print(f"\nEncoder output shape: {encoder_out.shape}")
    print(f"Encoder output min: {encoder_out.min():.4f}")
    print(f"Encoder output max: {encoder_out.max():.4f}")
    print(f"Encoder output mean: {encoder_out.mean():.4f}")
    print(f"Encoder output std: {encoder_out.std():.4f}")
    
    print(f"\nCTC logits shape: {ctc_logits.shape}")
    print(f"CTC logits min: {ctc_logits.min():.4f}")
    print(f"CTC logits max: {ctc_logits.max():.4f}")
    print(f"CTC logits mean: {ctc_logits.mean():.4f}")
    print(f"CTC logits std: {ctc_logits.std():.4f}")
    
    # Check predictions
    probs = torch.softmax(ctc_logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    
    # Count unique predictions
    unique_preds = torch.unique(preds[0])
    print(f"\nUnique predicted tokens: {len(unique_preds)}")
    print(f"Predicted tokens: {unique_preds[:20].tolist()}")
    
    # Check blank probability
    blank_prob = probs[0, :, 1].mean()  # Assuming blank is index 1
    print(f"\nAverage blank probability: {blank_prob:.4f} ({blank_prob*100:.2f}%)")
    
    # Check if encoder output has reasonable variance
    if encoder_out.std() < 0.1:
        print("\n⚠️  WARNING: Encoder output has very low variance!")
        print("   This suggests the encoder may not be learning properly.")
    
    # Check if CTC logits are too uniform
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
    max_entropy = np.log(vocab_size)
    print(f"\nPrediction entropy: {entropy:.4f} / {max_entropy:.4f}")
    print(f"Entropy ratio: {entropy/max_entropy:.2%}")
    
    if entropy / max_entropy > 0.9:
        print("\n⚠️  WARNING: Predictions are nearly uniform (high entropy)!")
        print("   The model is very uncertain about its predictions.")


def main():
    checkpoint_path = 'kaggle_asr_outputs/checkpoints/checkpoint_epoch_27.pt'
    
    print("\n" + "="*70)
    print("COMPARING TRAINING VS INFERENCE PIPELINES")
    print("="*70)
    
    # Test training pipeline
    features_train, audio_path = test_training_pipeline()
    
    # Test inference pipeline
    features_inference = test_inference_pipeline(audio_path)
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    diff = (features_train - features_inference).abs()
    print(f"\nAbsolute difference:")
    print(f"  Max: {diff.max():.6f}")
    print(f"  Mean: {diff.mean():.6f}")
    print(f"  Std: {diff.std():.6f}")
    
    if diff.max() < 1e-5:
        print("\n✅ Training and inference pipelines produce identical features!")
    else:
        print("\n⚠️  Training and inference pipelines produce different features!")
        print("   This could explain the poor inference performance.")
    
    # Test model with both
    print("\n" + "="*70)
    print("TESTING MODEL WITH TRAINING-STYLE FEATURES")
    test_model_forward(features_train, checkpoint_path)
    
    print("\n" + "="*70)
    print("TESTING MODEL WITH INFERENCE-STYLE FEATURES")
    test_model_forward(features_inference, checkpoint_path)


if __name__ == '__main__':
    main()
