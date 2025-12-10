#!/usr/bin/env python3
"""
Test ASR model with direct CTC logits inspection
"""
import torch
import json
import sys
from pathlib import Path
import argparse
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkanivani_asr import KonkaniVaniASR
from data.audio_processing.audio_processor import AudioProcessor


def load_model(checkpoint_path, vocab_path='data/vocab.json'):
    """Load model and vocab"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vocabulary
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    if 'char2idx' in vocab_data:
        char_to_idx = vocab_data['char2idx']
        idx_to_char = {v: k for k, v in char_to_idx.items()}
        vocab_size = len(char_to_idx)
    else:
        idx_to_char = {int(k): v for k, v in vocab_data.items()}
        char_to_idx = {v: int(k) for k, v in vocab_data.items()}
        vocab_size = len(idx_to_char)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    # Load weights (handle DataParallel)
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, idx_to_char, char_to_idx, device


def decode_ctc_greedy(logits, idx_to_char, blank_id=1):
    """
    Greedy CTC decoding with proper blank handling
    
    Args:
        logits: (batch, time, vocab_size) - raw logits from CTC head
        idx_to_char: dict mapping indices to characters
        blank_id: index of blank token
    
    Returns:
        decoded text
    """
    # Get predictions (argmax)
    preds = torch.argmax(logits, dim=-1)  # (batch, time)
    
    # Process first sequence in batch
    pred_tokens = preds[0].cpu().numpy()
    
    # CTC collapse: remove blanks and consecutive duplicates
    decoded = []
    prev_token = None
    
    special_tokens = {'<pad>', '<blank>', '<sos>', '<eos>', '<unk>'}
    
    for token in pred_tokens:
        # Skip blank
        if token == blank_id:
            prev_token = None
            continue
        
        # Skip consecutive duplicates
        if token == prev_token:
            continue
        
        # Decode token
        if token in idx_to_char:
            char = idx_to_char[token]
            if char not in special_tokens:
                decoded.append(char)
        
        prev_token = token
    
    return ''.join(decoded)


def analyze_ctc_output(logits, idx_to_char, top_k=10):
    """Analyze CTC output distribution"""
    # Get probabilities
    probs = torch.softmax(logits, dim=-1)  # (batch, time, vocab)
    
    # Average over time
    avg_probs = probs.mean(dim=1)[0]  # (vocab,)
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(avg_probs, top_k)
    
    print("\nTop-10 predicted characters (averaged over time):")
    for prob, idx in zip(top_probs, top_indices):
        char = idx_to_char.get(idx.item(), f'UNK({idx.item()})')
        print(f"  {char:15s} {prob.item()*100:6.2f}%")
    
    # Check if model is stuck on blanks
    blank_prob = avg_probs[1].item()  # Assuming blank is index 1
    print(f"\nBlank token probability: {blank_prob*100:.2f}%")
    
    # Entropy (measure of uncertainty)
    entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
    print(f"Prediction entropy: {entropy.item():.4f} (higher = more uncertain)")


def test_audio_file(model, audio_path, idx_to_char, char_to_idx, device):
    """Test on a single audio file with detailed analysis"""
    # Load audio
    audio_processor = AudioProcessor(sample_rate=16000, n_mels=80)
    waveform = audio_processor.load_audio(audio_path)
    mel_spec = audio_processor.compute_features(waveform, apply_augment=False)
    mel_spec = mel_spec.unsqueeze(0).to(device)
    
    print(f"\nAudio: {Path(audio_path).name}")
    print(f"Feature shape: {mel_spec.shape}")
    
    # Forward pass
    with torch.no_grad():
        # Get CTC logits directly
        encoder_out, _ = model.encoder(mel_spec)
        ctc_logits = model.ctc_head(encoder_out)
    
    print(f"CTC logits shape: {ctc_logits.shape}")
    
    # Analyze output
    analyze_ctc_output(ctc_logits, idx_to_char)
    
    # Decode
    transcription = decode_ctc_greedy(ctc_logits, idx_to_char, blank_id=char_to_idx.get('<blank>', 1))
    
    print(f"\nTranscription: {transcription if transcription else '(empty)'}")
    
    return transcription


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='kaggle_asr_outputs/checkpoints/best_model.pt')
    parser.add_argument('--audio', type=str, help='Audio file to test')
    parser.add_argument('--manifest', type=str, default='data/konkani-asr-v0/splits/manifests/test.json')
    parser.add_argument('--num_samples', type=int, default=3)
    
    args = parser.parse_args()
    
    print("="*70)
    print("DIRECT CTC INFERENCE TEST")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model, idx_to_char, char_to_idx, device = load_model(args.checkpoint)
    print(f"✓ Model loaded (vocab size: {len(idx_to_char)})")
    
    # Test on audio files
    if args.audio:
        test_audio_file(model, args.audio, idx_to_char, char_to_idx, device)
    else:
        # Load from manifest
        print(f"\nLoading test samples from manifest...")
        with open(args.manifest, 'r') as f:
            samples = [json.loads(line) for line in f][:args.num_samples]
        
        for i, sample in enumerate(samples, 1):
            print("\n" + "="*70)
            print(f"SAMPLE {i}/{len(samples)}")
            print("="*70)
            print(f"Ground truth: {sample['text'][:100]}...")
            
            transcription = test_audio_file(
                model, 
                sample['audio_filepath'],
                idx_to_char,
                char_to_idx,
                device
            )


if __name__ == '__main__':
    main()
