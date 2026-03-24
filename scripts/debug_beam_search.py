#!/usr/bin/env python3
"""Debug beam search decoder"""

import torch
import json
from beam_search_decoder import load_model, extract_features, BeamSearchDecoder

# Load model
print("Loading model...")
model, vocab_size = load_model('outputs/conformer_ctc_run1/best_conformer_ctc.pt')
print(f"Model loaded, vocab_size: {vocab_size}")

# Load test sample
print("\nLoading test sample...")
with open('data/konkani-mega-dataset/manifests/test.json', 'r') as f:
    sample = json.loads(f.readline())
print(f"Audio: {sample['audio_filepath']}")
print(f"Reference: {sample['text']}")

# Extract features
print("\nExtracting features...")
try:
    features = extract_features(sample['audio_filepath'])
    print(f"Features shape: {features.shape}")
except Exception as e:
    print(f"ERROR extracting features: {e}")
    exit(1)

# Get model output
print("\nGetting model output...")
with torch.no_grad():
    encoder_out, _ = model.encoder(features)
    print(f"Encoder output shape: {encoder_out.shape}")
    
    ctc_logits = model.ctc_head(encoder_out)
    print(f"CTC logits shape: {ctc_logits.shape}")
    
    # Check logits
    print(f"Logits min: {ctc_logits.min().item():.4f}")
    print(f"Logits max: {ctc_logits.max().item():.4f}")
    print(f"Logits mean: {ctc_logits.mean().item():.4f}")
    
    # Get predictions
    preds = ctc_logits.argmax(dim=-1)
    print(f"\nPredicted indices: {preds[0, :20].tolist()}")
    
    # Apply log softmax
    log_probs = torch.nn.functional.log_softmax(ctc_logits, dim=-1)
    log_probs = log_probs.squeeze(0)
    print(f"Log probs shape: {log_probs.shape}")

# Create decoder
print("\nCreating decoder...")
decoder = BeamSearchDecoder('data/konkani-mega-dataset/vocab.json')

# Greedy decode
print("\nGreedy decoding...")
text_greedy = decoder.greedy_decode(log_probs)
print(f"Greedy result: '{text_greedy}'")
print(f"Greedy length: {len(text_greedy)}")

# Beam search decode
print("\nBeam search decoding...")
text_beam = decoder.beam_search_decode(log_probs, beam_width=10)
print(f"Beam result: '{text_beam}'")
print(f"Beam length: {len(text_beam)}")

print("\nDone!")
