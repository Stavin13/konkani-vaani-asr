#!/usr/bin/env python3
"""
Test if the attention decoder produces better results than CTC
"""
import torch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.test_model_direct_ctc import load_model
from data.audio_processing.audio_processor import AudioProcessor


def decode_with_attention(model, mel_spec, idx_to_char, char_to_idx, max_len=100):
    """Use attention decoder for inference"""
    device = next(model.parameters()).device
    
    # Encode audio
    with torch.no_grad():
        encoder_out, encoder_mask = model.encoder(mel_spec)
        
        # Start with SOS token
        sos_id = char_to_idx.get('<sos>', 2)
        eos_id = char_to_idx.get('<eos>', 3)
        
        # Greedy decoding
        decoded_ids = [sos_id]
        
        for _ in range(max_len):
            # Prepare decoder input
            tgt = torch.LongTensor([decoded_ids]).to(device)
            
            # Create causal mask
            tgt_len = tgt.size(1)
            causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device),
                diagonal=1
            ).bool()
            
            # Decode
            output = model.decoder(tgt, encoder_out, tgt_mask=causal_mask, memory_mask=encoder_mask)
            
            # Get next token
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            # Stop if EOS
            if next_token == eos_id:
                break
            
            decoded_ids.append(next_token)
        
        # Convert to text
        special_tokens = {'<pad>', '<blank>', '<sos>', '<eos>', '<unk>'}
        decoded_text = []
        for token_id in decoded_ids[1:]:  # Skip SOS
            if token_id in idx_to_char:
                char = idx_to_char[token_id]
                if char not in special_tokens:
                    decoded_text.append(char)
        
        return ''.join(decoded_text)


def main():
    print("="*70)
    print("ATTENTION DECODER TEST")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model, idx_to_char, char_to_idx, device = load_model('kaggle_asr_outputs/checkpoints/best_model.pt')
    print(f"✓ Model loaded")
    
    # Load test sample
    with open('data/konkani-asr-v0/splits/manifests/test.json', 'r') as f:
        sample = json.loads(f.readline())
    
    print(f"\nGround truth: {sample['text'][:100]}...")
    print(f"Audio: {Path(sample['audio_filepath']).name}")
    
    # Load audio
    audio_processor = AudioProcessor(sample_rate=16000, n_mels=80)
    waveform = audio_processor.load_audio(sample['audio_filepath'])
    mel_spec = audio_processor.compute_features(waveform, apply_augment=False)
    mel_spec = mel_spec.unsqueeze(0).to(device)
    
    # Test attention decoder
    print("\nDecoding with attention decoder...")
    transcription = decode_with_attention(model, mel_spec, idx_to_char, char_to_idx)
    
    print(f"Prediction: {transcription if transcription else '(empty)'}")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    if transcription:
        print("✓ Attention decoder produces output!")
        print("  The model learned something, but CTC head needs more training.")
    else:
        print("✗ Attention decoder also produces empty output.")
        print("  Model needs more training overall.")
    print("="*70)


if __name__ == '__main__':
    main()
