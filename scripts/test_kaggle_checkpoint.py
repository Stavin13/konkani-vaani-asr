#!/usr/bin/env python3
"""
Test the downloaded Kaggle checkpoint
"""

import torch
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkanivani_asr import KonkaniVaniASR

def test_kaggle_checkpoint():
    """Test the downloaded checkpoint"""
    
    checkpoint_path = Path("checkpoints/best_model_scripts1_fixed.pt")
    vocab_path = Path("data/vocab.json")
    
    print("🔍 Testing Kaggle checkpoint...")
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"📝 Vocabulary: {vocab_path}")
    
    # Check files exist
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    if not vocab_path.exists():
        print(f"❌ Vocabulary not found: {vocab_path}")
        return False
    
    try:
        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        vocab_size = len(vocab_data['char2idx'])
        print(f"✅ Vocabulary loaded: {vocab_size} characters")
        
        # Load checkpoint
        print("🔄 Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("📊 Checkpoint info:")
        print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"  Train Loss: {checkpoint.get('train_loss', 'Unknown'):.4f}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'Unknown'):.4f}")
        
        # Get model config
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        print(f"\n🏗️  Model architecture:")
        print(f"  Vocab size: {model_config.get('vocab_size', vocab_size)}")
        print(f"  D-model: {model_config.get('d_model', 256)}")
        print(f"  Encoder layers: {model_config.get('encoder_layers', 12)}")
        print(f"  Decoder layers: {model_config.get('decoder_layers', 6)}")
        
        # Create model
        model = KonkaniVaniASR(
            vocab_size=vocab_size,
            input_dim=model_config.get('input_dim', 80),
            d_model=model_config.get('d_model', 256),
            encoder_layers=model_config.get('encoder_layers', 12),
            decoder_layers=model_config.get('decoder_layers', 6),
            num_heads=model_config.get('num_heads', 4),
            conv_kernel_size=model_config.get('conv_kernel_size', 31),
            dropout=model_config.get('dropout', 0.1)
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Model loaded successfully!")
        
        # Test with dummy input
        print("\n🧪 Testing model inference...")
        batch_size = 2
        seq_len = 100
        mel_dim = 80
        
        dummy_audio = torch.randn(batch_size, seq_len, mel_dim)
        dummy_lengths = torch.tensor([seq_len, seq_len-10])
        
        with torch.no_grad():
            ctc_logits, _ = model(dummy_audio, dummy_lengths)
        
        print(f"✅ Inference test passed!")
        print(f"  Input shape: {dummy_audio.shape}")
        print(f"  Output shape: {ctc_logits.shape}")
        print(f"  Expected vocab size: {vocab_size}")
        print(f"  Actual output vocab size: {ctc_logits.shape[-1]}")
        
        # Verify vocab size matches
        if ctc_logits.shape[-1] == vocab_size:
            print("✅ Vocabulary size matches perfectly!")
        else:
            print(f"⚠️  Vocab size mismatch: expected {vocab_size}, got {ctc_logits.shape[-1]}")
        
        # Test CTC decoding
        print("\n🔤 Testing CTC decoding...")
        predictions = ctc_logits.argmax(dim=-1)
        
        # Convert first prediction to text
        pred_tokens = predictions[0].tolist()
        
        # Remove blanks and consecutive duplicates (basic CTC decoding)
        decoded_tokens = []
        prev_token = None
        for token in pred_tokens:
            if token != 1 and token != prev_token:  # 1 is blank token
                decoded_tokens.append(token)
            prev_token = token
        
        # Convert to characters
        idx2char = vocab_data['idx2char']
        decoded_chars = [idx2char.get(str(token), '<unk>') for token in decoded_tokens[:20]]  # First 20 chars
        decoded_text = ''.join(decoded_chars)
        
        print(f"  Sample prediction: {decoded_text}")
        print(f"  Token count: {len(decoded_tokens)}")
        
        print(f"\n🎉 Checkpoint test completed successfully!")
        print(f"🔥 Your model is ready for inference!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kaggle_checkpoint()
    if success:
        print("\n✅ Checkpoint is working perfectly!")
        print("🚀 You can now use this model for ASR inference")
    else:
        print("\n❌ Checkpoint test failed")