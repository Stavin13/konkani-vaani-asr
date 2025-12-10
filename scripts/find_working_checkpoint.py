#!/usr/bin/env python3
"""
Test all checkpoints to find one that actually produces transcriptions
"""
import torch
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkanivani_asr import KonkaniVaniASR
from data.audio_processing.audio_processor import AudioProcessor


def quick_test_checkpoint(checkpoint_path, audio_path):
    """Quick test if checkpoint produces non-blank predictions"""
    try:
        device = torch.device('cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load vocab
        with open('data/vocab.json', 'r') as f:
            vocab_data = json.load(f)
        vocab_size = len(vocab_data.get('char2idx', vocab_data))
        
        # Create model
        model = KonkaniVaniASR(vocab_size=vocab_size, input_dim=80, d_model=256,
                               encoder_layers=12, decoder_layers=6, num_heads=4, dropout=0.1)
        
        # Load weights
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        
        # Process audio
        processor = AudioProcessor(sample_rate=16000, n_mels=80)
        waveform = processor.load_audio(audio_path)
        features = processor.compute_features(waveform, apply_augment=False)
        features_batch = features.unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            encoder_out, _ = model.encoder(features_batch)
            ctc_logits = model.ctc_head(encoder_out)
            probs = torch.softmax(ctc_logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        
        # Analyze predictions
        unique_preds = torch.unique(preds[0])
        blank_prob = probs[0, :, 1].mean().item()
        non_blank_tokens = len([t for t in unique_preds if t.item() not in [0, 1, 2, 3]])
        
        return {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_loss': checkpoint.get('val_loss', float('inf')),
            'unique_tokens': len(unique_preds),
            'non_blank_tokens': non_blank_tokens,
            'blank_prob': blank_prob,
            'working': non_blank_tokens > 5 and blank_prob < 0.90
        }
    except Exception as e:
        return {'error': str(e), 'working': False}


def main():
    # Find all checkpoints
    checkpoint_files = []
    for dir_path in ['kaggle_asr_outputs/checkpoints', 'checkpoints', 'archives/checkpoints_backup']:
        dir_path = Path(dir_path)
        if dir_path.exists():
            for pt_file in dir_path.glob('*.pt'):
                if not pt_file.name.startswith('.') and 'ner' not in pt_file.name.lower():
                    checkpoint_files.append(pt_file)
    
    # Get test audio
    manifest_path = 'data/konkani-asr-v0/splits/manifests/test.json'
    with open(manifest_path, 'r') as f:
        sample = json.loads(f.readline())
    audio_path = sample['audio_filepath']
    
    print("="*80)
    print("TESTING ALL CHECKPOINTS FOR WORKING TRANSCRIPTION")
    print("="*80)
    print(f"\nTest audio: {Path(audio_path).name}")
    print(f"Testing {len(checkpoint_files)} checkpoints...\n")
    
    results = []
    for i, ckpt_path in enumerate(checkpoint_files, 1):
        print(f"[{i}/{len(checkpoint_files)}] Testing {ckpt_path.name}...", end=' ')
        result = quick_test_checkpoint(ckpt_path, audio_path)
        result['path'] = str(ckpt_path)
        results.append(result)
        
        if result.get('error'):
            print(f"❌ Error: {result['error'][:50]}")
        elif result['working']:
            print(f"✅ WORKING! (blank: {result['blank_prob']:.1%}, tokens: {result['non_blank_tokens']})")
        else:
            print(f"❌ Not working (blank: {result['blank_prob']:.1%}, tokens: {result['non_blank_tokens']})")
    
    # Find working checkpoints
    working = [r for r in results if r.get('working')]
    not_working = [r for r in results if not r.get('working') and not r.get('error')]
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if working:
        print(f"\n✅ FOUND {len(working)} WORKING CHECKPOINT(S):")
        working.sort(key=lambda x: x['val_loss'])
        for r in working:
            print(f"\n  Path: {r['path']}")
            print(f"  Epoch: {r['epoch']}, Val Loss: {r['val_loss']:.4f}")
            print(f"  Blank prob: {r['blank_prob']:.1%}, Non-blank tokens: {r['non_blank_tokens']}")
    else:
        print(f"\n❌ NO WORKING CHECKPOINTS FOUND")
        print(f"\nAll {len(not_working)} checkpoints are predicting mostly blanks.")
        print("\nThis suggests:")
        print("  1. The model hasn't been trained long enough")
        print("  2. There's a training issue (learning rate, loss function, etc.)")
        print("  3. The CTC blank token might be incorrectly configured")
        
        # Show best checkpoint by val_loss
        if not_working:
            best = min(not_working, key=lambda x: x['val_loss'])
            print(f"\nBest checkpoint by val_loss:")
            print(f"  Path: {best['path']}")
            print(f"  Epoch: {best['epoch']}, Val Loss: {best['val_loss']:.4f}")
            print(f"  Blank prob: {best['blank_prob']:.1%}")
            print(f"  Unique tokens: {best['unique_tokens']}")


if __name__ == '__main__':
    main()
