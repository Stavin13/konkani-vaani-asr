#!/usr/bin/env python3
"""
Test the best ASR model on audio samples
"""
import torch
import json
import sys
from pathlib import Path
import argparse
import numpy as np
import librosa

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkanivani_asr import KonkaniVaniASR
from data.audio_processing.audio_processor import AudioProcessor


class ASRInference:
    """ASR inference wrapper"""
    
    def __init__(self, checkpoint_path, vocab_path='data/vocab.json', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load vocabulary
        print(f"Loading vocabulary from {vocab_path}...")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Handle different vocab formats
        if 'char2idx' in vocab_data:
            self.char_to_idx = vocab_data['char2idx']
            self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
            self.vocab_size = len(self.char_to_idx)
        else:
            self.idx_to_char = {int(k): v for k, v in vocab_data.items()}
            self.char_to_idx = {v: int(k) for k, v in vocab_data.items()}
            self.vocab_size = len(self.idx_to_char)
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        self.model = KonkaniVaniASR(
            vocab_size=self.vocab_size,
            input_dim=80,
            d_model=256,
            encoder_layers=12,
            decoder_layers=6,
            num_heads=4,
            dropout=0.1
        )
        
        # Load weights (handle DataParallel "module." prefix)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            # Remove "module." prefix from keys
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=16000,
            n_mels=80,
            n_fft=400,
            hop_length=160
        )
        
        print(f"✓ Model loaded successfully!")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Device: {self.device}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")
    
    def transcribe(self, audio_path):
        """Transcribe a single audio file"""
        # Load and process audio
        waveform = self.audio_processor.load_audio(audio_path)
        
        # Extract mel-spectrogram features
        mel_spec = self.audio_processor.compute_features(waveform, apply_augment=False)
        mel_spec = mel_spec.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Inference
        with torch.no_grad():
            predictions = self.model.recognize(mel_spec)
        
        # Decode predictions (CTC collapse)
        pred_tokens = predictions[0].cpu().numpy()
        
        # Debug: show raw predictions
        unique_tokens = np.unique(pred_tokens)
        token_chars = [self.idx_to_char.get(int(t), f'UNK({t})') for t in unique_tokens[:10]]
        
        transcription = self.decode_ctc(pred_tokens)
        
        return transcription, token_chars
    
    def decode_ctc(self, tokens):
        """Decode CTC output by collapsing repeats and removing blanks"""
        # Special tokens to skip
        special_tokens = {'<pad>', '<blank>', '<sos>', '<eos>', '<unk>'}
        
        decoded = []
        prev_token = None
        
        for token in tokens:
            # Skip blank (index 1 based on vocab)
            if token == 1:
                prev_token = token
                continue
            
            # Skip repeats
            if token == prev_token:
                continue
            
            # Decode token
            if token in self.idx_to_char:
                char = self.idx_to_char[token]
                # Skip special tokens
                if char not in special_tokens:
                    decoded.append(char)
            
            prev_token = token
        
        return ''.join(decoded)
    
    def transcribe_batch(self, audio_paths):
        """Transcribe multiple audio files"""
        results = []
        for audio_path in audio_paths:
            try:
                transcription, tokens = self.transcribe(audio_path)
                results.append({
                    'audio': str(audio_path),
                    'transcription': transcription,
                    'raw_tokens': tokens,
                    'success': True
                })
                print(f"✓ {Path(audio_path).name}: {transcription if transcription else '(empty)'}")
                print(f"  Raw tokens: {tokens[:5]}...")
            except Exception as e:
                results.append({
                    'audio': str(audio_path),
                    'error': str(e),
                    'success': False
                })
                print(f"✗ {Path(audio_path).name}: Error - {e}")
        
        return results


def find_test_audio_files(data_dir='data', max_files=10):
    """Find audio files for testing"""
    audio_files = []
    
    # Check konkani-asr-v0 dataset
    asr_data_dir = Path(data_dir) / 'konkani-asr-v0' / 'data'
    if asr_data_dir.exists():
        audio_files.extend(list(asr_data_dir.glob('**/*.wav'))[:max_files])
        audio_files.extend(list(asr_data_dir.glob('**/*.mp3'))[:max_files])
    
    # Check raw speech corpus
    corpus_dir = Path('KonkaniRawSpeechCorpus/Data')
    if corpus_dir.exists() and len(audio_files) < max_files:
        remaining = max_files - len(audio_files)
        audio_files.extend(list(corpus_dir.glob('**/*.wav'))[:remaining])
    
    return audio_files[:max_files]


def main():
    parser = argparse.ArgumentParser(description='Test ASR model inference')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='kaggle_asr_outputs/checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--audio',
        type=str,
        nargs='+',
        help='Audio file(s) to transcribe'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        default='data/vocab.json',
        help='Path to vocabulary file'
    )
    parser.add_argument(
        '--max_files',
        type=int,
        default=10,
        help='Maximum number of test files (if no audio specified)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ASR MODEL INFERENCE TEST")
    print("="*60)
    
    # Initialize inference
    asr = ASRInference(args.checkpoint, args.vocab)
    
    print("\n" + "="*60)
    print("TRANSCRIPTION RESULTS")
    print("="*60 + "\n")
    
    # Get audio files
    if args.audio:
        audio_files = [Path(f) for f in args.audio]
    else:
        print(f"Finding test audio files (max {args.max_files})...")
        audio_files = find_test_audio_files(max_files=args.max_files)
        
        if not audio_files:
            print("No audio files found! Please specify audio files with --audio")
            return
        
        print(f"Found {len(audio_files)} audio files\n")
    
    # Transcribe
    results = asr.transcribe_batch(audio_files)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    successful = sum(1 for r in results if r['success'])
    print(f"Successfully transcribed: {successful}/{len(results)}")
    
    # Save results
    output_file = Path('outputs/inference_results.json')
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()
