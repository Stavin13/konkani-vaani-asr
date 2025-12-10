#!/usr/bin/env python3
"""
Advanced ASR Inference with better decoding for the 50-epoch models
"""

import torch
import json
import sys
import librosa
import numpy as np
from pathlib import Path
from collections import OrderedDict
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkanivani_asr import KonkaniVaniASR

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from DataParallel checkpoints"""
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

class AdvancedASRInference:
    def __init__(self, checkpoint_path, vocab_path):
        self.checkpoint_path = Path(checkpoint_path)
        self.vocab_path = Path(vocab_path)
        
        print(f"🎯 Advanced ASR Inference System")
        print(f"📁 Checkpoint: {self.checkpoint_path}")
        print(f"📝 Vocabulary: {self.vocab_path}")
        
        self.load_vocabulary()
        self.load_model()
    
    def load_vocabulary(self):
        """Load vocabulary"""
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.char2idx = vocab_data['char2idx']
        self.idx2char = vocab_data['idx2char']
        self.vocab_size = len(self.char2idx)
        
        print(f"✅ Vocabulary loaded: {self.vocab_size} characters")
        
        # Show important tokens
        special_tokens = ['<pad>', '<blank>', '<sos>', '<eos>', '<unk>']
        print(f"🔤 Special tokens:")
        for token in special_tokens:
            if token in self.char2idx:
                print(f"  {token}: {self.char2idx[token]}")
    
    def load_model(self):
        """Load model with DataParallel fix"""
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        print(f"\n📊 Model info:")
        print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'Unknown')}")
        
        # Get model config
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        # Create model
        self.model = KonkaniVaniASR(
            vocab_size=self.vocab_size,
            input_dim=model_config.get('input_dim', 80),
            d_model=model_config.get('d_model', 256),
            encoder_layers=model_config.get('encoder_layers', 12),
            decoder_layers=model_config.get('decoder_layers', 6),
            num_heads=model_config.get('num_heads', 4),
            conv_kernel_size=model_config.get('conv_kernel_size', 31),
            dropout=model_config.get('dropout', 0.1)
        )
        
        # Fix DataParallel state dict if needed
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('module.') for key in state_dict.keys()):
            print(f"🔧 Fixing DataParallel state dict...")
            state_dict = remove_module_prefix(state_dict)
        
        # Load state dict
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
    
    def preprocess_audio(self, audio_path, duration=None):
        """Preprocess audio with multiple approaches"""
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=16000, duration=duration)
            
            print(f"📊 Audio: {len(audio)/sr:.2f}s, {len(audio)} samples")
            
            # Method 1: Standard mel-spectrogram (log scale)
            mel_log = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=80, hop_length=160, win_length=400, n_fft=400
            )
            mel_log = librosa.power_to_db(mel_log).T
            mel_log_tensor = torch.FloatTensor(mel_log).unsqueeze(0)
            
            # Method 2: Linear mel-spectrogram
            mel_linear = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=80, hop_length=160, win_length=400, n_fft=400
            ).T
            mel_linear_tensor = torch.FloatTensor(mel_linear).unsqueeze(0)
            
            # Method 3: Normalized log mel
            mel_norm = librosa.power_to_db(librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=80, hop_length=160, win_length=400, n_fft=400
            )).T
            # Normalize to [-1, 1]
            mel_norm = (mel_norm - mel_norm.mean()) / (mel_norm.std() + 1e-8)
            mel_norm_tensor = torch.FloatTensor(mel_norm).unsqueeze(0)
            
            return {
                'log': mel_log_tensor,
                'linear': mel_linear_tensor, 
                'normalized': mel_norm_tensor,
                'duration': len(audio) / sr
            }
            
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            return None
    
    def advanced_ctc_decode(self, logits, method='greedy'):
        """Advanced CTC decoding with multiple methods"""
        blank_token = self.char2idx.get('<blank>', 1)
        
        if method == 'greedy':
            # Standard greedy decoding
            predictions = logits.argmax(dim=-1).squeeze(0)
            
        elif method == 'confidence':
            # Confidence-based decoding
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            max_probs, predictions = probs.max(dim=-1)
            
            # Only keep high-confidence predictions
            confidence_threshold = 0.3
            confident_mask = max_probs > confidence_threshold
            predictions = predictions * confident_mask.long()
            
        elif method == 'beam':
            # Simple beam search (top-2)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            top2_probs, top2_indices = probs.topk(2, dim=-1)
            
            # Use top prediction if confidence is high, otherwise blank
            predictions = torch.where(
                top2_probs[:, 0] > 0.4,
                top2_indices[:, 0],
                torch.tensor(blank_token)
            )
        
        # Remove blanks and consecutive duplicates
        decoded_tokens = []
        prev_token = None
        
        for token in predictions.tolist():
            if token != blank_token and token != prev_token:
                decoded_tokens.append(token)
            prev_token = token
        
        # Convert to characters
        decoded_chars = []
        for token in decoded_tokens:
            char = self.idx2char.get(str(token), '<unk>')
            if char not in ['<pad>', '<blank>', '<sos>', '<eos>', '<unk>']:
                decoded_chars.append(char)
        
        return ''.join(decoded_chars), decoded_tokens
    
    def transcribe_with_multiple_methods(self, audio_path, duration=10):
        """Transcribe using multiple preprocessing and decoding methods"""
        print(f"\n🎵 Transcribing: {audio_path.name}")
        
        # Preprocess audio
        audio_data = self.preprocess_audio(audio_path, duration=duration)
        if not audio_data:
            return None
        
        results = {}
        
        # Test different preprocessing methods
        for prep_method, mel_tensor in audio_data.items():
            if prep_method == 'duration':
                continue
                
            print(f"\n🔧 Testing preprocessing: {prep_method}")
            print(f"📈 Feature shape: {mel_tensor.shape}")
            
            # Forward pass
            audio_length = torch.LongTensor([mel_tensor.size(1)])
            
            with torch.no_grad():
                ctc_logits, _ = self.model(mel_tensor, audio_length)
            
            # Test different decoding methods
            for decode_method in ['greedy', 'confidence', 'beam']:
                decoded_text, tokens = self.advanced_ctc_decode(ctc_logits, decode_method)
                
                key = f"{prep_method}_{decode_method}"
                results[key] = {
                    'text': decoded_text,
                    'tokens': len(tokens),
                    'length': len(decoded_text)
                }
                
                print(f"  {decode_method:>10}: '{decoded_text}' ({len(tokens)} tokens)")
        
        return results
    
    def batch_test_multiple_files(self, audio_dir, num_files=5):
        """Test multiple files with the best method"""
        audio_dir = Path(audio_dir)
        audio_files = list(audio_dir.glob("*.wav"))
        
        if not audio_files:
            print(f"❌ No audio files found in {audio_dir}")
            return
        
        # Test a few files to find best method
        print(f"\n🧪 Testing {min(num_files, len(audio_files))} files...")
        
        test_files = audio_files[:num_files]
        all_results = []
        
        for i, audio_file in enumerate(test_files, 1):
            print(f"\n{'='*60}")
            print(f"📁 File {i}/{len(test_files)}: {audio_file.name}")
            
            results = self.transcribe_with_multiple_methods(audio_file, duration=5)
            if results:
                all_results.append({
                    'file': audio_file.name,
                    'results': results
                })
        
        # Analyze results
        print(f"\n{'='*60}")
        print(f"📊 ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        if all_results:
            # Find method with most non-empty results
            method_scores = {}
            
            for result in all_results:
                for method, data in result['results'].items():
                    if method not in method_scores:
                        method_scores[method] = {'texts': 0, 'total_tokens': 0, 'total_chars': 0}
                    
                    if data['length'] > 0:
                        method_scores[method]['texts'] += 1
                    method_scores[method]['total_tokens'] += data['tokens']
                    method_scores[method]['total_chars'] += data['length']
            
            print(f"🏆 Method Performance:")
            print(f"{'Method':<20} {'Non-empty':<10} {'Avg Tokens':<12} {'Avg Chars':<10}")
            print(f"{'-'*20} {'-'*10} {'-'*12} {'-'*10}")
            
            for method, scores in sorted(method_scores.items(), 
                                       key=lambda x: (x[1]['texts'], x[1]['total_chars']), 
                                       reverse=True):
                avg_tokens = scores['total_tokens'] / len(all_results)
                avg_chars = scores['total_chars'] / len(all_results)
                print(f"{method:<20} {scores['texts']:<10} {avg_tokens:<12.1f} {avg_chars:<10.1f}")
            
            # Show best results
            best_method = max(method_scores.keys(), 
                            key=lambda x: (method_scores[x]['texts'], method_scores[x]['total_chars']))
            
            print(f"\n🎯 Best method: {best_method}")
            print(f"\n🔤 Sample outputs with {best_method}:")
            
            for result in all_results[:3]:
                if best_method in result['results']:
                    text = result['results'][best_method]['text']
                    display_text = text[:50] + '...' if len(text) > 50 else text
                    print(f"  📁 {result['file']}: '{display_text}'")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Advanced ASR Inference')
    parser.add_argument('--checkpoint', default='kaggle_asr_outputs/checkpoints/checkpoint_epoch_45.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--vocab', default='data/vocab.json',
                       help='Path to vocabulary file')
    parser.add_argument('--audio-dir', default='data/konkani-asr-v0/data/processed_segments_diarized/audio_segments',
                       help='Directory containing audio files')
    parser.add_argument('--num-files', type=int, default=5,
                       help='Number of files to test')
    
    args = parser.parse_args()
    
    print("🎯 KonkaniVani ASR - Advanced Inference")
    print("=" * 50)
    
    # Check files exist
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    if not Path(args.vocab).exists():
        print(f"❌ Vocabulary not found: {args.vocab}")
        return
    
    if not Path(args.audio_dir).exists():
        print(f"❌ Audio directory not found: {args.audio_dir}")
        return
    
    # Initialize inference system
    asr = AdvancedASRInference(args.checkpoint, args.vocab)
    
    # Run batch test
    asr.batch_test_multiple_files(args.audio_dir, args.num_files)
    
    print(f"\n🎉 Advanced inference completed!")

if __name__ == "__main__":
    main()