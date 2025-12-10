#!/usr/bin/env python3
"""
Production ASR Inference Script
Tests the trained Kaggle checkpoint with real Konkani audio segments
"""

import torch
import json
import sys
import librosa
import numpy as np
from pathlib import Path
import random
import time
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.konkanivani_asr import KonkaniVaniASR

class ProductionASRInference:
    def __init__(self, checkpoint_path, vocab_path, audio_dir):
        self.checkpoint_path = Path(checkpoint_path)
        self.vocab_path = Path(vocab_path)
        self.audio_dir = Path(audio_dir)
        
        print(f"🎯 Production ASR Inference System")
        print(f"📁 Checkpoint: {self.checkpoint_path}")
        print(f"📝 Vocabulary: {self.vocab_path}")
        print(f"🎵 Audio Directory: {self.audio_dir}")
        
        # Load components
        self.load_vocabulary()
        self.load_model()
        self.find_audio_files()
    
    def load_vocabulary(self):
        """Load vocabulary for decoding"""
        print(f"\n📚 Loading vocabulary from: {self.vocab_path}")
        
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.char2idx = vocab_data['char2idx']
        self.idx2char = vocab_data['idx2char']
        self.vocab_size = len(self.char2idx)
        
        print(f"✅ Vocabulary loaded successfully!")
        print(f"  Vocabulary size: {self.vocab_size} characters")
        print(f"  Keys in vocab file: {list(vocab_data.keys())}")
        
        # Show sample characters
        sample_chars = list(self.char2idx.keys())[:50]
        print(f"  Sample characters: {sample_chars}")
    
    def load_model(self):
        """Load the trained ASR model"""
        print(f"\n🤖 Loading model from: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        print(f"📊 Model training info:")
        print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"  Train Loss: {checkpoint.get('train_loss', 'Unknown'):.4f}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'Unknown'):.4f}")
        
        # Get model config
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        # Create model with correct parameters
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
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"🏗️  Architecture: {model_config.get('encoder_layers', 12)} encoder + {model_config.get('decoder_layers', 6)} decoder layers")
        print(f"📐 D-model: {model_config.get('d_model', 256)}")
        print(f"🎯 Vocab size: {self.vocab_size}")
    
    def find_audio_files(self):
        """Find available audio files"""
        print(f"\n🔍 Searching for audio files in: {self.audio_dir}")
        
        # Look for audio files
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
        self.audio_files = []
        
        for ext in audio_extensions:
            self.audio_files.extend(list(self.audio_dir.glob(ext)))
            # Also search recursively
            self.audio_files.extend(list(self.audio_dir.glob(f"**/{ext}")))
        
        # Remove duplicates
        self.audio_files = list(set(self.audio_files))
        
        if not self.audio_files:
            print(f"❌ No audio files found in {self.audio_dir}")
            print(f"   Searched for: {audio_extensions}")
            return
        
        print(f"✅ Found {len(self.audio_files)} audio files")
        
        # Show sample files
        sample_files = self.audio_files[:5]
        print(f"📁 Sample files:")
        for f in sample_files:
            print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    def preprocess_audio(self, audio_path):
        """Preprocess audio file for inference"""
        try:
            # Load audio with librosa
            audio, sr = librosa.load(str(audio_path), sr=16000)
            
            # Handle different audio lengths
            max_len = 16000 * 10  # 10 seconds max
            if len(audio) > max_len:
                audio = audio[:max_len]
            elif len(audio) < 1600:  # Too short (< 0.1 seconds)
                audio = np.pad(audio, (0, 1600 - len(audio)))
            
            # Extract mel-spectrogram features
            mel = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_mels=80, 
                hop_length=160, 
                win_length=400,
                n_fft=400
            )
            
            # Convert to log scale
            mel = librosa.power_to_db(mel)
            
            # Transpose to (time, n_mels)
            mel = mel.T
            
            # Convert to tensor and add batch dimension
            mel_tensor = torch.FloatTensor(mel).unsqueeze(0)
            
            return mel_tensor, len(audio) / sr  # Return tensor and duration
            
        except Exception as e:
            print(f"❌ Error processing {audio_path}: {e}")
            return None, None
    
    def ctc_decode(self, logits):
        """CTC decoding with blank removal and deduplication"""
        # Get predictions (greedy decoding)
        predictions = logits.argmax(dim=-1).squeeze(0)  # Remove batch dimension
        
        # Remove blanks and consecutive duplicates
        decoded_tokens = []
        prev_token = None
        
        for token in predictions.tolist():
            # Skip blank token (index 1) and consecutive duplicates
            if token != 1 and token != prev_token:
                decoded_tokens.append(token)
            prev_token = token
        
        # Convert tokens to characters
        decoded_chars = []
        for token in decoded_tokens:
            char = self.idx2char.get(str(token), '<unk>')
            # Skip special tokens
            if char not in ['<pad>', '<blank>', '<sos>', '<eos>', '<unk>']:
                decoded_chars.append(char)
        
        return ''.join(decoded_chars)
    
    def transcribe_single_audio(self, audio_path, verbose=True):
        """Transcribe a single audio file"""
        if verbose:
            print(f"\n🎵 Transcribing: {audio_path.name}")
        
        # Preprocess audio
        audio_features, duration = self.preprocess_audio(audio_path)
        if audio_features is None:
            return None
        
        if verbose:
            print(f"📊 Audio duration: {duration:.2f}s")
            print(f"📈 Feature shape: {audio_features.shape}")
        
        # Get audio length for the model
        audio_length = torch.LongTensor([audio_features.size(1)])
        
        # Inference
        start_time = time.time()
        
        with torch.no_grad():
            ctc_logits, _ = self.model(audio_features, audio_length)
        
        inference_time = time.time() - start_time
        
        if verbose:
            print(f"⚡ Inference time: {inference_time:.3f}s")
            print(f"📤 Output shape: {ctc_logits.shape}")
        
        # Decode to text
        decoded_text = self.ctc_decode(ctc_logits)
        
        if verbose:
            print(f"🔤 Transcribed text: '{decoded_text}'")
            print(f"📏 Text length: {len(decoded_text)} characters")
        
        return {
            'file': audio_path.name,
            'path': str(audio_path),
            'duration': duration,
            'audio_shape': list(audio_features.shape),
            'output_shape': list(ctc_logits.shape),
            'inference_time': inference_time,
            'transcribed_text': decoded_text,
            'text_length': len(decoded_text)
        }
    
    def batch_transcribe(self, num_samples=10, random_selection=True):
        """Transcribe multiple audio files"""
        print(f"\n🎯 Batch transcription: {num_samples} files")
        
        if len(self.audio_files) < num_samples:
            num_samples = len(self.audio_files)
            print(f"⚠️  Only {num_samples} files available")
        
        # Select files
        if random_selection:
            test_files = random.sample(self.audio_files, num_samples)
            print(f"🎲 Selected {num_samples} random files")
        else:
            test_files = self.audio_files[:num_samples]
            print(f"📋 Using first {num_samples} files")
        
        results = []
        total_time = 0
        total_audio_duration = 0
        
        print(f"\n{'='*80}")
        print(f"🚀 STARTING BATCH TRANSCRIPTION")
        print(f"{'='*80}")
        
        for i, audio_file in enumerate(test_files, 1):
            print(f"\n[{i}/{num_samples}] Processing: {audio_file.name}")
            
            result = self.transcribe_single_audio(audio_file, verbose=False)
            if result:
                results.append(result)
                total_time += result['inference_time']
                total_audio_duration += result['duration']
                
                # Show progress
                print(f"  ✅ Duration: {result['duration']:.2f}s | "
                      f"Inference: {result['inference_time']:.3f}s | "
                      f"Text: '{result['transcribed_text'][:50]}{'...' if len(result['transcribed_text']) > 50 else ''}'")
            else:
                print(f"  ❌ Failed to process")
        
        # Summary
        print(f"\n{'='*80}")
        print(f"📊 BATCH TRANSCRIPTION SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Successful transcriptions: {len(results)}/{num_samples}")
        print(f"🎵 Total audio duration: {total_audio_duration:.2f}s")
        print(f"⚡ Total inference time: {total_time:.3f}s")
        print(f"📈 Average inference time: {total_time/len(results):.3f}s per file")
        print(f"🚀 Real-time factor: {total_time/total_audio_duration:.2f}x")
        
        # Text analysis
        if results:
            text_lengths = [r['text_length'] for r in results]
            avg_length = sum(text_lengths) / len(text_lengths)
            
            print(f"\n📝 Text Analysis:")
            print(f"  Average text length: {avg_length:.1f} characters")
            print(f"  Text length range: {min(text_lengths)} - {max(text_lengths)}")
            
            # Show sample transcriptions
            print(f"\n🔤 Sample Transcriptions:")
            for i, result in enumerate(results[:5], 1):
                text = result['transcribed_text']
                display_text = text[:100] + '...' if len(text) > 100 else text
                print(f"  {i}. {result['file']}: '{display_text}'")
        
        return results
    
    def benchmark_performance(self, num_files=20):
        """Benchmark model performance"""
        print(f"\n⚡ Performance Benchmark: {num_files} files")
        
        test_files = random.sample(self.audio_files, min(num_files, len(self.audio_files)))
        
        times = []
        durations = []
        successful = 0
        
        print(f"🔄 Processing {len(test_files)} files...")
        
        for i, audio_file in enumerate(test_files, 1):
            print(f"  [{i}/{len(test_files)}] {audio_file.name}", end=" ... ")
            
            audio_features, duration = self.preprocess_audio(audio_file)
            if audio_features is None:
                print("❌ Failed")
                continue
            
            audio_length = torch.LongTensor([audio_features.size(1)])
            
            start_time = time.time()
            with torch.no_grad():
                ctc_logits, _ = self.model(audio_features, audio_length)
            inference_time = time.time() - start_time
            
            times.append(inference_time)
            durations.append(duration)
            successful += 1
            print(f"✅ {inference_time:.3f}s")
        
        if times:
            avg_time = sum(times) / len(times)
            avg_duration = sum(durations) / len(durations)
            rtf = sum(times) / sum(durations)  # Real-time factor
            
            print(f"\n📊 Performance Benchmark Results:")
            print(f"  ✅ Successful inferences: {successful}/{num_files}")
            print(f"  ⚡ Average inference time: {avg_time:.3f}s")
            print(f"  🎵 Average audio duration: {avg_duration:.2f}s")
            print(f"  🚀 Real-time factor: {rtf:.2f}x")
            print(f"  📈 Throughput: {1/avg_time:.1f} files/second")
            print(f"  🏃 Speed: {avg_duration/avg_time:.1f}x faster than real-time")
        
        return times, durations
    
    def interactive_mode(self):
        """Interactive transcription mode"""
        print(f"\n🎮 Interactive Transcription Mode")
        print(f"Available commands:")
        print(f"  'random' - transcribe a random file")
        print(f"  'batch N' - transcribe N random files")
        print(f"  'benchmark' - run performance benchmark")
        print(f"  'list' - show available files")
        print(f"  'quit' - exit")
        
        while True:
            try:
                command = input(f"\n🎯 Enter command: ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'random':
                    if self.audio_files:
                        random_file = random.choice(self.audio_files)
                        self.transcribe_single_audio(random_file)
                    else:
                        print("❌ No audio files available")
                elif command.startswith('batch'):
                    try:
                        parts = command.split()
                        num = int(parts[1]) if len(parts) > 1 else 5
                        self.batch_transcribe(num_samples=num)
                    except (ValueError, IndexError):
                        print("❌ Invalid batch command. Use: batch N")
                elif command == 'benchmark':
                    self.benchmark_performance()
                elif command == 'list':
                    print(f"\n📁 Available files ({len(self.audio_files)}):")
                    for i, f in enumerate(self.audio_files[:20], 1):
                        print(f"  {i}. {f.name}")
                    if len(self.audio_files) > 20:
                        print(f"  ... and {len(self.audio_files) - 20} more")
                else:
                    print("❌ Unknown command")
                    
            except KeyboardInterrupt:
                print(f"\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Production ASR Inference')
    parser.add_argument('--checkpoint', default='checkpoints/best_model_scripts1_fixed.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--vocab', default='data/vocab.json',
                       help='Path to vocabulary file')
    parser.add_argument('--audio-dir', default='data/konkani-asr-v0/data/processed_segments_diarized/audio_segments',
                       help='Directory containing audio files')
    parser.add_argument('--mode', choices=['batch', 'benchmark', 'interactive'], default='batch',
                       help='Operation mode')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples for batch mode')
    parser.add_argument('--output', help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    print("🎯 KonkaniVani ASR - Production Inference")
    print("=" * 60)
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if files exist
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
    asr = ProductionASRInference(args.checkpoint, args.vocab, args.audio_dir)
    
    if not asr.audio_files:
        print("❌ No audio files found. Exiting.")
        return
    
    # Run based on mode
    results = None
    
    if args.mode == 'batch':
        print(f"\n🚀 Running batch transcription...")
        results = asr.batch_transcribe(num_samples=args.num_samples)
        
    elif args.mode == 'benchmark':
        print(f"\n⚡ Running performance benchmark...")
        asr.benchmark_performance(num_files=20)
        
    elif args.mode == 'interactive':
        print(f"\n🎮 Starting interactive mode...")
        asr.interactive_mode()
    
    # Save results if requested
    if results and args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path}")
    
    print(f"\n🎉 Production inference completed!")
    print(f"🕒 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()