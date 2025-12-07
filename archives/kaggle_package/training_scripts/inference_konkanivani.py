"""
KonkaniVani ASR Inference Script
================================
Transcribe Konkani audio files using trained model
"""

import torch
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from models.konkanivani_asr import create_konkanivani_model
from data.audio_processing.audio_processor import AudioProcessor
from data.audio_processing.text_tokenizer import KonkaniTokenizer


class KonkaniASRInference:
    """Inference engine for KonkaniVani ASR"""
    
    def __init__(self, checkpoint_path, vocab_file, device='cpu'):
        """
        Initialize inference engine
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            vocab_file: Path to vocabulary JSON
            device: Device to run inference on
        """
        self.device = torch.device(device)
        
        # Load tokenizer
        self.tokenizer = KonkaniTokenizer(vocab_file)
        print(f"✅ Loaded vocabulary: {self.tokenizer.vocab_size} characters")
        
        # Create model
        self.model = create_konkanivani_model(vocab_size=self.tokenizer.vocab_size)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Loaded model from {checkpoint_path}")
        print(f"   Trained for {checkpoint['epoch']} epochs")
        print(f"   Best val loss: {checkpoint['val_loss']:.4f}")
        
        # Audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=16000,
            n_mels=80,
            spec_augment=False  # No augmentation for inference
        )
    
    def transcribe(self, audio_path):
        """
        Transcribe a single audio file
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            transcript: Transcribed text
        """
        # Load and process audio
        audio_features, duration = self.audio_processor.process_audio_file(audio_path)
        
        # Add batch dimension
        audio_features = audio_features.unsqueeze(0).to(self.device)
        audio_lengths = torch.tensor([audio_features.size(1)], dtype=torch.long).to(self.device)
        
        # Run inference
        with torch.no_grad():
            ctc_logits, _ = self.model(audio_features, audio_lengths)
            
            # CTC greedy decoding
            ctc_preds = ctc_logits.argmax(dim=-1).squeeze(0).cpu().tolist()
            
            # Decode to text
            transcript = self.tokenizer.decode_ctc(ctc_preds)
        
        return transcript, duration
    
    def transcribe_batch(self, audio_paths):
        """
        Transcribe multiple audio files
        
        Args:
            audio_paths: List of audio file paths
        
        Returns:
            transcripts: List of transcribed texts
        """
        transcripts = []
        
        for audio_path in audio_paths:
            transcript, duration = self.transcribe(audio_path)
            transcripts.append({
                'audio_path': audio_path,
                'transcript': transcript,
                'duration': duration
            })
        
        return transcripts


def main():
    parser = argparse.ArgumentParser(description='Transcribe Konkani audio with KonkaniVani ASR')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, default='vocab.json',
                        help='Path to vocabulary file')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to audio file or directory')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to run inference on')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for transcripts (optional)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"\n{'='*80}")
    print("KONKANIVANI ASR INFERENCE")
    print(f"{'='*80}")
    print(f"Device: {device}\n")
    
    # Initialize inference engine
    inference = KonkaniASRInference(args.checkpoint, args.vocab, device)
    
    # Get audio files
    audio_path = Path(args.audio)
    if audio_path.is_file():
        audio_files = [str(audio_path)]
    elif audio_path.is_dir():
        audio_files = [str(f) for f in audio_path.glob('*.wav')]
    else:
        print(f"❌ Error: {args.audio} not found")
        return
    
    print(f"Found {len(audio_files)} audio file(s)\n")
    
    # Transcribe
    results = []
    for audio_file in audio_files:
        print(f"Transcribing: {Path(audio_file).name}")
        transcript, duration = inference.transcribe(audio_file)
        
        print(f"  Duration: {duration:.2f}s")
        print(f"  Transcript: {transcript}\n")
        
        results.append({
            'audio_file': audio_file,
            'transcript': transcript,
            'duration': duration
        })
    
    # Save results
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved transcripts to {args.output}")
    
    print(f"\n{'='*80}")
    print("TRANSCRIPTION COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
