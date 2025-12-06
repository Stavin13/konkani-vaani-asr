#!/usr/bin/env python3
"""
Generate Synthetic Konkani Audio Dataset using Text-to-Speech

This script converts your 47k Konkani text samples into audio files
using Google Text-to-Speech (gTTS), creating a dataset for ASR training.
"""

import pandas as pd
from gtts import gTTS
from pathlib import Path
from tqdm import tqdm
import time
import os
import sys

class SyntheticAudioGenerator:
    """Generate synthetic audio from Konkani text using TTS"""
    
    def __init__(
        self,
        text_csv="../../data/processed/custom_konkani_sentiment_fixed.csv",
        output_dir="../../data/audio/synthetic",
        lang='mr'  # Use Hindi for Konkani (Devanagari)
    ):
        self.text_csv = text_csv
        self.output_dir = Path(output_dir)
        self.lang = lang
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*60)
        print("SYNTHETIC KONKANI AUDIO GENERATOR")
        print("="*60)
        print(f"Input: {text_csv}")
        print(f"Output: {output_dir}")
        print(f"TTS Language: {lang} (Hindi voice for Konkani)")
    
    def generate_dataset(self, max_samples=None, start_from=0):
        """
        Generate audio for text samples
        
        Args:
            max_samples: Limit number of samples (None = all)
            start_from: Start from this index (for resuming)
        """
        
        # Load text data
        print(f"\nLoading text data...")
        df = pd.read_csv(self.text_csv)
        
        if max_samples:
            df = df.iloc[start_from:start_from + max_samples]
        else:
            df = df.iloc[start_from:]
        
        print(f"Generating audio for {len(df)} samples (starting from index {start_from})")
        
        manifest = []
        errors = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating audio"):
            # Use Devanagari text
            text = row['devanagari']
            audio_id = f"konkani_{idx:06d}"
            audio_path = self.output_dir / f"{audio_id}.wav"
            
            # Skip if already exists
            if audio_path.exists():
                try:
                    import librosa
                    audio, sr = librosa.load(str(audio_path), sr=16000)
                    duration = len(audio) / sr
                    
                    manifest.append({
                        'audio_id': audio_id,
                        'audio_path': f"audio/synthetic/{audio_id}.wav",
                        'transcription': text,
                        'duration': duration,
                        'label': row['label'],
                        'split': row['split']
                    })
                except:
                    pass
                continue
            
            try:
                # Generate audio using gTTS
                tts = gTTS(text=text, lang=self.lang, slow=False)
                tts.save(str(audio_path))
                
                # Get audio duration
                try:
                    import librosa
                    audio, sr = librosa.load(str(audio_path), sr=16000)
                    duration = len(audio) / sr
                except:
                    # Estimate duration (rough: 3 chars per second)
                    duration = len(text) / 3.0
                
                # Add to manifest
                manifest.append({
                    'audio_id': audio_id,
                    'audio_path': f"audio/synthetic/{audio_id}.wav",
                    'transcription': text,
                    'duration': duration,
                    'label': row['label'],
                    'split': row['split']
                })
                
                # Rate limiting (gTTS has limits, be gentle)
                time.sleep(0.15)
                
            except Exception as e:
                error_msg = f"Error generating {audio_id}: {str(e)}"
                errors.append(error_msg)
                if len(errors) <= 10:  # Only print first 10 errors
                    print(f"\n{error_msg}")
                continue
        
        # Save manifest
        manifest_df = pd.DataFrame(manifest)
        manifest_path = self.output_dir.parent / "audio_manifest.csv"
        
        # If resuming, append to existing manifest
        if manifest_path.exists() and start_from > 0:
            existing_df = pd.read_csv(manifest_path)
            manifest_df = pd.concat([existing_df, manifest_df], ignore_index=True)
            manifest_df = manifest_df.drop_duplicates(subset=['audio_id'])
        
        manifest_df.to_csv(manifest_path, index=False)
        
        # Print statistics
        print("\n" + "="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        print(f"✓ Generated: {len(manifest)} audio files")
        print(f"✓ Errors: {len(errors)}")
        print(f"✓ Manifest: {manifest_path}")
        
        if len(manifest) > 0:
            total_duration = manifest_df['duration'].sum()
            print(f"\nDataset Statistics:")
            print(f"  Total samples: {len(manifest_df)}")
            print(f"  Total duration: {total_duration/3600:.2f} hours")
            print(f"  Average duration: {manifest_df['duration'].mean():.2f} seconds")
            
            # Split statistics
            print(f"\nSplit Distribution:")
            for split in ['train', 'validation', 'test']:
                split_df = manifest_df[manifest_df['split'] == split]
                split_duration = split_df['duration'].sum()
                print(f"  {split}: {len(split_df)} samples ({split_duration/3600:.2f} hours)")
        
        return manifest_df


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic Konkani audio')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to generate (default: all)')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Start from this index (for resuming)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: generate only 10 samples')
    
    args = parser.parse_args()
    
    # Test mode
    if args.test:
        print("\n🧪 TEST MODE: Generating 10 samples only\n")
        args.max_samples = 10
    
    # Create generator
    generator = SyntheticAudioGenerator()
    
    # Generate dataset
    try:
        manifest = generator.generate_dataset(
            max_samples=args.max_samples,
            start_from=args.start_from
        )
        
        print("\n✓ Synthetic audio dataset ready!")
        print("\nNext steps:")
        print("  1. Verify audio quality: play a few samples")
        print("  2. Train ASR model: python src/training/train_asr_synthetic.py")
        print("  3. Integrate with sentiment: python src/inference/test_audio_sentiment.py")
        
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user")
        print("You can resume with: --start-from <last_index>")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
