#!/usr/bin/env python3
"""
ASR with Language Model post-processing
Improves letter-by-letter predictions using word dictionary
"""
import torch
import json
from pathlib import Path
import sys
from difflib import get_close_matches

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.test_best_model import ASRInference


class LanguageModelCorrector:
    """Simple language model for correcting ASR outputs"""
    
    def __init__(self, dictionary_path=None):
        """
        Args:
            dictionary_path: Path to Konkani word dictionary (one word per line)
        """
        self.dictionary = set()
        
        if dictionary_path and Path(dictionary_path).exists():
            with open(dictionary_path, 'r', encoding='utf-8') as f:
                self.dictionary = set(line.strip() for line in f if line.strip())
            print(f"Loaded {len(self.dictionary)} words from dictionary")
        else:
            print("No dictionary provided - will use basic corrections only")
    
    def correct_word(self, word, max_distance=2):
        """
        Correct a single word using dictionary
        
        Args:
            word: Word to correct
            max_distance: Maximum edit distance for suggestions
        
        Returns:
            Corrected word or original if no match
        """
        if not word or not self.dictionary:
            return word
        
        # If word is in dictionary, return as-is
        if word in self.dictionary:
            return word
        
        # Find close matches
        matches = get_close_matches(word, self.dictionary, n=1, cutoff=0.8)
        
        if matches:
            return matches[0]
        
        return word
    
    def correct_text(self, text):
        """
        Correct entire text
        
        Args:
            text: Text to correct
        
        Returns:
            Corrected text
        """
        if not text:
            return text
        
        # Split into words
        words = text.split()
        
        # Correct each word
        corrected_words = [self.correct_word(word) for word in words]
        
        return ' '.join(corrected_words)
    
    def add_word_boundaries(self, text):
        """
        Add spaces between words if missing
        Uses simple heuristics for Devanagari
        """
        if not text:
            return text
        
        # For now, return as-is
        # TODO: Implement word boundary detection for Konkani
        return text


class ASRWithLanguageModel:
    """ASR with language model post-processing"""
    
    def __init__(self, checkpoint_path, vocab_path='data/vocab.json', 
                 dictionary_path=None):
        """
        Args:
            checkpoint_path: Path to ASR model checkpoint
            vocab_path: Path to vocabulary
            dictionary_path: Path to Konkani dictionary
        """
        # Load ASR model
        self.asr = ASRInference(checkpoint_path, vocab_path)
        
        # Load language model
        self.lm = LanguageModelCorrector(dictionary_path)
    
    def transcribe(self, audio_path):
        """
        Transcribe audio with language model correction
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            tuple: (raw_transcription, corrected_transcription)
        """
        # Get raw ASR output
        raw_text, tokens = self.asr.transcribe(audio_path)
        
        # Apply language model correction
        corrected_text = self.lm.correct_text(raw_text)
        
        return raw_text, corrected_text


def create_dictionary_from_training_data(manifest_path, output_path):
    """
    Create word dictionary from training data
    
    Args:
        manifest_path: Path to training manifest
        output_path: Path to save dictionary
    """
    words = set()
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '')
            
            # Extract words
            for word in text.split():
                # Remove punctuation
                word = word.strip('.,!?;:()[]{}')
                if word:
                    words.add(word)
    
    # Save dictionary
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in sorted(words):
            f.write(f"{word}\n")
    
    print(f"Created dictionary with {len(words)} unique words")
    print(f"Saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ASR with Language Model')
    parser.add_argument('--checkpoint', type=str, 
                       default='kaggle_asr_outputs/checkpoints/checkpoint_epoch_27.pt',
                       help='Path to ASR checkpoint')
    parser.add_argument('--audio', type=str, nargs='+',
                       help='Audio files to transcribe')
    parser.add_argument('--create-dict', action='store_true',
                       help='Create dictionary from training data')
    parser.add_argument('--manifest', type=str,
                       default='data/konkani-asr-v0/splits/manifests/train.json',
                       help='Training manifest for dictionary creation')
    parser.add_argument('--dictionary', type=str,
                       default='data/konkani_dictionary.txt',
                       help='Path to Konkani dictionary')
    
    args = parser.parse_args()
    
    # Create dictionary if requested
    if args.create_dict:
        print("Creating dictionary from training data...")
        create_dictionary_from_training_data(args.manifest, args.dictionary)
        return
    
    # Initialize ASR with language model
    print("Loading ASR model with language model...")
    asr_lm = ASRWithLanguageModel(
        args.checkpoint,
        dictionary_path=args.dictionary if Path(args.dictionary).exists() else None
    )
    
    # Transcribe audio files
    if not args.audio:
        print("No audio files specified. Use --audio to specify files.")
        return
    
    print("\n" + "="*70)
    print("TRANSCRIPTION WITH LANGUAGE MODEL")
    print("="*70 + "\n")
    
    for audio_path in args.audio:
        print(f"Audio: {audio_path}")
        raw, corrected = asr_lm.transcribe(audio_path)
        print(f"  Raw:       {raw if raw else '(empty)'}")
        print(f"  Corrected: {corrected if corrected else '(empty)'}")
        print()


if __name__ == '__main__':
    main()
