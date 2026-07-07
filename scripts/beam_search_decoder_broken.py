#!/usr/bin/env python3
"""
Beam Search Decoder with KenLM Language Model Integration
==========================================================
Implements greedy and beam search decoding for CTC models with optional LM fusion.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import torchaudio
from pyctcdecode import build_ctcdecoder


class BeamSearchDecoder:
    """Beam search decoder with optional KenLM language model"""
    
    def __init__(self, vocab_path, lm_path=None, unigram_path=None, alpha=1.0, beta=1.0):
        """
        Args:
            vocab_path: Path to vocab.json file
            lm_path: Path to KenLM binary file (optional)
            alpha: Language model weight
            beta: Word insertion bonus
        """
        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Handle different vocab formats
        if 'piece2id' in vocab_data:
            self.vocab = vocab_data['piece2id']
        elif 'char2idx' in vocab_data:
            self.vocab = vocab_data['char2idx']
        else:
            self.vocab = vocab_data
        
        # Create labels list matching model output size
        # Model outputs vocab_size classes, so labels must match exactly
        vocab_size = vocab_data.get('vocab_size', len(self.vocab))
        self.labels = [f'_UNK{i}_' for i in range(vocab_size)]  # Default placeholders
        
        # Track blank index for CTC
        # CRITICAL: Training uses blank=0 (mapped to <pad> in json, but index 0 in loss)
        self.blank_idx = 0 
        
        for char, idx_val in self.vocab.items():
            idx = int(idx_val)
            if idx >= vocab_size:
                continue
            
            if idx == self.blank_idx:
                self.labels[idx] = '' # CTC Blank
            elif len(char) == 1:
                self.labels[idx] = char.replace('▁', ' ')
            else:
                # Use unique non-printing chars for special tokens (pyctcdecode compatibility)
                self.labels[idx] = chr(0xE000 + idx)
        
        # Build decoder
        if lm_path and Path(lm_path).exists():
            print(f"Building decoder with LM: {lm_path}")
            print(f"  alpha (LM weight): {alpha}")
            print(f"  beta (word bonus): {beta}")
            unigrams = None
            if unigram_path and Path(unigram_path).exists():
                with open(unigram_path, 'r', encoding='utf-8') as f:
                    unigrams = [line.strip() for line in f if line.strip()]
            
            self.decoder = build_ctcdecoder(
                labels=self.labels,
                kenlm_model_path=str(lm_path),
                unigrams=unigrams,
                alpha=alpha,
                beta=beta
            )
        else:
            print("Building decoder without LM")
            self.decoder = build_ctcdecoder(labels=self.labels)
        
        print(f"  Decoder initialized with {len(self.labels)} labels")
        print(f"  Blank index: {self.blank_idx}")

    def decode_long_audio(self, model, audio_path, device, beam_width=None, chunk_sec=10.0, overlap_sec=1.5):
        """
        Inference with sliding window for long audio files
        """
        import torch
        import torchaudio
        
        # 1. Load Audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Merge to mono if necessary
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.to(device)
        num_samples = waveform.shape[1]
        
        samples_per_chunk = int(chunk_sec * sample_rate)
        overlap_samples = int(overlap_sec * sample_rate)
        step_samples = samples_per_chunk - overlap_samples
        
        # Feature extractor needs to be clean
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=80, n_fft=400, hop_length=160
        ).to(device)
        
        import torch.nn.functional as F

    def decode_long_audio(self, model, audio_path, device, beam_width=None, chunk_sec=10.0, overlap_sec=3.0):
        """
        Inference with long-form audio using codebase's original feature extraction
        """
        import torch
        import torchaudio
        import soundfile as sf
        
        # 1. Load Audio
        waveform_np, sr = sf.read(audio_path)
        if len(waveform_np.shape) > 1:
            waveform_np = waveform_np.mean(axis=1) # Mono
        
        num_samples = len(waveform_np)
        samples_per_chunk = int(chunk_sec * 16000)
        overlap_samples = int(overlap_sec * 16000)
        step_samples = samples_per_chunk - overlap_samples
        
        # Helper: Extract using the exact logic that worked for greedy
        def _get_chunk_logits(chunk_wav):
            # Same processing as extract_features
            wav = torch.from_numpy(chunk_wav).float().to(device)
            if wav.ndim == 1: wav = wav.unsqueeze(0)
            
            # Use T.MelSpectrogram identical to extract_features
            import torchaudio.transforms as T
            mel_fn = T.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=80).to(device)
            mel = mel_fn(wav)
            mel = (mel + 1e-9).log()
            # (1, 80, time) -> (1, time, 80)
            features = mel.squeeze(0).transpose(0, 1).unsqueeze(0)
            
            with torch.no_grad():
                output = model(features)
                logits = output[0] if isinstance(output, tuple) else output
                return F.log_softmax(logits, dim=-1).squeeze(0)

        # If audio fits in one chunk
        if num_samples <= samples_per_chunk:
            logits = _get_chunk_logits(waveform_np)
            return self.beam_search_decode(logits, beam_width=beam_width) if beam_width else self.greedy_decode(logits)
        
        # Sliding Window
        all_transcripts = []
        for start in range(0, num_samples - samples_per_chunk + 1, step_samples):
            end = start + samples_per_chunk
            chunk = waveform_np[start:end]
            logits = _get_chunk_logits(chunk)
            
            text = self.beam_search_decode(logits, beam_width=beam_width) if beam_width else self.greedy_decode(logits)
            if text:
                all_transcripts.append(text)
        
        # Stitching with basic deduplication
        if not all_transcripts: return ""
        
        final_words = []
        for i, t in enumerate(all_transcripts):
            words = t.split()
            if i == 0:
                final_words.extend(words)
            else:
                # Add words that aren't already in the tail of our transcript
                tail = final_words[-5:] if len(final_words) > 5 else final_words
                for w in words:
                    if w not in tail:
                        final_words.append(w)
        
        return " ".join(final_words)
    
    def greedy_decode(self, logits):
        """
        Greedy CTC decoding
        
        Args:
            logits: (time, vocab_size) - log probabilities
        
        Returns:
            text: Decoded string
        """
        # Get best token at each timestep
        tokens = logits.argmax(dim=-1).cpu().numpy()
        
        # Remove blanks and consecutive duplicates
        decoded = []
        prev_token = None
        for token in tokens:
            # Skip blank and special tokens
            if token == self.blank_idx:
                prev_token = token
                continue
            
            # Skip if same as previous (CTC collapse)
            if token == prev_token:
                continue
            
            # Add character if it's not a special token placeholder
            if token < len(self.labels):
                char = self.labels[token]
                # Skip special token placeholders
                if not (char.startswith('_') and char.endswith('_')):
                    decoded.append(char)
            
            prev_token = token
        
        return ''.join(decoded)
    
    def beam_search_decode(self, logits, beam_width=15):
        """
        Beam search decoding with optional LM
        
        Args:
            logits: (time, vocab_size) - log probabilities
            beam_width: Beam width for search
        
        Returns:
            text: Decoded string
        """
        # Convert logits to probabilities
        logits_np = logits.cpu().numpy()
        
        # Decode with beam search
        text = self.decoder.decode(logits_np, beam_width=beam_width)
        
        # Post-process: remove special token placeholders
        # Remove _PAD_, _UNK_, _SOS_, _EOS_, etc.
        import re
        text = re.sub(r'_[A-Z]+\d*_', '', text)
        
        return text


def load_model(model_path, device='cpu'):
    """
    Load trained ASR model
    
    Args:
        model_path: Path to checkpoint
        device: Device to load model on
    
    Returns:
        model: Loaded model
        vocab_size: Vocabulary size
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        vocab_size = checkpoint.get('vocab_size', None)
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        vocab_size = checkpoint.get('vocab_size', None)
    else:
        state_dict = checkpoint
        vocab_size = None
    
    # Infer vocab size from CTC head if not in checkpoint
    if vocab_size is None:
        for key in state_dict.keys():
            if 'ctc_head' in key and 'weight' in key:
                vocab_size = state_dict[key].shape[0]
                break
    
    print(f"  Vocab size: {vocab_size}")
    
    # Import model class
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from models.conformer_ctc import create_model
    
    # Create model
    model = create_model(vocab_size)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model, vocab_size


def extract_features(audio_path, device='cpu'):
    """
    Extract mel-spectrogram features from audio
    
    Args:
        audio_path: Path to audio file
        device: Device for computation
    
    Returns:
        features: (1, time, features) tensor
    """
    import soundfile as sf
    import torch
    import torchaudio.transforms as T
    
    # Load audio with soundfile directly
    waveform_np, sample_rate = sf.read(audio_path)
    
    # Convert to torch tensor
    waveform = torch.from_numpy(waveform_np).float()
    
    # Handle mono/stereo
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension
    else:
        waveform = waveform.T  # (samples, channels) -> (channels, samples)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = T.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Extract mel-spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80
    )
    mel_spec = mel_transform(waveform)
    
    # Log scale
    mel_spec = torch.log(mel_spec + 1e-9)
    
    # Transpose to (time, features) and add batch dimension
    features = mel_spec.squeeze(0).transpose(0, 1).unsqueeze(0)
    
    return features.to(device)


def decode_audio(model, audio_path, decoder, beam_width=None, device='cpu'):
    """
    Decode audio file using specified decoder
    
    Args:
        model: ASR model
        audio_path: Path to audio file
        decoder: BeamSearchDecoder instance
        beam_width: Beam width (None for greedy)
        device: Device for computation
    
    Returns:
        text: Decoded text
    """
    return decoder.decode_long_audio(model, audio_path, device, beam_width=beam_width)


def main():
    """Test the decoder"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test beam search decoder')
    parser.add_argument('--model', type=str, default='outputs/conformer_ctc_run1/best_conformer_ctc.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, default='data/konkani-mega-dataset/vocab.json',
                        help='Path to vocab.json')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--lm', type=str, default=None, help='Path to KenLM binary')
    parser.add_argument('--beam-width', type=int, default=15, help='Beam width')
    parser.add_argument('--alpha', type=float, default=1.0, help='LM weight')
    parser.add_argument('--beta', type=float, default=0.0, help='Word bonus')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Load model
    model, vocab_size = load_model(args.model, args.device)
    
    # Create decoder
    decoder = BeamSearchDecoder(args.vocab, args.lm, args.alpha, args.beta)
    
    print("\n" + "="*60)
    print("Testing Decoding Strategies")
    print("="*60)
    
    # Greedy decoding
    print("\n1. Greedy Decoding:")
    text_greedy = decode_audio(model, args.audio, decoder, beam_width=None, device=args.device)
    print(f"   {text_greedy}")
    
    # Beam search
    print(f"\n2. Beam Search (width={args.beam_width}):")
    text_beam = decode_audio(model, args.audio, decoder, beam_width=args.beam_width, device=args.device)
    print(f"   {text_beam}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
