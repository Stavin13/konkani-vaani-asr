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
    
    def __init__(self, vocab_path, lm_path=None, alpha=1.0, beta=0.0):
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
        self.blank_idx = vocab_data.get('blank_id', None)
        
        for char, idx_val in self.vocab.items():
            idx = int(idx_val)
            if idx >= vocab_size:
                continue  # Skip indices beyond vocab size
            
            if char == '<blank>':
                self.labels[idx] = ''  # CTC blank (ONLY blank should be empty)
                if self.blank_idx is None:
                    self.blank_idx = idx
            elif char == '<pad>':
                # Pad is like blank, map to empty but remember it's not the CTC blank
                self.labels[idx] = '_PAD_'
            elif char in ['<unk>', '<sos>', '<eos>']:
                # Other special tokens - use unique identifiers
                self.labels[idx] = f'_{char[1:-1].upper()}_'
            else:
                self.labels[idx] = char.replace('▁', ' ')
        
        # If blank_idx is set from json but wasn't '' in labels yet, make sure it's ''
        if self.blank_idx is not None and self.blank_idx < len(self.labels):
            self.labels[self.blank_idx] = ''
        
        # If blank_idx not found, use index 1 (common default)
        if self.blank_idx is None:
            self.blank_idx = 1
            if self.labels[1] != '':
                print(f"WARNING: Blank index {self.blank_idx} is not empty, forcing it")
            self.labels[1] = ''
        
        # Build decoder
        if lm_path and Path(lm_path).exists():
            print(f"Building decoder with LM: {lm_path}")
            print(f"  alpha (LM weight): {alpha}")
            print(f"  beta (word bonus): {beta}")
            self.decoder = build_ctcdecoder(
                labels=self.labels,
                kenlm_model_path=str(lm_path),
                alpha=alpha,
                beta=beta
            )
        else:
            print("Building decoder without LM")
            self.decoder = build_ctcdecoder(labels=self.labels)
        
        print(f"  Decoder initialized with {len(self.labels)} labels")
        print(f"  Blank index: {self.blank_idx}")
    
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
    
    from models.konkanivani_asr import create_konkanivani_model
    
    # Create model
    model = create_konkanivani_model(vocab_size)
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
    # Extract features
    features = extract_features(audio_path, device)
    
    # Get encoder output and CTC logits
    with torch.no_grad():
        encoder_out, _ = model.encoder(features)
        ctc_logits = model.ctc_head(encoder_out)
        
        # Apply log softmax
        log_probs = F.log_softmax(ctc_logits, dim=-1)
        
        # Remove batch dimension: (1, time, vocab) -> (time, vocab)
        log_probs = log_probs.squeeze(0)
    
    # Decode
    if beam_width is None:
        text = decoder.greedy_decode(log_probs)
    else:
        text = decoder.beam_search_decode(log_probs, beam_width=beam_width)
    
    return text


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
