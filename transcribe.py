import torch
import torch.nn.functional as F
import torchaudio
import json
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.conformer_ctc import create_model
from scripts.beam_search_decoder import BeamSearchDecoder

class LongFormInference:
    """ASR Inference for long audio files using sliding window and Beam Search + KenLM"""
    
    def __init__(self, checkpoint_path, vocab_path, lm_path=None, unigram_path=None, device='cuda'):
        # 1. Setup Device
        if device == 'mps' and not torch.backends.mps.is_available():
            device = 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        self.device = torch.device(device)
        
        # 2. Load Model
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint)
        
        # Determine architecture
        cfg = checkpoint.get('config', {})
        d_model = cfg.get('d_model', 256)
        num_layers = cfg.get('num_layers', 12)
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            v_data = json.load(f)
        v_size = v_data.get('vocab_size', 79)
        
        self.model = create_model(vocab_size=v_size, d_model=d_model, num_layers=num_layers)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()
        
        # 3. Setup Mel Transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=80, n_fft=400, hop_length=160
        ).to(self.device)
        
        # 4. Setup Decoder (Optimal params Alpha=1.0, Beta=1.0 from Audit)
        self.decoder = BeamSearchDecoder(vocab_path, lm_path, unigram_path=unigram_path, alpha=1.0, beta=1.0)
        print(f"Successfully loaded Konkani ASR with {'LM' if lm_path else 'Greedy'} decoding.")

    def transcribe_long(self, audio_path, chunk_sec=10.0, overlap_sec=1.5):
        """Processes long audio in overlapping chunks to maintain focus and 12% accuracy"""
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        full_audio = waveform.squeeze(0).numpy()
        total_samples = len(full_audio)
        chunk_samples = int(chunk_sec * 16000)
        overlap_samples = int(overlap_sec * 16000)
        step_samples = chunk_samples - overlap_samples
        
        full_transcript = []
        
        # Iterate over audio in chunks
        for start in range(0, total_samples, step_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = full_audio[start:end]
            
            # Pad if too short for model
            if len(chunk) < 1600: continue
            
            chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                mel = self.mel_transform(chunk_tensor)
                mel = torch.log(mel.transpose(1, 2) + 1e-9)
                logits, _ = self.model(mel)
                
                # Use Beam Search + LM
                chunk_text = self.decoder.beam_search_decode(logits.squeeze(0), beam_width=15)
                
                if chunk_text.strip():
                    full_transcript.append(chunk_text.strip())
            
            if end == total_samples: break
            
        return " ".join(full_transcript)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Konkani Long-Form Transcription')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio/video file')
    parser.add_argument('--checkpoint', type=str, default='outputs/conformer_ctc_run1/best_conformer_ctc.pt')
    parser.add_argument('--vocab', type=str, default='data/konkani-10k/vocab.json')
    parser.add_argument('--lm', type=str, default='models/language_models/konkani_3gram.binary')
    parser.add_argument('--unigrams', type=str, default='models/language_models/unigrams.txt')
    parser.add_argument('--device', type=str, default='mps')
    
    args = parser.parse_args()
    
    inf = LongFormInference(args.checkpoint, args.vocab, args.lm, args.unigrams, args.device)
    result = inf.transcribe_long(args.audio)
    
    print("\n--- TRANSCRIPTION ---\n")
    print(result)
    print("\n---------------------\n")
