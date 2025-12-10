"""ASR Model Wrapper for Deployment"""
import torch
from pathlib import Path
import numpy as np

from .konkanivani_asr import KonkaniVaniASR


class ASRModel:
    """Wrapper for ASR model inference"""
    
    def __init__(self, checkpoint_path=None, device=None):
        """Initialize ASR model"""
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        
        # Resolve checkpoint path
        if checkpoint_path is None:
            # Try multiple possible locations
            base_dir = Path(__file__).parent.parent.parent
            possible_paths = [
                base_dir / "kaggle_best_model" / "checkpoints" / "best_model.pt",
                base_dir / "kaggle_asr_outputs" / "checkpoints" / "best_model.pt",
                Path("kaggle_best_model/checkpoints/best_model.pt"),
                Path("kaggle_asr_outputs/checkpoints/best_model.pt"),
            ]
            
            for path in possible_paths:
                if path.exists():
                    checkpoint_path = str(path)
                    break
            
            if checkpoint_path is None:
                raise FileNotFoundError(
                    "Could not find ASR checkpoint. Tried:\n" + 
                    "\n".join(f"  - {p}" for p in possible_paths)
                )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get vocab_size from checkpoint or infer from model
        if 'vocab_size' in checkpoint:
            vocab_size = checkpoint['vocab_size']
        else:
            # Infer from model state dict (ctc_head or embedding layer)
            state_dict = checkpoint['model_state_dict']
            # Remove 'module.' prefix if present (from DataParallel)
            ctc_key = 'ctc_head.weight' if 'ctc_head.weight' in state_dict else 'module.ctc_head.weight'
            vocab_size = state_dict[ctc_key].shape[0]
        
        # Create model
        self.model = KonkaniVaniASR(
            vocab_size=vocab_size,
            input_dim=80,
            d_model=256,
            encoder_layers=12,
            decoder_layers=6
        )
        
        # Load state dict (handle DataParallel wrapper)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            # Remove 'module.' prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Load vocabulary from checkpoint or from file
        if 'idx_to_char' in checkpoint and 'char_to_idx' in checkpoint:
            self.idx_to_char = checkpoint['idx_to_char']
            self.char_to_idx = checkpoint['char_to_idx']
        else:
            # Load from vocab.json file
            import json
            base_dir = Path(__file__).parent.parent.parent
            vocab_paths = [
                base_dir / "data" / "vocab.json",
                Path("data/vocab.json"),
            ]
            
            vocab_file = None
            for vp in vocab_paths:
                if vp.exists():
                    vocab_file = vp
                    break
            
            if vocab_file is None:
                raise FileNotFoundError("Could not find vocab.json file")
            
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            self.char_to_idx = vocab_data['char2idx']
            # Convert string keys to int for idx_to_char
            self.idx_to_char = {int(k): v for k, v in vocab_data['idx2char'].items()}
    
    def transcribe(self, audio_path):
        """
        Transcribe audio file to Konkani text
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            transcription: Konkani text
        """
        # Load and preprocess audio using soundfile directly
        import soundfile as sf
        import numpy as np
        
        # Load audio with soundfile
        audio_data, sample_rate = sf.read(audio_path)
        
        # Convert to torch tensor
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        waveform = torch.from_numpy(audio_data).float()
        
        # Add channel dimension if mono
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            # Transpose if needed (samples, channels) -> (channels, samples)
            waveform = waveform.transpose(0, 1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import librosa
            # Convert to numpy for resampling
            waveform_np = waveform.squeeze(0).numpy()
            waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=16000)
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
            sample_rate = 16000
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract mel-spectrogram using librosa
        import librosa
        waveform_np = waveform.squeeze(0).numpy()
        mel_spec_np = librosa.feature.melspectrogram(
            y=waveform_np,
            sr=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80
        )
        mel_spec = torch.from_numpy(mel_spec_np).float()
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Transpose to (time, features)
        mel_spec = mel_spec.squeeze(0).transpose(0, 1)
        
        # Add batch dimension
        mel_spec = mel_spec.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model.recognize(mel_spec)
        
        # Decode predictions
        transcription = self._decode_predictions(predictions[0])
        
        return transcription
    
    def _decode_predictions(self, predictions):
        """Decode token predictions to text"""
        # Remove blanks and duplicates (CTC decoding)
        decoded = []
        prev_token = None
        
        for token in predictions.cpu().numpy():
            if token != 0 and token != prev_token:  # 0 is blank token
                if token in self.idx_to_char:
                    decoded.append(self.idx_to_char[token])
            prev_token = token
        
        return ''.join(decoded)
