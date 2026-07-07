import torch
import json
from pathlib import Path
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import numpy as np

# Use pyctcdecode for beam search + KenLM
from pyctcdecode import build_ctcdecoder

class BeamSearchDecoder:
    def __init__(self, vocab_path, lm_path=None, unigram_path=None, alpha=0.5, beta=1.5):
        """
        Initialize CTC Beam Search Decoder with optional KenLM support
        """
        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            self.idx2char = {int(k): c for k, c in vocab_data['idx2char'].items()}
            self.vocab_size = vocab_data['vocab_size']
        
        # Create label list for pyctcdecode
        self.labels = [self.idx2char.get(i, "") for i in range(self.vocab_size)]
        
        # In CTC models, the blank token is usually at index 0 or index vocab_size-1
        self.blank_idx = 0
        self.labels[self.blank_idx] = ""
        
        # Initialize decoder (Standard Library Decoder for LM)
        if lm_path and Path(lm_path).exists():
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
            self.decoder = build_ctcdecoder(labels=self.labels)

    def greedy_decode(self, log_probs):
        """Standard Greedy CTC Decoding"""
        arg_maxes = torch.argmax(log_probs, dim=-1)
        decode_ids = []
        prev_id = -1
        for i in arg_maxes.tolist():
            if i != self.blank_idx and i != prev_id:
                decode_ids.append(i)
            prev_id = i
        return "".join([self.labels[i] for i in decode_ids])

    def beam_search_decode(self, log_probs, beam_width=10, lm_weight=1.0, 
                           length_norm=0.0, space_reward=0.0, repeat_penalty=0.0):
        """
        Standard Manual Beam Search
        """
        T_len, V_len = log_probs.shape
        # Initialize beams: {prefix_tuple: (log_p_blank, log_p_non_blank)}
        # We use LogAddExp for numerical stability
        beams = {('',): (0.0, -float('inf'))}
        
        for t in range(T_len):
            new_beams = {}
            lp = log_probs[t]
            
            # Prune indices to keep search fast
            top_indices = torch.argsort(lp, descending=True)[:beam_width].tolist()
            
            for prefix, (p_b, p_nb) in beams.items():
                # 1. Prediction is blank
                p_blank_curr = lp[self.blank_idx].item()
                n_p_b, n_p_nb = new_beams.get(prefix, (-float('inf'), -float('inf')))
                new_beams[prefix] = (np.logaddexp(n_p_b, np.logaddexp(p_b, p_nb) + p_blank_curr), n_p_nb)
                
                # 2. Prediction is character
                for idx in top_indices:
                    if idx == self.blank_idx: continue
                    char = self.labels[idx]
                    p_char_curr = lp[idx].item()
                    
                    if prefix and prefix[-1] == char:
                        # Case: Repeating character
                        # a) With blank in between (Becomes new prefix)
                        new_prefix = prefix + (char,)
                        n_p_b, n_p_nb = new_beams.get(new_prefix, (-float('inf'), -float('inf')))
                        new_beams[new_prefix] = (n_p_b, np.logaddexp(n_p_nb, p_b + p_char_curr))
                        
                        # b) Without blank (Stays in current prefix)
                        n_p_b, n_p_nb = new_beams.get(prefix, (-float('inf'), -float('inf')))
                        new_beams[prefix] = (n_p_b, np.logaddexp(n_p_nb, p_nb + p_char_curr))
                    else:
                        # Case: New character
                        new_prefix = prefix + (char,)
                        n_p_b, n_p_nb = new_beams.get(new_prefix, (-float('inf'), -float('inf')))
                        new_beams[new_prefix] = (n_p_b, np.logaddexp(n_p_nb, np.logaddexp(p_b, p_nb) + p_char_curr))
            
            # Prune to beam width
            def get_search_score(p_tuple, probs):
                p_b, p_nb = probs
                raw_score = np.logaddexp(p_b, p_nb)
                # Standard additive hacks (more stable than division in Python loop)
                length = len(p_tuple)
                if length == 0: return raw_score
                return raw_score + (length * length_norm) + (p_tuple.count(' ') * space_reward)

            beams = dict(sorted(new_beams.items(), key=lambda x: get_search_score(x[0], x[1]), reverse=True)[:beam_width])
            
        # Final scoring (LM + Length Norm)
        candidates = []
        for prefix, (p_b, p_nb) in beams.items():
            sentence = "".join(prefix)
            acoustic_score = np.logaddexp(p_b, p_nb)
            
            final_score = acoustic_score
            
            if hasattr(self.decoder, 'language_model') and self.decoder.language_model:
                lm_score = self.decoder.language_model.score(sentence)
                final_score += lm_weight * lm_score
            
            # Length normalization (additive log length)
            if length_norm > 0:
                final_score += len(sentence) * length_norm
                
            candidates.append((sentence, final_score))
            
        # Select best
        return max(candidates, key=lambda x: x[1])[0]

def load_model(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    vocab_size = state_dict['ctc_head.weight'].shape[0]
    
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from models.conformer_ctc import create_model
    model = create_model(vocab_size)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, vocab_size

def extract_features(audio_path, device='cpu'):
    import soundfile as sf
    waveform_np, sample_rate = sf.read(audio_path)
    waveform = torch.from_numpy(waveform_np).float()
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
    
    if sample_rate != 16000:
        resampler = T.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=80)
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(mel_spec + 1e-9)
    features = mel_spec.squeeze(0).transpose(0, 1).unsqueeze(0)
    return features.to(device)

def decode_audio(model, audio_path, decoder, beam_width=None, device='cpu'):
    features = extract_features(audio_path, device)
    with torch.no_grad():
        encoder_out, _ = model.encoder(features)
        ctc_logits = model.ctc_head(encoder_out)
        log_probs = F.log_softmax(ctc_logits, dim=-1)
        log_probs = log_probs.squeeze(0)
    
    if beam_width is None:
        text = decoder.greedy_decode(log_probs)
    else:
        text = decoder.beam_search_decode(log_probs, beam_width=beam_width)
    return text
