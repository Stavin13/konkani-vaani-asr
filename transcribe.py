import torch
import torch.nn.functional as F
import torchaudio
import json
import os
import random
from models.conformer_ctc import create_model

# ─────────────────────────────────────────────────────────────
# PATH REMAPPING (Copied from training script)
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def remap_path(unix_path):
    for prefix in ["/Volumes/data&proj/konkani/", "/Volumes/data&proj/konkani", "/Volumes/"]:
        if unix_path.startswith(prefix):
            rel = unix_path[len(prefix):]
            candidate = os.path.join(BASE_DIR, rel.replace("/", os.sep))
            if os.path.exists(candidate): return candidate
            
            parts = rel.split("/", 1)
            if len(parts) > 1:
                candidate2 = os.path.join(BASE_DIR, parts[1].replace("/", os.sep))
                if os.path.exists(candidate2): return candidate2
    
    if os.path.exists(unix_path): return unix_path
    
    fname = os.path.basename(unix_path)
    corpus_dir = os.path.join(BASE_DIR, "KonkaniRawSpeechCorpus")
    if os.path.exists(corpus_dir):
        for root, _, files in os.walk(corpus_dir):
            if fname in files:
                return os.path.join(root, fname)
    return ""

class KonkaniInference:
    def __init__(self, checkpoint_path, vocab_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.char2idx = vocab['char2idx']
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        cfg = checkpoint.get('config', {})
        d_model = cfg.get('d_model', 256)
        num_layers = cfg.get('num_layers', 12)
        
        self.model = create_model(vocab_size=len(self.char2idx), d_model=d_model, num_layers=num_layers)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=80, n_fft=400, hop_length=160
        ).to(self.device)
        
        print(f"Loaded model from epoch {checkpoint['epoch']} (Loss: {checkpoint['loss']:.4f})")

    def transcribe(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        waveform = waveform.to(self.device)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        with torch.no_grad():
            mel = self.mel_transform(waveform)
            mel = mel.transpose(1, 2)
            mel = torch.log(mel + 1e-9)
            
            logits, _ = self.model(mel)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)[0]
            
            decoded_chars = []
            prev_idx = -1
            for idx in preds.tolist():
                if idx != prev_idx and idx != 0:
                    decoded_chars.append(self.idx2char.get(idx, ''))
                prev_idx = idx
                
            return "".join(decoded_chars)

if __name__ == "__main__":
    inf = KonkaniInference(
        checkpoint_path='outputs/conformer_ctc_run1/best_conformer_ctc.pt',
        vocab_path='data/konkani-10k/vocab.json'
    )
    
    # Select 10 random files from manifest
    manifest_path = 'data/konkani-10k/train_manifest.json'
    with open(manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    random_samples = random.sample(lines, 10)
    
    print("\n" + "="*80)
    print(f"{'INDEX':<6} | {'TARGET':<35} | {'PREDICTION'}")
    print("="*80)
    
    for i, line in enumerate(random_samples):
        s = json.loads(line)
        local_path = remap_path(s['audio_filepath'])
        target = s['text']
        
        if os.path.exists(local_path):
            pred = inf.transcribe(local_path)
            print(f"#{i+1:<5} | {target:<35} | {pred}")
        else:
            print(f"#{i+1:<5} | [FILE NOT FOUND] {os.path.basename(local_path)}")
    print("="*80)
