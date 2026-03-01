import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import json
import os
from models.conformer_ctc import create_model
from pathlib import Path

# Paths
PHASE1_CKPT = 'outputs/conformer_ctc_run1/best_conformer_ctc.pt'
PHASE2_CKPT = 'outputs/conformer_ctc_chunk6/best_stage2_model.pt'
VOCAB_PATH = 'data/konkani-10k/vocab.json'
MANIFEST_PATH = 'data/konkani-10k/test_manifest.json'

class InferenceEngine:
    def __init__(self, ckpt_path, vocab_path, device='cuda'):
        self.device = device
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.char2idx = vocab['char2idx']
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        vocab_size = len(self.char2idx)
        
        # Load model architecture info from checkpoint if possible, else use default
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Determine config
        d_model = 256
        num_layers = 12
        if 'config' in checkpoint:
            d_model = checkpoint['config'].get('d_model', 256)
            num_layers = checkpoint['config'].get('num_layers', 12)
            
        self.model = create_model(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(device)
        self.model.eval()
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=80, n_fft=400, hop_length=160
        ).to(device)

    @torch.no_grad()
    def transcribe(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=16000)
        audio = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
        
        mel = self.mel_transform(audio).transpose(1, 2)
        mel = torch.log(mel + 1e-9)
        mel_lens = torch.LongTensor([mel.size(1)]).to(self.device)
        
        logits, _ = self.model(mel, mel_lens)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)[0]
        
        # Greedy CTC decoding
        chars = []
        prev = -1
        for idx in preds.tolist():
            if idx != prev and idx != 0: # 0 is blank
                chars.append(self.idx2char.get(idx, ''))
            prev = idx
        return "".join(chars)

def remap_path(unix_path):
    # Simple remap for this script
    base = Path("E:/konkani")
    parts = unix_path.split("/")
    if "konkani-10k" in unix_path:
        rel = unix_path.split("konkani-10k/")[-1]
        return str(base / "data" / "konkani-10k" / rel.replace("/", os.sep))
    elif "KonkaniRawSpeechCorpus" in unix_path:
        rel = unix_path.split("KonkaniRawSpeechCorpus/")[-1]
        return str(base / "KonkaniRawSpeechCorpus" / rel.replace("/", os.sep))
    return unix_path

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Comparing models on {device}...")
    
    eng1 = InferenceEngine(PHASE1_CKPT, VOCAB_PATH, device)
    eng2 = InferenceEngine(PHASE2_CKPT, VOCAB_PATH, device)
    
    # Load 10 samples from test manifest
    samples = []
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= 10: break
            
    print("\n" + "="*100)
    print(f"{'TARGET':<30} | {'PHASE 1 (Pilot)':<30} | {'PHASE 2 (Chunk 6)':<30}")
    print("-" * 100)
    
    for s in samples:
        target = s['text']
        audio_p = remap_path(s['audio_filepath'])
        if not os.path.exists(audio_p):
            continue
            
        p1 = eng1.transcribe(audio_p)
        p2 = eng2.transcribe(audio_p)
        
        print(f"{target[:30]:<30} | {p1[:30]:<30} | {p2[:30]:<30}")

if __name__ == "__main__":
    main()
