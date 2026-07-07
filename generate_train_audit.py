#!/usr/bin/env python3
"""
Generate Train Audit CSV
Runs the 88-hour training set (data/konkani-ultimate/train.json) through the Conformer CTC model
using Greedy Decoding and outputs to train_audit.csv with hyp_greedy and ref columns.
"""
import json, os, sys, time, csv
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from models.conformer_ctc import ConformerCTC

# Paths
CHECKPOINT   = BASE / "outputs/conformer_ctc_run1/best_conformer_ctc.pt"
VOCAB_FILE   = BASE / "data/konkani-10k/vocab.json"
TRAIN_MANIFEST = BASE / "data/konkani-ultimate/train.json"
OUTPUT_CSV   = BASE / "train_audit.csv"

# ── Loader ───────────────────────────────────────────────────────────────────
def process_audio(path, device):
    try:
        import librosa
        audio, s = librosa.load(path, sr=16000)
        wav = torch.FloatTensor(audio)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.size(0) > 1: wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0).to(device)
        
        import torchaudio.transforms as T
        mel_fn = T.MelSpectrogram(16000, n_mels=80, n_fft=400, hop_length=160, win_length=400).to(device)
        mel = mel_fn(wav.unsqueeze(0))
        mel = mel.transpose(1, 2)
        mel = torch.log(mel + 1e-9)
        mel_len = torch.tensor([(wav.size(0) // 160) + 1], device=device)
        return mel.float(), mel_len
    except Exception as e: 
        return None, None

# ── Tokenizer ─────────────────────────────────────────────────────────────────
class CharTokenizer:
    def __init__(self):
        v = json.load(open(VOCAB_FILE, encoding='utf-8'))
        self.idx2char = {int(k): c for k, c in v['idx2char'].items()}
        self.vocab_size = v['vocab_size']
        self.blank_id = 0

    def decode(self, ids):
        chars = []
        prev = -1
        for i in ids:
            if i != self.blank_id and i != prev:
                chars.append(self.idx2char.get(i, ''))
            prev = i
        return "".join(chars).strip()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Train Audit on: {device}")
    
    if not CHECKPOINT.exists():
        print(f"Error: Checkpoint not found at {CHECKPOINT}")
        sys.exit(1)
        
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    v_size = state['ctc_head.weight'].shape[0]
    
    print(f"Loading Model (Vocab Size: {v_size})...")
    model = ConformerCTC(vocab_size=v_size, input_dim=80, d_model=256, num_layers=12)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    
    tok = CharTokenizer()
    
    print(f"Loading Manifest: {TRAIN_MANIFEST}")
    samples = [json.loads(l) for l in open(TRAIN_MANIFEST, encoding='utf-8')]
    print(f"Total Samples to Process: {len(samples)}")
    
    print(f"Writing to {OUTPUT_CSV}...")
    
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["audio_filepath", "ref", "hyp_greedy"])
        
        # We will track processed count and failures
        processed = 0
        failures = 0
        
        # You can adjust batching or keep it simple loop if avoiding OOM is priority
        for s in tqdm(samples, desc="Processing Train Set"):
            path = s.get('audio_filepath', '')
            ref_text = s.get('text', '').strip()
            
            mel, mel_len = process_audio(path, device)
            
            if mel is None:
                failures += 1
                continue
                
            with torch.no_grad():
                logits, _ = model(mel, mel_len)
            
            ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
            hyp_text = tok.decode(ids)
            
            writer.writerow([path, ref_text, hyp_text])
            processed += 1
            
            # Optional: flush every 1000 lines so you don't lose data if it crashes
            if processed % 1000 == 0:
                f.flush()
                
    print(f"\nCompleted! Processed {processed} files. Failed: {failures}.")
    print(f"Output saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
