#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
import librosa
from tqdm import tqdm
import multiprocessing

BASE = Path("/Volumes/data&proj/konkani")
sys.path.insert(0, str(BASE))

from models.conformer_ctc import ConformerCTC

CHECKPOINT   = BASE / "outputs/conformer_ctc_run1/best_conformer_ctc.pt"
LM_DIR       = BASE / "models/language_models"
VOCAB_FILE   = BASE / "data/konkani-10k/vocab.json"
TEST_MANIFEST= BASE / "data/konkani-10k/test_manifest.json"

class CharTokenizer:
    def __init__(self):
        v = json.load(open(VOCAB_FILE, encoding='utf-8'))
        self.idx2char = {int(k): c for k, c in v['idx2char'].items()}
        self.vocab_size = v['vocab_size']
        self.blank_id = 0

    def labels(self):
        L = []
        for i in range(self.vocab_size):
            p = self.idx2char.get(i, f"<id{i}>")
            if p in ["<pad>", "<blank>", "<sos>", "<eos>", "<unk>"]:
                if i == self.blank_id: L.append("")
                else: L.append(f"<{p[1:-1]}_{i}>")
            elif p == "": L.append(f"<empty_{i}>")
            else: L.append(p)
        return L

def process_audio(path, device):
    try:
        import librosa
        import torchaudio.transforms as T
        audio, s = librosa.load(path, sr=16000)
        wav = torch.FloatTensor(audio)
        if wav.dim() == 1: wav = wav.unsqueeze(0)
        if wav.size(0) > 1: wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0).to(device)
        mel_fn = T.MelSpectrogram(16000, n_mels=80, n_fft=400, hop_length=160, win_length=400).to(device)
        mel = mel_fn(wav.unsqueeze(0))
        mel = mel.transpose(1, 2)
        mel = torch.log(mel + 1e-9)
        mel_len = torch.tensor([(wav.size(0) // 160) + 1], device=device)
        return mel.float(), mel_len
    except: return None, None

def edit_dist(a, b):
    m, n = len(a), len(b)
    dp = list(range(n+1))
    for i in range(1, m+1):
        prev, dp[0] = dp[0], i
        for j in range(1, n+1):
            t = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = t
    return dp[n]

def wer_cer(ref, hyp):
    c_dist = edit_dist(list(ref), list(hyp))
    c_len = max(len(ref), 1)
    r_words, h_words = ref.split(), hyp.split()
    w_dist = edit_dist(r_words, h_words)
    w_len = max(len(r_words), 1)
    return w_dist / w_len, c_dist / c_len

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Grid Search on {device}")
    
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    v_size = state['ctc_head.weight'].shape[0]
    model = ConformerCTC(vocab_size=v_size, input_dim=80, d_model=256, num_layers=12)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    
    tok = CharTokenizer()
    labels = tok.labels()
    samples = [json.loads(l) for l in open(TEST_MANIFEST, encoding='utf-8')]
    
    # Optional subset for speed (we'll use 400 for grid search, which is very representative)
    samples = samples[:400]
    
    print(f"Saving logits for {len(samples)} samples...")
    logits_cache = []
    
    for s in tqdm(samples):
        ref = s['text'].strip()
        mel, mel_len = process_audio(s['audio_filepath'], device)
        if mel is None: continue
        with torch.no_grad():
            logits, _ = model(mel, mel_len)
        lp = F.log_softmax(logits.float(), dim=-1).squeeze(0).cpu().numpy()
        logits_cache.append({"ref": ref, "lp": lp})
        
    print("Done computing logits! Starting LM Grid Search...")
    import pyctcdecode
    lm_path = str(LM_DIR / "konkani_3gram.binary")
    
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    betas =  [0.0, 0.5, 1.0, 1.5, 2.0]
    
    results = []
    import warnings
    warnings.filterwarnings("ignore")

    for a in alphas:
        for b in betas:
            decoder = pyctcdecode.build_ctcdecoder(labels, kenlm_model_path=lm_path, alpha=a, beta=b)
            
            total_wer, total_cer = 0.0, 0.0
            for item in logits_cache:
                hyp = decoder.decode(item["lp"], beam_width=20)
                w, c = wer_cer(item["ref"], hyp)
                total_wer += w
                total_cer += c
            
            avg_w = (total_wer / len(logits_cache)) * 100
            avg_c = (total_cer / len(logits_cache)) * 100
            results.append((a, b, avg_w, avg_c))
            print(f"Alpha: {a:.1f}, Beta: {b:.1f} -> WER: {avg_w:.2f}%, CER: {avg_c:.2f}%")

    best = min(results, key=lambda x: x[2])
    print("\n================== BEST HYPERPARAMETERS ==================")
    print(f"BEST_ALPHA={best[0]:.1f}")
    print(f"BEST_BETA={best[1]:.1f}")
    print(f"BEST_WER={best[2]:.2f}")
    print(f"BEST_CER={best[3]:.2f}")

if __name__ == "__main__":
    main()
