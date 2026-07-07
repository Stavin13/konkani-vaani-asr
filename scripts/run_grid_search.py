#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
import librosa
from tqdm import tqdm
import multiprocessing

# Use relative paths or handle the & in the absolute path correctly
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from models.conformer_ctc import ConformerCTC

# Paths
CHECKPOINT    = BASE / "outputs/conformer_ctc_run1/best_conformer_ctc.pt"
LM_DIR        = BASE / "models/language_models"
VOCAB_FILE    = BASE / "data/konkani-10k/vocab.json"
TEST_MANIFEST = BASE / "data/konkani-combined/val.json"

class CharTokenizer:
    def __init__(self):
        v = json.load(open(VOCAB_FILE, encoding='utf-8'))
        self.idx2char = {int(k): c for k, c in v['idx2char'].items()}
        self.vocab_size = v['vocab_size']
        # CRITICAL FIX: Training script used blank=0. 
        # vocab.json has <pad> at 0 and <blank> at 1, but the loss function 
        # and greedy_decode in training script both used index 0 as the blank.
        self.blank_id = 0 

    def labels(self):
        L = []
        for i in range(self.vocab_size):
            p = self.idx2char.get(i, "")
            if i == self.blank_id:
                L.append("") # pyctcdecode blank
            elif len(p) == 1:
                L.append(p)
            else:
                # Use unique non-printing chars for special tokens to satisfy pyctcdecode
                # Using high unicode range to avoid collisions
                L.append(chr(0xE000 + i)) 
        return L

def process_audio(path, device):
    try:
        import torchaudio.transforms as T
        import librosa
        if not os.path.exists(path):
            return None, None
            
        audio, s = librosa.load(path, sr=16000)
        if len(audio) < 400: # Min length for n_fft
            return None, None
            
        wav = torch.FloatTensor(audio).unsqueeze(0)
        # We do mel on CPU for reliability during this audit
        mel_fn = T.MelSpectrogram(16000, n_mels=80, n_fft=400, hop_length=160, win_length=400)
        mel = mel_fn(wav)
        mel = mel.transpose(1, 2)
        mel = torch.log(mel + 1e-9)
        mel_len = torch.tensor([mel.size(1)], device=device)
        return mel.to(device).float(), mel_len
    except Exception as e:
        # print(f"\nError processing {path}: {e}")
        return None, None

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
    print(f"LM Grid Search Audit on {device}")
    
    if not CHECKPOINT.exists():
        print(f"Error: Checkpoint {CHECKPOINT} not found!")
        return

    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    v_size = state['ctc_head.weight'].shape[0]
    model = ConformerCTC(vocab_size=v_size, input_dim=80, d_model=256, num_layers=12)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    
    tok = CharTokenizer()
    labels = tok.labels()
    
    if not TEST_MANIFEST.exists():
        print(f"Error: Manifest {TEST_MANIFEST} not found!")
        return
        
    samples = [json.loads(l) for l in open(TEST_MANIFEST, encoding='utf-8')]
    samples = samples[:250] # Subset for faster grid search
    
    print(f"Extracting logits for {len(samples)} samples...")
    logits_cache = []
    
    for s in tqdm(samples):
        ref = s['text'].strip()
        mel, mel_len = process_audio(s['audio_filepath'], device)
        if mel is None: continue
        with torch.no_grad():
            logits, _ = model(mel, mel_len)
        lp = F.log_softmax(logits.float(), dim=-1).squeeze(0).cpu().numpy()
        logits_cache.append({"ref": ref, "lp": lp})
        
    print(f"\nComputing Baseline (Greedy)...")
    total_wer_g, total_cer_g = 0.0, 0.0
    for item in logits_cache:
        # Simple greedy decode
        tokens = item["lp"].argmax(axis=-1)
        decoded = []
        prev = None
        for t in tokens:
            if t != tok.blank_id and t != prev:
                char = labels[t]
                if char != "":
                    decoded.append(char)
            prev = t
        hyp = "".join(decoded)
        w, c = wer_cer(item["ref"], hyp)
        total_wer_g += w
        total_cer_g += c
        
    base_w = (total_wer_g / len(logits_cache)) * 100
    base_c = (total_cer_g / len(logits_cache)) * 100
    print(f"GREEDY BASELINE -> WER: {base_w:.2f}%, CER: {base_c:.2f}%")

    import pyctcdecode
    import warnings
    warnings.filterwarnings("ignore")

    lm_files = {
        "3-gram": str(LM_DIR / "konkani_3gram.binary"),
        "4-gram": str(LM_DIR / "konkani_4gram.binary")
    }

    alphas = [0.1, 0.5, 1.0, 1.5]
    betas =  [0.0, 1.0, 2.0]
    
    for lm_name, lm_path in lm_files.items():
        if not Path(lm_path).exists():
            # Try .arpa if .binary doesn't exist
            lm_path = lm_path.replace(".binary", ".arpa")
            if not Path(lm_path).exists():
                print(f"\nSkipping {lm_name} (not found at {lm_path})")
                continue
        
        print(f"\n--- Grid Search for {lm_name} LM ({lm_path}) ---")
        best_lm_wer = float('inf')
        best_params = (0, 0)

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
                
                improvement = base_w - avg_w
                mark = "✅ IMPROVED" if improvement > 0 else "❌ WORSE"
                print(f"Alpha: {a:<4} Beta: {b:<4} -> WER: {avg_w:>6.2f}% ({improvement:>+6.2f}) {mark}")
                
                if avg_w < best_lm_wer:
                    best_lm_wer = avg_w
                    best_params = (a, b)

        print(f"Best for {lm_name}: Alpha={best_params[0]}, Beta={best_params[1]}, WER={best_lm_wer:.2f}%")

if __name__ == "__main__":
    main()
