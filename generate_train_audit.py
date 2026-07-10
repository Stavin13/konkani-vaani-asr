#!/usr/bin/env python3
"""
Generate Train Audit CSV
Runs the 88-hour training set (data/konkani-ultimate/train.json) through the Conformer CTC model
using Greedy Decoding and outputs to train_audit.csv with hyp_greedy and ref columns.
Now includes Mamba post‑correction using the best trained model.
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
CHECKPOINT      = BASE / "outputs/conformer_ctc_run1/best_conformer_ctc.pt"
VOCAB_FILE      = BASE / "data/konkani-10k/vocab.json"
TRAIN_MANIFEST  = BASE / "data/konkani-ultimate/train.json"
OUTPUT_CSV      = BASE / "train_audit.csv"

# ---------- Mamba paths (corrected) ----------
MAMBA_CHECKPOINT = BASE / "mamba/best_model_test2.pt"        # use the actual best model
MAMBA_VOCAB      = BASE / "data/vocab.json"            # same vocab as used in training

import torchaudio.transforms as T
mel_fn_cache = {}

def process_audio(path, device):
    try:
        import librosa
        audio, s = librosa.load(path, sr=16000)
        wav = torch.FloatTensor(audio)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0).to(device)
        
        if device not in mel_fn_cache:
            mel_fn_cache[device] = T.MelSpectrogram(
                16000, n_mels=80, n_fft=400, hop_length=160, win_length=400
            ).to(device)
        mel_fn = mel_fn_cache[device]
        
        mel = mel_fn(wav.unsqueeze(0))
        mel = mel.transpose(1, 2)
        mel = torch.log(mel + 1e-9)
        mel_len = torch.tensor([(wav.size(0) // 160) + 1], device=device)
        return mel.float(), mel_len
    except Exception as e:
        return None, None

# ── Tokenizer for ASR ─────────────────────────────────────────────────────────
class CharTokenizer:
    def __init__(self):
        with open(VOCAB_FILE, encoding='utf-8') as f:
            v = json.load(f)
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
    
    print(f"Loading ASR Model (Vocab Size: {v_size})...")
    model = ConformerCTC(vocab_size=v_size, input_dim=80, d_model=256, num_layers=12)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    
    tok = CharTokenizer()
    labels = []
    for i in range(tok.vocab_size):
        p = tok.idx2char.get(i, f"<id{i}>")
        if p in ["<pad>", "<blank>", "<sos>", "<eos>", "<unk>"]:
            if i == tok.blank_id:
                labels.append("")
            else:
                labels.append(f"<{p[1:-1]}_{i}>")
        elif p == "":
            labels.append(f"<empty_{i}>")
        else:
            labels.append(p)
        
    print("Loading pyctcdecode decoders...")
    try:
        from pyctcdecode import build_ctcdecoder
        dec_beam = build_ctcdecoder(labels)
        lm_path = BASE / "models/language_models/konkani_4gram.binary"
        dec_lm = build_ctcdecoder(labels, kenlm_model_path=str(lm_path)) if lm_path.exists() else None
    except Exception as e:
        print(f"Failed to load pyctcdecode: {e}")
        dec_beam = None
        dec_lm = None

    # ---------- Load Mamba Corrector (fixed path) ----------
    print("Loading Mamba Corrector...")
    mamba_model = None
    mamba_tok = None
    try:
        # Import from your training script (must be in the same directory)
        from train_mamba_corrector import TinyMambaCorrectorModel, KonkaniCharTokenizer
        
        mamba_tok = KonkaniCharTokenizer(str(MAMBA_VOCAB))
        if not MAMBA_CHECKPOINT.exists():
            print(f"Warning: Mamba checkpoint not found at {MAMBA_CHECKPOINT}. Skipping Mamba.")
        else:
            state_dict = torch.load(MAMBA_CHECKPOINT, map_location=device)
            config = state_dict.get('config', {})
            clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.get('model_state', state_dict).items()}
            
            # Infer vocab size from the checkpoint's embedding layer
            ckpt_vocab_size = clean_state['embedding.weight'].shape[0] if 'embedding.weight' in clean_state else mamba_tok.vocab_size
            if ckpt_vocab_size != mamba_tok.vocab_size:
                print(f"Warning: Mamba checkpoint vocab size ({ckpt_vocab_size}) differs from tokenizer ({mamba_tok.vocab_size}). Using checkpoint size.")
            
            mamba_model = TinyMambaCorrectorModel(
                vocab_size=ckpt_vocab_size,
                d_model=config.get("d_model", 256),
                n_layers=config.get("n_layers", 6),
                d_state=config.get("d_state", 16),
                d_conv=config.get("d_conv", 4),
                expand=config.get("expand", 2),
                dropout=config.get("dropout", 0.1)
            )
            mamba_model.load_state_dict(clean_state)
            mamba_model.eval().to(device)
            print("Mamba model loaded successfully.")
    except Exception as e:
        print(f"Failed to load Mamba: {e}")
        mamba_model = None
        mamba_tok = None

    # ---------- Process Manifest ----------
    print(f"Loading Manifest: {TRAIN_MANIFEST}")
    with open(TRAIN_MANIFEST, encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    print(f"Total Samples to Process: {len(samples)}")
    
    print(f"Writing to {OUTPUT_CSV}...")
    
    import torch.nn.functional as F
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["audio_filepath", "ref", "hyp_greedy", "hyp_beam", "hyp_lm", "hyp_mamba", "hyp_mamba_lm"])
        
        processed = 0
        failures = 0
        
        for s in tqdm(samples, desc="Processing Train Set"):
            path = s.get('audio_filepath', '')
            ref_text = s.get('text', '').strip()
            
            mel, mel_len = process_audio(path, device)
            if mel is None:
                failures += 1
                continue
                
            with torch.no_grad():
                logits, _ = model(mel, mel_len)
            
            # 1. Greedy
            ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
            hyp_greedy = tok.decode(ids)
            
            # 2. Beam & LM (if available)
            lp = F.log_softmax(logits.float(), dim=-1).squeeze(0).cpu().numpy()
            hyp_beam = dec_beam.decode(lp, beam_width=15) if dec_beam else ""
            hyp_lm = dec_lm.decode(lp, beam_width=15) if dec_lm else ""
            
            # 3. Mamba corrections (if model loaded)
            hyp_mamba = ""
            hyp_mamba_lm = ""
            if mamba_model is not None and mamba_tok is not None:
                # Mamba on greedy output
                if hyp_greedy.strip():
                    src_ids = mamba_tok.encode(hyp_greedy) + [mamba_tok.sep_id]
                    src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
                    mask = torch.ones_like(src_t)
                    with torch.no_grad():
                        out_ids = mamba_model.generate(
                            src_t,
                            attention_mask=mask,
                            max_new=len(ref_text) + 20,
                            eos_token_id=mamba_tok.eos_token_id
                        )
                    hyp_mamba = mamba_tok.decode(out_ids)
                
                # Mamba on beam+KenLM output (if available)
                if hyp_lm.strip():
                    src_ids_lm = mamba_tok.encode(hyp_lm) + [mamba_tok.sep_id]
                    src_t_lm = torch.tensor([src_ids_lm], dtype=torch.long, device=device)
                    mask_lm = torch.ones_like(src_t_lm)
                    with torch.no_grad():
                        out_ids_lm = mamba_model.generate(
                            src_t_lm,
                            attention_mask=mask_lm,
                            max_new=len(ref_text) + 20,
                            eos_token_id=mamba_tok.eos_token_id
                        )
                    hyp_mamba_lm = mamba_tok.decode(out_ids_lm)
            
            # Write row
            writer.writerow([path, ref_text, hyp_greedy, hyp_beam, hyp_lm, hyp_mamba, hyp_mamba_lm])
            processed += 1
            
            if processed % 1000 == 0:
                f.flush()
                
    print(f"\nCompleted! Processed {processed} files. Failed: {failures}.")
    print(f"Output saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()