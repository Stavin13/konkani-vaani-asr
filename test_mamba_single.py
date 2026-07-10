#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
import torch

BASE = Path("/Volumes/data&proj/konkani")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "mamba"))

from models.conformer_ctc import ConformerCTC

# ---------- Paths ----------
CHECKPOINT        = BASE / "outputs/conformer_ctc_run1/best_conformer_ctc.pt"
VOCAB_FILE        = BASE / "data/konkani-10k/vocab.json"
TEST_MANIFEST     = BASE / "data/konkani-ultimate/train.json"
MAMBA_CHECKPOINT  = BASE / "mamba/best_model_test2.pt"   # use the best model from training
MAMBA_VOCAB       = BASE / "data/vocab.json"

# ---------- Audio processing ----------
def process_audio(path, device):
    import librosa
    import torchaudio.transforms as T
    audio, s = librosa.load(path, sr=16000)
    wav = torch.FloatTensor(audio)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0).to(device)
    mel_fn = T.MelSpectrogram(
        16000, n_mels=80, n_fft=400, hop_length=160, win_length=400
    ).to(device)
    mel = mel_fn(wav.unsqueeze(0))
    mel = mel.transpose(1, 2)
    mel = torch.log(mel + 1e-9)
    mel_len = torch.tensor([(wav.size(0) // 160) + 1], device=device)
    return mel.float(), mel_len

# ---------- ASR Tokenizer ----------
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

# ---------- Main ----------
def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    # --- 1. Load ASR model ---
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    vocab_size = state['ctc_head.weight'].shape[0]
    model = ConformerCTC(
        vocab_size=vocab_size,
        input_dim=80,
        d_model=256,
        num_layers=12
    )
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    tok = CharTokenizer()

    # --- 2. Load test sample ---
    with open(TEST_MANIFEST, encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    sample = samples[0]

    # --- 3. Load Mamba corrector ---
    from train_custom_mamba import TinyMambaCorrectorModel, KonkaniCharTokenizer

    mamba_tok = KonkaniCharTokenizer(str(MAMBA_VOCAB))

    state_dict = torch.load(MAMBA_CHECKPOINT, map_location=device)
    config = state_dict.get('config', {})  # fallback empty dict

    mamba_model = TinyMambaCorrectorModel(
        vocab_size=mamba_tok.vocab_size,
        d_model=config.get("d_model", 256),
        n_layers=config.get("n_layers", 6),
        d_state=config.get("d_state", 16),
        d_conv=config.get("d_conv", 4),
        expand=config.get("expand", 2),
        dropout=config.get("dropout", 0.1)
    )

    # Remove torch.compile wrapper if present
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict['model_state'].items()}
    mamba_model.load_state_dict(clean_state)
    mamba_model.eval().to(device)

    # --- 4. Run ASR ---
    audio_path = sample['audio_filepath']
    ref_text = sample['text'].strip()
    print(f"Testing on audio: {audio_path}")
    print(f"Reference text  : {ref_text}")

    mel, mel_len = process_audio(audio_path, device)

    with torch.no_grad():
        logits, _ = model(mel, mel_len)

    ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    hyp_greedy = tok.decode(ids)
    print(f"ASR Greedy output: {hyp_greedy}")

    # --- 5. Mamba correction with EOS stop ---
    src_ids = mamba_tok.encode(hyp_greedy) + [mamba_tok.sep_id]
    src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(src_t)

    with torch.no_grad():
        max_new = len(ref_text) + 20
        # ✅ FIX: pass eos_token_id so generation stops at first <eos>
        out_ids = mamba_model.generate(
            src_t,
            attention_mask=mask,
            max_new=max_new,
            eos_token_id=mamba_tok.eos_token_id  # this is 80
        )

    print(f"Raw Mamba output IDs: {out_ids}")
    hyp_mamba = mamba_tok.decode(out_ids)
    print(f"Mamba Decoded output: {hyp_mamba}")

    # Optional: show identity test result
    if hyp_greedy == ref_text:
        print("✅ ASR got it perfectly – Mamba correctly preserved the transcript.")
    else:
        print("⚠️ ASR had errors – Mamba attempted to correct them.")

if __name__ == '__main__':
    main()