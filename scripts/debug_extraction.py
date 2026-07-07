import json
import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from tqdm import tqdm
import numpy as np
from jiwer import process_words
import unicodedata
import re
import sys
import soundfile as sf

# Paths
BASE = Path("/Volumes/data&proj/konkani")
CHECKPOINT = BASE / "outputs/conformer_v2_200ep/best_model.pt"
VOCAB_FILE = BASE / "data/konkani-10k/vocab.json"
TEST_MANIFEST = BASE / "data/konkani-combined/manifests/test.json"
LM_3GRAM = BASE / "models/language_models/konkani_3gram.binary"
LM_4GRAM = BASE / "models/language_models/konkani_4gram.binary"
UNIGRAMS = BASE / "models/language_models/unigrams.txt"

# Model import
sys.path.insert(0, str(BASE))
from models.conformer_ctc_v2 import ConformerCTCv2

# Text normalization
MISSING_DIGITS = set("०३४५६७८")
def normalise_text(text: str, clean: bool = False) -> str:
    if text is None: return ""
    text = unicodedata.normalize("NFC", text)
    if clean:
        cleaned = []
        for ch in text:
            cat = unicodedata.category(ch)
            if (cat.startswith("L") or cat.startswith("M") or
                    cat.startswith("N") or ch.isspace()):
                if ch not in MISSING_DIGITS:
                    cleaned.append(ch)
        text = "".join(cleaned)
    return re.sub(r" +", " ", text).strip()

class CharTokenizer:
    def __init__(self):
        v = json.load(open(VOCAB_FILE, encoding="utf-8"))
        self.idx2char = {int(k): c for k, c in v["idx2char"].items()}
        self.vocab_size = v["vocab_size"]
        self.blank_id = 1 

    def labels(self):
        L = []
        for i in range(self.vocab_size):
            p = self.idx2char.get(i, f"<id{i}>")
            if p in ["<pad>", "<blank>", "<sos>", "<eos>", "<unk>"]:
                L.append("") if i == self.blank_id else L.append(f"<{p[1:-1]}_{i}>")
            else:
                L.append(p)
        return L

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available(): device = torch.device("cuda")
    print(f"Device: {device}")

    # Load Model
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    v_size = state["ctc_head.weight"].shape[0]
    d_model = state["ctc_head.weight"].shape[1]
    
    model = ConformerCTCv2(vocab_size=v_size, input_dim=80, d_model=d_model, num_layers=12)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    
    tok = CharTokenizer()
    labels = tok.labels()

    # Load decoders
    from pyctcdecode import build_ctcdecoder
    unigrams = None
    if UNIGRAMS.exists():
        with open(UNIGRAMS, encoding="utf-8") as f:
            unigrams = [l.strip() for l in f if l.strip()]

    dec_3gram = build_ctcdecoder(labels, kenlm_model_path=str(LM_3GRAM), unigrams=unigrams, alpha=0.5, beta=1.5)
    dec_4gram = build_ctcdecoder(labels, kenlm_model_path=str(LM_4GRAM), unigrams=unigrams, alpha=0.5, beta=1.5)

    samples = [json.loads(l) for l in open(TEST_MANIFEST, encoding="utf-8")]
    samples = samples[:10]

    # Pre-build Mel transform
    import torchaudio.transforms as T
    mel_fn = T.MelSpectrogram(
        sample_rate=16000, n_mels=80, n_fft=400,
        hop_length=160, win_length=400
    ).to(device)

    # Results holders
    stats = {
        "3gram": {"S": [], "D": [], "I": [], "errors": 0},
        "4gram": {"S": [], "D": [], "I": [], "errors": 0}
    }

    for s in tqdm(samples):
        try:
            wav, sr = sf.read(s["audio_filepath"])
            wav = torch.from_numpy(wav).float()
            if wav.ndim > 1: wav = wav.mean(-1)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
            wav = wav.to(device)
            
            with torch.no_grad():
                mel = mel_fn(wav.unsqueeze(0))
                mel = torch.log(mel + 1e-9)
                mel = (mel - (-10.0)) / (4.0 + 1e-5)
                m_len = torch.tensor([mel.size(2)], device=device)
                logits, _ = model(mel.transpose(1, 2), m_len)
            
            lp = F.log_softmax(logits, dim=-1)
            lp_merged = lp.clone()
            lp_merged[:, :, 1] = torch.logsumexp(lp[:, :, [0, 1]], dim=-1)
            lp_merged[:, :, 0] = -1e9
            lp_np = lp_merged.squeeze(0).cpu().float().numpy()

            ref = normalise_text(s["text"], clean=True)

            for name, dec in [("3gram", dec_3gram), ("4gram", dec_4gram)]:
                hyp = normalise_text(dec.decode(lp_np, beam_width=20), clean=True)
                print(f"[{name}] REF: {ref} | HYP: {hyp}")
                res = process_words(ref, hyp)
                stats[name]["S"].append(res.substitutions)
                stats[name]["D"].append(res.deletions)
                stats[name]["I"].append(res.insertions)
                if hyp != ref:
                    stats[name]["errors"] += 1
        except Exception as e:
            print(f"Error: {e}")
            continue

    print("\nSUMMARY:")
    for name in ["3gram", "4gram"]:
        st = stats[name]
        print(f"{name}: n={len(st['S'])}")

if __name__ == "__main__":
    main()
