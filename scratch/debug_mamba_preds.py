import json, sys
from pathlib import Path
import torch

BASE = Path("/Volumes/data&proj/konkani")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "mamba"))

from run_char_eval import CharTokenizer, process_audio, wer_cer
from train_custom_mamba import TinyMambaCorrectorModel, KonkaniCharTokenizer

VOCAB_FILE = BASE / "data/konkani-10k/vocab.json"
TEST_MANIFEST = BASE / "data/konkani-10k/test_manifest.json"
MAMBA_CHECKPOINT = BASE / "mamba/best_model_mamba_test.pt"
MAMBA_VOCAB = BASE / "data/vocab.json"

device = "cpu"
tok = CharTokenizer()
mamba_tok = KonkaniCharTokenizer(str(MAMBA_VOCAB))

# Load Mamba
state_dict = torch.load(MAMBA_CHECKPOINT, map_location=device)
config = state_dict.get('config', {})
mamba_model = TinyMambaCorrectorModel(
    vocab_size=83,
    d_model=config.get("d_model", 256),
    n_layers=config.get("n_layers", 6),
    d_state=config.get("d_state", 16),
    d_conv=config.get("d_conv", 4),
    expand=config.get("expand", 2),
    dropout=config.get("dropout", 0.1)
)
clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict['model_state'].items()}
mamba_model.load_state_dict(clean_state_dict)
mamba_model.eval()

samples = [json.loads(l) for l in open(TEST_MANIFEST, encoding='utf-8')][:10]

for i, s in enumerate(samples):
    ref = s['text'].strip()
    
    # ASR model outputs are already saved in train_audit.csv or we can check.
    # But wait, let's load conformer model and get its greedy output
    # Actually, we can just run conformal model.
    # Let's import ConformerCTC
    from models.conformer_ctc import ConformerCTC
    CHECKPOINT = BASE / "outputs/conformer_ctc_run1/best_conformer_ctc.pt"
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    v_size = state['ctc_head.weight'].shape[0]
    model = ConformerCTC(vocab_size=v_size, input_dim=80, d_model=256, num_layers=12)
    model.load_state_dict(state, strict=False)
    model.eval()
    
    mel, mel_len = process_audio(s['audio_filepath'], device)
    if mel is None: continue
    with torch.no_grad():
        logits, _ = model(mel, mel_len)
    ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    hyp_greedy = tok.decode(ids)
    
    # Mamba
    src_ids = mamba_tok.encode(hyp_greedy) + [mamba_tok.sep_id]
    src_t   = torch.tensor([src_ids], dtype=torch.long, device=device)
    mask    = torch.ones_like(src_t)
    with torch.no_grad():
        out_ids = mamba_model.generate(src_t, attention_mask=mask, max_new=len(ref)+20)
    hyp_mamba = mamba_tok.decode(out_ids)
    
    print(f"\nSample {i+1}:")
    print(f"  ASR Greedy: '{hyp_greedy}'")
    print(f"  Mamba Corr: '{hyp_mamba}'")
    print(f"  Reference : '{ref}'")
