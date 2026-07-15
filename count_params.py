import torch
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE))
from models.conformer_ctc import ConformerCTC
from scripts.generate_predictions_analysis_excel import load_model, load_mamba_model

conformer_ckpt = BASE / "outputs/conformer_ctc_run1/best_conformer_ctc.pt"
mamba_ckpt = BASE / "mamba/best_model_test2.pt"
vocab_path = BASE / "data/konkani-10k/vocab.json"
device = torch.device('cpu')

print("Loading Conformer...")
conformer = load_model(conformer_ckpt, device)
conformer_params = sum(p.numel() for p in conformer.parameters())
conformer_trainable = sum(p.numel() for p in conformer.parameters() if p.requires_grad)

print("Loading Mamba...")
mamba, _, _ = load_mamba_model(mamba_ckpt, vocab_path, device)
mamba_params = sum(p.numel() for p in mamba.parameters())
mamba_trainable = sum(p.numel() for p in mamba.parameters() if p.requires_grad)

total_params = conformer_params + mamba_params
total_trainable = conformer_trainable + mamba_trainable

print(f"\n--- Parameter Counts ---")
print(f"Conformer Encoder Params: {conformer_params:,} ({conformer_params / 1e6:.2f}M)")
print(f"Mamba Decoder Params:     {mamba_params:,} ({mamba_params / 1e6:.2f}M)")
print(f"------------------------")
print(f"TOTAL PIPELINE PARAMS:    {total_params:,} ({total_params / 1e6:.2f}M)")
