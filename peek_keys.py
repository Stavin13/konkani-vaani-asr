import torch
ckpt = torch.load('outputs/conformer_ctc_run1/best_conformer_ctc.pt', map_location='cpu', weights_only=False)
keys = list(ckpt['model_state_dict'].keys())
print(f"Num keys: {len(keys)}")
print(f"Sample keys: {keys[:15]}")
