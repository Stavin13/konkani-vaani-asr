import torch
ckpt = torch.load('e:/konkani/outputs/conformer_v2_200ep/latest_checkpoint.pt', map_location='cpu', weights_only=False)
for name, param in ckpt['model_state_dict'].items():
    if torch.isnan(param).any():
        print(f"NaN found in {name}")
    if torch.isinf(param).any():
        print(f"Inf found in {name}")
print("Check complete.")
