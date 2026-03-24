import torch
ckpt = torch.load('outputs/conformer_v2_finetune_10k/latest_checkpoint.pt', map_location='cpu', weights_only=False)
print(f"Epoch: {ckpt['epoch']}")
if 'wer' in ckpt: print(f"Current WER: {ckpt['wer']:.2%}")
if 'loss' in ckpt: print(f"Loss: {ckpt['loss']:.4f}")
