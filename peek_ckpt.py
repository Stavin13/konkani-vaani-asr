import torch
ckpt_path = 'outputs/conformer_ctc_run1/best_conformer_ctc.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print(f"Checkpoint keys: {list(ckpt.keys())}")
if 'epoch' in ckpt: print(f"Epoch: {ckpt['epoch']}")
if 'val_loss' in ckpt: print(f"Val Loss: {ckpt['val_loss']}")
if 'wer' in ckpt: print(f"WER: {ckpt['wer']:.2%}")
if 'cer' in ckpt: print(f"CER: {ckpt['cer']:.2%}")
if 'vocab_size' in ckpt: print(f"Vocab Size: {ckpt['vocab_size']}")
if 'config' in ckpt: print(f"Config: {ckpt['config']}")
