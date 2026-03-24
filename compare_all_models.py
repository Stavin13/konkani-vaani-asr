import torch
import torch.nn as nn
import json, os, sys, codecs
from jiwer import wer, cer
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path

# Fix console encoding on Windows
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Import from training scripts
from train_conformer_v2 import Tokenizer as BPETokenizer, KonkaniDataset, collate_fn, compute_mel, build_mel_transform
from train_conformer_v2_char import CharTokenizer
from models.conformer_ctc_v2 import create_model_v2 as create_model_bpe
from models.conformer_ctc import create_model as create_model_char # OLD VERSION

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CONFIG = {
    'val_manifest': 'data/konkani-20gb/val.json',
    'bpe_model': {
        'path': 'outputs/conformer_v2_200ep/best_model.pt',
        'bpe_vocab': 'data/bpe_tokenizer/bpe_vocab.json',
        'bpe_sp_model': 'data/bpe_tokenizer/konkani_bpe.model',
        'char_vocab': 'data/konkani-10k/vocab.json',
        'd_model': 256,
        'num_layers': 12,
    },
    'char_model': {
        'path': 'outputs/conformer_ctc_run1/best_conformer_ctc.pt',
        'char_vocab': 'data/konkani-10k/vocab.json',
        'd_model': 256,
        'num_layers': 12,
    },
    'batch_size': 4,
    'max_samples': 200, 
    'beam_width': 10,
}

def _build_beam_decoder(tokenizer):
    try:
        from pyctcdecode import build_ctc_decoder
        import numpy as np
        
        # Ensure we have a valid list of labels matchingvocab size
        labels = [""] * tokenizer.vocab_size
        for i, char in tokenizer.idx2char.items():
            if i < len(labels):
                labels[i] = char
        
        # CTC decoder expects labels[blank_id] to be "" but pyctcdecode handles it via index
        return build_ctc_decoder(labels, kenlm_model_path=None)
    except Exception as e:
        print(f"Warning: Beam Search builder failed ({e}). Falling back to Greedy.")
        return None

def evaluate(model, tokenizer, loader, mel_transform, device, is_bpe=True):
    all_preds, all_targets = [], []
    decoder = _build_beam_decoder(tokenizer)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Beam Search ({'BPE' if is_bpe else 'CHAR'})"):
            audio      = batch['audio'].to(device)
            audio_lens = batch['audio_lens'].to(device)
            t_strs     = batch['text_strs']
            
            mel, mel_lens = compute_mel(audio, audio_lens, mel_transform)
            
            # Forward pass
            # Note: v2 model returns (logits, out_lens)
            # Old model returns (logits, out_lens) where out_lens is same as input lens
            logits, out_lens = model(mel, mel_lens)
            
            if decoder:
                import torch.nn.functional as F
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                for i in range(probs.shape[0]):
                    beam_res = decoder.decode(probs[i, :out_lens[i]], beam_width=CONFIG['beam_width'])
                    all_preds.append(beam_res)
            else:
                preds_ids = torch.argmax(logits, dim=-1)
                for i in range(preds_ids.size(0)):
                    ids = preds_ids[i, :out_lens[i]].tolist()
                    all_preds.append(tokenizer.decode_ctc(ids))
            
            all_targets.extend(t_strs)
            
    return all_preds, all_targets

def load_bpe_model(device):
    print(f"\n[BPE] Loading tokenizer...")
    tokenizer = BPETokenizer(
        CONFIG['bpe_model']['bpe_vocab'], 
        CONFIG['bpe_model']['bpe_sp_model'], 
        CONFIG['bpe_model']['char_vocab']
    )
    print(f"[BPE] Creating Model v2...")
    model = create_model_bpe(vocab_size=tokenizer.vocab_size, d_model=CONFIG['bpe_model']['d_model'], num_layers=CONFIG['bpe_model']['num_layers'])
    ckpt = torch.load(CONFIG['bpe_model']['path'], map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model, tokenizer

def load_char_model(device):
    print(f"\n[Char] Loading tokenizer...")
    tokenizer = CharTokenizer(CONFIG['char_model']['char_vocab'])
    print(f"[Char] Creating Old Model (v1)...")
    # Old model vocab might be different, let's trust tokenizer.vocab_size
    model = create_model_char(vocab_size=tokenizer.vocab_size, d_model=CONFIG['char_model']['d_model'], num_layers=CONFIG['char_model']['num_layers'])
    ckpt = torch.load(CONFIG['char_model']['path'], map_location='cpu', weights_only=False)
    # Use strict=False to ignore num_batches_tracked mismatches
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device).eval()
    return model, tokenizer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mel_transform = build_mel_transform(device)
    
    # Load Models
    bpe_model, bpe_tokenizer   = load_bpe_model(device)
    old_char_model, char_tokenizer = load_char_model(device)
    
    # Dataset
    ds = KonkaniDataset(CONFIG['val_manifest'], bpe_tokenizer, 16000*15, augment=False)
    if CONFIG['max_samples']: ds.samples = ds.samples[:CONFIG['max_samples']]
    loader = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    print("\n--- Starting Evaluation (Beam Width: 10) ---")
    bpe_preds, targets = evaluate(bpe_model, bpe_tokenizer, loader, mel_transform, device, is_bpe=True)
    char_preds, _     = evaluate(old_char_model, char_tokenizer, loader, mel_transform, device, is_bpe=False)
    
    # --- Results ---
    bpe_wer, bpe_cer   = wer(targets, bpe_preds), cer(targets, bpe_preds)
    char_wer, char_cer = wer(targets, char_preds), cer(targets, char_preds)
    
    print(f"\n{'='*75}")
    print(f"      BEAM SEARCH COMPARISON (Width: {CONFIG['beam_width']})")
    print(f"{'='*75}")
    print(f"  MODEL           |    WER     |    CER     |  STAGES                 ")
    print(f"------------------|------------|------------|-------------------------")
    print(f"  BPE v2 (Current)|    {bpe_wer:>6.2%}  |    {bpe_cer:>6.2%}  |  200 Epochs  ")
    print(f"  Char v1 (Old)   |    {char_wer:>6.2%}  |    {char_cer:>6.2%}  |  Epoch 31 (Pilot) ")
    print(f"{'='*75}\n")
    
    print("--- SAMPLES ---")
    for i in range(min(10, len(targets))):
        print(f"TGT : {targets[i]}")
        print(f"BPE : {bpe_preds[i]}")
        print(f"CHR : {char_preds[i]}")
        print("-" * 30)

if __name__ == '__main__':
    main()
