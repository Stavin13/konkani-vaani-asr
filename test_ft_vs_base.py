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
from train_conformer_v2 import Tokenizer, KonkaniDataset, collate_fn, compute_mel, build_mel_transform
from models.conformer_ctc_v2 import create_model_v2

MODELS = {
    'BASELINE (200ep)': 'outputs/conformer_v2_200ep/best_model.pt',
    'FINE-TUNED (10k)': 'outputs/conformer_v2_finetune_10k/best_model_ft.pt'
}

CONFIG = {
    'val_manifest': 'data/konkani-10k/val_manifest.json',
    'bpe_vocab': 'data/bpe_tokenizer/bpe_vocab.json',
    'bpe_sp_model': 'data/bpe_tokenizer/konkani_bpe.model',
    'char_vocab': 'data/konkani-10k/vocab.json',
}

def load_model(path, device, vocab_size):
    model = create_model_v2(vocab_size=vocab_size)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    # Check if nested in model_state_dict
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

# Custom DS with path correction
class TestDS(KonkaniDataset):
    def load_samples(self, manifest_path):
        samples = []
        old_p = "/Volumes/data&proj/konkani/KonkaniRawSpeechCorpus/"
        new_p = "E:/konkani/KonkaniRawSpeechCorpus/"
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                path = item['audio_filepath'].replace(old_p, new_p).replace("\\", "/")
                if os.path.exists(path):
                    samples.append({'path': path, 'text': item['text']})
        return samples[:50] # Just 50 samples for a quick look

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer(CONFIG['bpe_vocab'], CONFIG['bpe_sp_model'], CONFIG['char_vocab'])
    mel_transform = build_mel_transform(device)
    
    ds = TestDS(CONFIG['val_manifest'], tokenizer, 16000*15, augment=False)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)
    
    results = {}
    for name, path in MODELS.items():
        print(f"\n--- Testing {name} ---")
        model = load_model(path, device, tokenizer.vocab_size)
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(loader):
                mel, mel_lens = compute_mel(batch['audio'].to(device), batch['audio_lens'].to(device), mel_transform)
                logits, out_lens = model(mel, mel_lens)
                p_ids = torch.argmax(logits, dim=-1)
                pred = tokenizer.decode_ctc(p_ids[0, :out_lens[0]].tolist())
                all_preds.append(pred)
                all_targets.append(batch['text_strs'][0])
                
        results[name] = {
            'wer': wer(all_targets, all_preds),
            'cer': cer(all_targets, all_preds),
            'preds': all_preds,
            'targets': all_targets
        }

    print(f"\n{'='*60}")
    print(f"{'MODEL':<20} | {'WER':<10} | {'CER':<10}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<20} | {res['wer']:>9.2%} | {res['cer']:>9.2%}")
    print(f"{'='*60}\n")
    
    print("--- TRANSCRIPTION SAMPLES ---")
    targets = results['BASELINE (200ep)']['targets']
    base_preds = results['BASELINE (200ep)']['preds']
    ft_preds   = results['FINE-TUNED (10k)']['preds']
    
    for i in range(min(8, len(targets))):
        print(f"TGT: {targets[i]}")
        print(f"BSE: {base_preds[i]}")
        print(f"FT : {ft_preds[i]}")
        print("-" * 30)

if __name__ == '__main__':
    main()
