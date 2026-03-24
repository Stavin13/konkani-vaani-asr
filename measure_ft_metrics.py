import torch
from jiwer import cer, wer
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from models.conformer_ctc_v2 import create_model_v2
from train_conformer_v2 import Tokenizer, KonkaniDataset, collate_fn, compute_mel, build_mel_transform

# Peeking into the latest metrics
ckpt_path = 'outputs/conformer_v2_finetune_10k/latest_checkpoint.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

def check_cer():
    # Because we don't save CER history in the checkpoint, we must run it on val set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer('data/bpe_tokenizer/bpe_vocab.json', 'data/bpe_tokenizer/konkani_bpe.model', 'data/konkani-10k/vocab.json')
    model = create_model_v2(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Path correction logic for a one-off run
    class LocalDS(KonkaniDataset):
        def load_samples(self, manifest_path):
            samples = []
            old_p = "/Volumes/data&proj/konkani/KonkaniRawSpeechCorpus/"
            new_p = "E:/konkani/KonkaniRawSpeechCorpus/"
            import json
            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    path = item['audio_filepath'].replace(old_p, new_p).replace("\\", "/")
                    if os.path.exists(path):
                        samples.append({'path': path, 'text': item['text']})
            return samples[:200] # Use only 200 samples for a quick measurement
            
    val_ds = LocalDS('data/konkani-10k/val_manifest.json', tokenizer, 16000*15, augment=False)
    loader = DataLoader(val_ds, batch_size=4, collate_fn=collate_fn)
    mel_transform = build_mel_transform(device)
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Measuring CER..."):
            mel, mel_lens = compute_mel(batch['audio'].to(device), batch['audio_lens'].to(device), mel_transform)
            logits, out_lens = model(mel, mel_lens)
            p_ids = torch.argmax(logits, dim=-1)
            for i in range(p_ids.size(0)):
                all_preds.append(tokenizer.decode_ctc(p_ids[i, :out_lens[i]].tolist()))
            all_targets.extend(batch['text_strs'])
            
    c_err = cer(all_targets, all_preds)
    w_err = wer(all_targets, all_preds)
    print(f"\n--- EPOCH {ckpt['epoch']} METRICS ---")
    print(f"WER: {w_err:.2%}")
    print(f"CER: {c_err:.2%}")

if __name__ == '__main__':
    check_cer()
