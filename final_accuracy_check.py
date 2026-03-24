import torch
import json, os, sys, codecs
from jiwer import wer, cer
from tqdm import tqdm
from torch.utils.data import DataLoader
from train_conformer_v2 import Tokenizer, KonkaniDataset, collate_fn, compute_mel, build_mel_transform
from models.conformer_ctc_v2 import create_model_v2

# Force UTF-8 for Devanagari output
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

MODELS = {
    'BASELINE (200ep)': 'outputs/conformer_v2_200ep/best_model.pt',
    'FINE-TUNED (10k)': 'outputs/conformer_v2_finetune_10k/best_model_ft.pt'
}

def load_model(path, device, vocab_size):
    model = create_model_v2(vocab_size=vocab_size).to(device)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    return model.eval()

class DeepTestDS(KonkaniDataset):
    def load_samples(self, manifest_path):
        samples = []
        old_p = "/Volumes/data&proj/konkani/KonkaniRawSpeechCorpus/"
        new_p = "E:/konkani/KonkaniRawSpeechCorpus/"
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                p = item['audio_filepath'].replace(old_p, new_p).replace("\\", "/")
                if os.path.exists(p): samples.append({'path': p, 'text': item['text']})
        return samples # Full 896 samples

def run_score():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer('data/bpe_tokenizer/bpe_vocab.json', 'data/bpe_tokenizer/konkani_bpe.model', 'data/konkani-10k/vocab.json')
    mel_t = build_mel_transform(device)
    
    ds = DeepTestDS('data/konkani-10k/val_manifest.json', tokenizer, 16000*15, augment=False)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    
    final_scores = {}
    
    for name, path in MODELS.items():
        print(f"\nEvaluating {name}...")
        model = load_model(path, device, tokenizer.vocab_size)
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Scoring {name}"):
                mel, mel_l = compute_mel(batch['audio'].to(device), batch['audio_lens'].to(device), mel_t)
                logits, out_lens = model(mel, mel_l)
                p_ids = torch.argmax(logits, dim=-1)
                for i in range(p_ids.size(0)):
                    pred = tokenizer.decode_ctc(p_ids[i, :out_lens[i]].tolist())
                    all_preds.append(pred)
                    all_targets.append(batch['text_strs'][i])
                    
        final_scores[name] = {
            'wer': wer(all_targets, all_preds),
            'cer': cer(all_targets, all_preds)
        }

    print("\n" + "="*50)
    print(f"{'MODEL':<20} | {'WER':<10} | {'CER':<10}")
    print("-" * 50)
    for model_name, metrics in final_scores.items():
        print(f"{model_name:<20} | {metrics['wer']:>9.2%} | {metrics['cer']:>9.2%}")
    print("="*50 + "\n")

if __name__ == '__main__':
    run_score()
