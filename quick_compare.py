import torch
import json, os, sys, codecs
from jiwer import wer, cer
from torch.utils.data import DataLoader
from train_conformer_v2 import Tokenizer, KonkaniDataset, collate_fn, compute_mel, build_mel_transform
from models.conformer_ctc_v2 import create_model_v2

# Force UTF-8
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

MODELS = {
    'BASELINE': 'outputs/conformer_v2_200ep/best_model.pt',
    'FINE-TUNED': 'outputs/conformer_v2_finetune_10k/best_model_ft.pt'
}

def load_model(path, vocab_size, device):
    model = create_model_v2(vocab_size=vocab_size).to(device)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    return model.eval()

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer('data/bpe_tokenizer/bpe_vocab.json', 'data/bpe_tokenizer/konkani_bpe.model', 'data/konkani-10k/vocab.json')
    mel_t = build_mel_transform(device)
    
    # Custom DS with path correction
    class ManualDS(KonkaniDataset):
        def load_samples(self, manifest_path):
            samples = []
            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    p = item['audio_filepath'].replace("/Volumes/data&proj/konkani/KonkaniRawSpeechCorpus/", "E:/konkani/KonkaniRawSpeechCorpus/").replace("\\", "/")
                    if os.path.exists(p): samples.append({'path': p, 'text': item['text']})
            return samples[:10]

    ds = ManualDS('data/konkani-10k/val_manifest.json', tokenizer, 16000*15, augment=False)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)
    
    base_model = load_model(MODELS['BASELINE'], tokenizer.vocab_size, device)
    ft_model   = load_model(MODELS['FINE-TUNED'], tokenizer.vocab_size, device)
    
    print("\n" + "="*80)
    print(f"{'TARGET TEXT':<40} | {'BASELINE':<20} | {'FINE-TUNED'}")
    print("-" * 80)
    
    with torch.no_grad():
        for batch in loader:
            audio, lens = batch['audio'].to(device), batch['audio_lens'].to(device)
            mel, mel_l = compute_mel(audio, lens, mel_t)
            
            # Base
            log_b, out_b = base_model(mel, mel_l)
            pred_b = tokenizer.decode_ctc(torch.argmax(log_b, -1)[0, :out_b[0]].tolist())
            
            # FT
            log_f, out_f = ft_model(mel, mel_l)
            pred_f = tokenizer.decode_ctc(torch.argmax(log_f, -1)[0, :out_f[0]].tolist())
            
            print(f"{batch['text_strs'][0][:40]:<40} | {pred_b[:20]:<20} | {pred_f}")

    print("="*80 + "\n")

if __name__ == '__main__':
    test()
