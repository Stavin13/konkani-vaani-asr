import torch
import traceback
from jiwer import wer, cer
from tqdm import tqdm
from train_conformer_v2 import Tokenizer, KonkaniDataset, collate_fn, compute_mel, build_mel_transform, CONFIG
from models.conformer_ctc_v2 import create_model_v2
from torch.utils.data import DataLoader

def test_model():
    device = torch.device('cpu')
    print(f"Running on: {device} (CPU to avoid conflict with training GPU process)\n")

    print("Loading tokenizer...")
    tokenizer = Tokenizer(CONFIG['bpe_vocab'], CONFIG['bpe_model'], CONFIG['char_vocab'])

    print("Creating model...")
    model = create_model_v2(
        vocab_size=tokenizer.vocab_size, d_model=CONFIG['d_model'],
        num_layers=CONFIG['num_layers'], freq_mask_param=CONFIG['freq_mask'],
        time_mask_param=CONFIG['time_mask']
    )

    ckpt_path = 'outputs/conformer_v2_200ep/best_model.pt'
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ep = ckpt['epoch'] + 1
    print(f"  Saved at Epoch   : {ep}")
    print(f"  Val Loss         : {ckpt['val_loss']:.4f}")
    print(f"  WER (train-time) : {ckpt['wer']:.2%}")
    print(f"  CER (train-time) : {ckpt['cer']:.2%}")
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    mel_transform = build_mel_transform(device)

    val_ds = KonkaniDataset(CONFIG['val_manifest'], tokenizer, CONFIG['max_audio_len'], augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collate_fn, shuffle=False, num_workers=0)

    all_preds, all_targets = [], []

    print(f"\nRunning greedy decode on full val set ({len(val_ds)} samples)...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            audio      = batch['audio'].to(device)
            audio_lens = batch['audio_lens'].to(device)
            t_strs     = batch['text_strs']

            mel, mel_lens = compute_mel(audio, audio_lens, mel_transform)
            preds = model.greedy_decode(mel, mel_lens, tokenizer.id2piece)

            all_preds.extend(preds)
            all_targets.extend(t_strs)

    avg_wer = wer(all_targets, all_preds)
    avg_cer = cer(all_targets, all_preds)

    print(f"\n{'='*50}")
    print(f"  FULL VAL SET RESULTS (Epoch {ep})")
    print(f"{'='*50}")
    print(f"  WER  : {avg_wer:.2%}")
    print(f"  CER  : {avg_cer:.2%}")
    print(f"{'='*50}\n")

    # Show 10 random samples
    import random
    print("--- Sample Predictions ---\n")
    indices = random.sample(range(len(all_targets)), min(10, len(all_targets)))
    for i in indices:
        print(f"  TARGET : {all_targets[i]}")
        print(f"  PRED   : {all_preds[i] if all_preds[i].strip() else '[empty]'}")
        print()


if __name__ == '__main__':
    try:
        test_model()
    except Exception:
        traceback.print_exc()
