import torch
import torchaudio
import torch.nn.functional as F
import librosa
import numpy as np
import json
import os
from pathlib import Path
from pyctcdecode import build_ctcdecoder
import sentencepiece as spm
import sys
sys.path.insert(0, ".")
from models.conformer_ctc_v2 import create_model_v2

# CONFIG
MODEL_PATH = r"outputs/conformer_v2_200ep/best_model.pt"
VOCAB_PATH = r"data/bpe_tokenizer/bpe_vocab.json"
SPM_PATH = r"data/bpe_tokenizer/konkani_bpe.model"
LM_PATH = r"models/language_models/konkani_3gram.arpa"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_all():
    print(f"Loading checkpoint from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    
    # Vocab and tokenizer
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    sp = spm.SentencePieceProcessor(model_file=SPM_PATH)
    
    # Model architecture (assuming defaults from v2)
    model = create_model_v2(vocab_size=vocab['vocab_size'], d_model=256, num_layers=12)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(DEVICE)
    model.eval()
    
    # LM Decoder
    print("Building CTC Decoder with LM...")
    # Map labels: index -> BPE piece
    labels = [""] * vocab['vocab_size']
    # piece2id
    for piece, id in vocab['piece2id'].items():
        if id < vocab['vocab_size']:
            # For BPE, we use the raw piece. 
            # In pyctcdecode, whitespace is denoted as a space character.
            # Our pieces already have '▁' for word starts (SentencePiece default)
            labels[id] = piece.replace('▁', ' ') 

    # Key for decoding: Blank must be "" at index 4 (or 0)
    # The config said blank_id is 0 but vocab lists <pad> at 0.
    # Let me check the vocab once more.
    labels[0] = "" # <pad> and blank must be empty string
    
    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=LM_PATH, # Point to the .arpa or .binary
        alpha=0.5, # LM weight
        beta=1.5   # Word insertion bonus
    )
    
    return model, sp, decoder, vocab

def preprocess_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)
    # Computes mel features directly (mimicking training preprocessing)
    audio_t = torch.FloatTensor(audio).to(DEVICE)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=80, n_fft=400, hop_length=160
    ).to(DEVICE)
    mel = mel_transform(audio_t.unsqueeze(0)).transpose(1, 2)
    mel = torch.log(mel + 1e-9)
    # Norm (assuming Mean-Var normalization was used)
    mean = mel.mean(dim=1, keepdim=True)
    std = mel.std(dim=1, keepdim=True) + 1e-9
    mel = (mel - mean) / std
    return mel

def test():
    model, sp, decoder, vocab = load_all()
    
    # Random audio sample from the corpus
    sample_files = [
        r"KonkaniRawSpeechCorpus/Data/Command and Control Words-W1/Female/16To20/LDC-IL_Scheduled_Konkani_Female_16To20_Command and Control Words-W1_SP-0009_W1-0017.wav",
        r"KonkaniRawSpeechCorpus/Data/Command and Control Words-W1/Female/16To20/LDC-IL_Scheduled_Konkani_Female_16To20_Command and Control Words-W1_SP-0009_W1-0107.wav"
    ]
    
    print("\n" + "="*80)
    print(f"{'AUDIO FILE':<20} | {'GREEDY (RAW)':<25} | {'BEAM + LM':<25}")
    print("="*80)
    
    for f in sample_files:
        if not os.path.exists(f): continue
        
        mel = preprocess_audio(f)
        with torch.no_grad():
            logits, _ = model(mel)
            log_probs = F.log_softmax(logits.float(), dim=-1).squeeze(0).cpu().numpy()
            
            # Greedy
            greedy_ids = np.argmax(log_probs, axis=-1)
            # CTC collapsible logic
            distinct_ids = []
            prev = -1
            for i in greedy_ids:
                if i != prev and i != 0: # skip blank/pad
                    distinct_ids.append(int(i))
                prev = i
            greedy_text = sp.decode(distinct_ids)
            
            # Beam + LM
            beam_text = decoder.decode(log_probs, beam_width=20)
            
            print(f"{os.path.basename(f)[:20]:<20} | {greedy_text:<25} | {beam_text:<25}")

if __name__ == "__main__":
    test()
