#!/usr/bin/env python3
import argparse, json, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

sys.path.insert(0, '.')
from models.conformer_ctc_v2 import ConformerCTCv2

CHECKPOINT_DIR       = Path("E:/konkani/outputs/conformer_v2_finetune_10k")
LM_DIR               = Path("E:/konkani/models/language_models")
CHAR_VOCAB_PATH      = Path("data/konkani-10k/vocab.json")
BPE_VOCAB_PATH       = Path("data/bpe_tokenizer/bpe_vocab.json")
BPE_MODEL_PATH       = Path("data/bpe_tokenizer/konkani_bpe.model")
TEST_MANIFEST        = Path("data/konkani-10k/test_manifest.json")
LOCAL_AUDIO_FALLBACK = Path("data/audio/synthetic")
AUDIO_ROOT           = Path("E:/konkani")   # Mac /Volumes/data&proj/konkani/ → here

MAC_PREFIX = "/Volumes/data&proj/konkani/"

def resolve_audio_path(raw_path):
    if os.path.exists(raw_path):
        return raw_path
    if raw_path.startswith(MAC_PREFIX):
        rel = raw_path[len(MAC_PREFIX):]
        candidate = AUDIO_ROOT / rel
        if candidate.exists():
            return str(candidate)
    candidate = LOCAL_AUDIO_FALLBACK / os.path.basename(raw_path)
    return str(candidate) if candidate.exists() else None

def load_audio(path, sr=16000):
    try:
        wav, s = torchaudio.load(path)
        if s != sr: wav = torchaudio.transforms.Resample(s, sr)(wav)
        return wav.squeeze(0).float()
    except Exception as e:
        print(f"  [warn] {path}: {e}"); return None

def compute_mel(audio, device):
    fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=80, n_fft=512, hop_length=256, win_length=512).to(device)
    with torch.no_grad():
        mel = torch.log(fn(audio.to(device)) + 1e-8)
    return mel.transpose(1,2).unsqueeze(0)

class CharTokenizer:
    def __init__(self):
        v = json.load(open(CHAR_VOCAB_PATH, encoding='utf-8'))
        self.idx2char = v['idx2char']; self.char2idx = v['char2idx']
        self.vocab_size = v['vocab_size']; self.blank_id = self.char2idx.get('<blank>', 1)
    def decode_ids(self, ids):
        chars, prev = [], -1
        for i in ids:
            if i != prev and i not in (0, self.blank_id):
                chars.append(self.idx2char.get(str(i), ''))
            prev = i
        return ''.join(chars)
    def encode(self, text): return [self.char2idx.get(c, 4) for c in text]
    def labels(self):
        m = max(int(k) for k in self.idx2char)
        L = [self.idx2char.get(str(i),'') for i in range(m+1)]
        L[0]=''; L[self.blank_id]=''
        return L

class BPETokenizer:
    def __init__(self):
        import sentencepiece as spm
        v = json.load(open(BPE_VOCAB_PATH, encoding='utf-8'))
        self.vocab_size = v['vocab_size']; self.blank_id = v.get('blank_id',0)
        self.id2piece = v['id2piece']
        self.sp = spm.SentencePieceProcessor(); self.sp.load(str(BPE_MODEL_PATH))
    def decode_ids(self, ids):
        f, prev = [], -1
        for i in ids:
            if i != prev and i not in (0, self.blank_id): f.append(i)
            prev = i
        return self.sp.decode(f) if f else ''
    def encode(self, text): return self.sp.encode(text)
    def labels(self):
        L = [self.id2piece.get(str(i),'') for i in range(self.vocab_size)]
        L[0]=''
        if self.blank_id != 0: L[self.blank_id]=''
        return L

def edit_dist(a, b):
    m,n=len(a),len(b); dp=list(range(n+1))
    for i in range(1,m+1):
        prev,dp[0]=dp[0],i
        for j in range(1,n+1):
            t=dp[j]; dp[j]=prev if a[i-1]==b[j-1] else 1+min(prev,dp[j],dp[j-1]); prev=t
    return dp[n]

def wer(r,h): a,b=r.split(),h.split(); return edit_dist(a,b)/max(len(a),1)
def cer(r,h): return edit_dist(list(r),list(h))/max(len(r),1)
def greedy(logits, tok):
    return tok.decode_ids(torch.argmax(logits.float(),-1).squeeze(0).tolist())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='best', choices=['best','latest'])
    p.add_argument('--lm', default='none', choices=['none','ngram','neural'])
    p.add_argument('--ngram', default='4gram', choices=['3gram','4gram'])
    p.add_argument('--beam', type=int, default=10)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--device', default='auto')
    args = p.parse_args()

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    print(f"Device: {device}")

    ckpt_file = CHECKPOINT_DIR / ('best_model_ft.pt' if args.checkpoint=='best' else 'latest_checkpoint.pt')
    print(f"Loading: {ckpt_file}")
    ckpt  = torch.load(ckpt_file, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)

    vocab_size = state['ctc_head.weight'].shape[0]
    d_model    = state['ctc_head.weight'].shape[1]
    print(f"Checkpoint vocab_size={vocab_size}, d_model={d_model}")

    tok = BPETokenizer() if vocab_size == 500 else CharTokenizer()
    print(f"Tokenizer: {'BPE' if vocab_size==500 else 'char'} ({tok.vocab_size} tokens)")

    model = ConformerCTCv2(vocab_size=vocab_size, input_dim=80, d_model=d_model,
                           num_layers=12, num_heads=4, conv_kernel_size=31, dropout=0.0)
    miss, unex = model.load_state_dict(state, strict=False)
    if miss:  print(f"  missing: {len(miss)}")
    if unex:  print(f"  unexpected: {len(unex)}")
    model.eval().to(device)
    print("Model OK")

    kd = None
    if args.lm == 'ngram':
        lm_bin = LM_DIR / f"konkani_{args.ngram}.binary"
        if not lm_bin.exists(): lm_bin = LM_DIR / f"konkani_{args.ngram}.arpa"
        try:
            from pyctcdecode import build_ctcdecoder
            kd = build_ctcdecoder(tok.labels(), kenlm_model=str(lm_bin), alpha=0.5, beta=1.0)
            print(f"KenLM ready (beam={args.beam})")
        except ImportError:
            print("pyctcdecode not installed, using greedy")

    samples = [json.loads(l) for l in open(TEST_MANIFEST, encoding='utf-8') if l.strip()]
    if args.max_samples: samples = samples[:args.max_samples]
    print(f"Test samples: {len(samples)}")

    # quick path check on first sample
    first = resolve_audio_path(samples[0]['audio_filepath'])
    print(f"First audio resolves to: {first}")

    total_wer = total_cer = skipped = 0
    results = []

    for s in tqdm(samples, desc="Eval"):
        ap = resolve_audio_path(s['audio_filepath'])
        if not ap: skipped+=1; continue
        audio = load_audio(ap)
        if audio is None: skipped+=1; continue

        mel = compute_mel(audio, device)
        lengths = torch.tensor([mel.size(1)], device=device)
        with torch.no_grad():
            logits, _ = model(mel, lengths)

        ref = s['text'].strip()
        if kd:
            lp = F.log_softmax(logits.float(),-1).squeeze(0).cpu().numpy()
            hyp = kd.decode(lp, beam_width=args.beam)
        else:
            hyp = greedy(logits, tok)

        w,c = wer(ref,hyp), cer(ref,hyp)
        total_wer+=w; total_cer+=c
        results.append({'ref':ref,'hyp':hyp,'wer':w,'cer':c})
        del mel, logits
        if device=='cuda': torch.cuda.empty_cache()

    n = len(results)
    if n == 0: print("No samples evaluated — check audio paths."); return

    avg_wer = total_wer/n*100
    avg_cer = total_cer/n*100
    print(f"""
╔══════════════════════════════════════════╗
  Checkpoint : {ckpt_file.name}
  Vocab      : {vocab_size} tokens
  LM         : {args.lm}
  Evaluated  : {n}/{len(samples)}  (skipped {skipped})
  ─────────────────────────────────────────
  WER        : {avg_wer:.2f}%
  CER        : {avg_cer:.2f}%
╚══════════════════════════════════════════╝""")

    for r in results[:5]:
        print(f"  REF: {r['ref']}\n  HYP: {r['hyp']}\n  WER={r['wer']*100:.1f}% CER={r['cer']*100:.1f}%\n")

    json.dump({'checkpoint':str(ckpt_file),'vocab_size':vocab_size,'lm':args.lm,
               'avg_wer':avg_wer,'avg_cer':avg_cer,'evaluated':n,'skipped':skipped,'samples':results},
              open('eval_results.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)
    print("Saved eval_results.json")

if __name__ == '__main__':
    main()
