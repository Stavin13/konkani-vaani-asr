#!/usr/bin/env python3
"""
Train a SentencePiece BPE tokenizer on Konkani corpus text.
Extracts text from manifests and trains a BPE model.
Run this ONCE before training: python scripts/train_bpe_tokenizer.py
"""
import json, os, sys
from pathlib import Path

MANIFESTS = [
    'data/konkani-10k/train_manifest.json',
    'data/konkani-raw-corpus/manifests/train.json',
    'data/konkani-raw-corpus/manifests/val.json',
]
OUTPUT_DIR = 'data/bpe_tokenizer'
VOCAB_SIZE  = 500 # 1k BPE for ~17h / 20GB Konkani chunk
MODEL_PREFIX = f'{OUTPUT_DIR}/konkani_bpe'

def extract_text(manifests):
    texts = []
    for mf in manifests:
        if not os.path.exists(mf):
            print(f'  Skipping missing: {mf}')
            continue
        with open(mf, encoding='utf-8') as f:
            for line in f:
                try:
                    s = json.loads(line)
                    t = s.get('text', '').strip()
                    if t:
                        texts.append(t)
                except:
                    pass
        print(f'  Loaded {mf}')
    return texts

def main():
    try:
        import sentencepiece as spm
    except ImportError:
        print('Install sentencepiece: pip install sentencepiece')
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Extracting text from manifests...')
    texts = extract_text(MANIFESTS)
    print(f'Total sentences: {len(texts):,}')

    corpus_file = f'{OUTPUT_DIR}/corpus.txt'
    with open(corpus_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(texts))
    print(f'Corpus written: {corpus_file}')

    print(f'Training BPE tokenizer (vocab_size={VOCAB_SIZE})...')
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type='bpe',
        character_coverage=1.0,       # Full coverage for Devanagari
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
        user_defined_symbols=['<blank>'],  # index 4 = CTC blank
        input_sentence_size=500000,
        shuffle_input_sentence=True,
    )

    # Save vocab as JSON for compatibility with training script
    sp = spm.SentencePieceProcessor()
    sp.load(f'{MODEL_PREFIX}.model')
    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    vocab_out = {
        'vocab_size': sp.get_piece_size(),
        'blank_id': 0,   # <pad> used as CTC blank
        'model_path': f'{MODEL_PREFIX}.model',
        'piece2id': vocab,
        'id2piece': {str(i): sp.id_to_piece(i) for i in range(sp.get_piece_size())}
    }
    vocab_json = f'{OUTPUT_DIR}/bpe_vocab.json'
    with open(vocab_json, 'w', encoding='utf-8') as f:
        json.dump(vocab_out, f, ensure_ascii=False, indent=2)

    print(f'\nDone! Files saved:')
    print(f'  Model : {MODEL_PREFIX}.model')
    print(f'  Vocab : {vocab_json}')
    print(f'  Vocab size: {sp.get_piece_size()}')

if __name__ == '__main__':
    main()
