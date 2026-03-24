import sentencepiece as spm, json

sp = spm.SentencePieceProcessor()
sp.load('data/bpe_tokenizer/konkani_bpe.model')

samples = [json.loads(l) for l in open('data/konkani-20gb/train.json', encoding='utf-8').readlines()[:500]]

fail, ok = 0, 0
for s in samples:
    toks = sp.encode(s['text'], out_type=int)
    mel = int(s['duration'] * 16000) // 160
    if mel < len(toks):
        fail += 1
    else:
        ok += 1

# Show first 5 examples
for s in samples[:5]:
    toks = sp.encode(s['text'], out_type=int)
    mel = int(s['duration'] * 16000) // 160
    status = 'OK' if mel >= len(toks) else 'FAIL'
    print(f'[{status}] text="{s["text"]}" | tokens={len(toks)} | mel_frames={mel}')

print(f'\nTotal checked: {len(samples)} | OK: {ok} | CTC-FAIL (T<S): {fail} ({fail/len(samples)*100:.1f}%)')
