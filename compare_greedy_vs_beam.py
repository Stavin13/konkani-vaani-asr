#!/usr/bin/env python3
"""
Greedy vs Beam Search Comparison for best_conformer_ctc.pt
===========================================================
Runs both decoding strategies on the test set and prints a side-by-side
comparison of WER / CER / accuracy.

Usage:
    python compare_greedy_vs_beam.py [--beam_width 5] [--max_samples 200]
"""

import torch
import json
import sys
import time
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import jiwer

sys.path.insert(0, str(Path(__file__).parent))

from models.conformer_ctc import ConformerCTC
from data.audio_processing.audio_processor import AudioProcessor
from data.audio_processing.text_tokenizer import KonkaniTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, vocab_path, device):
    """Load ConformerCTC checkpoint with automatic architecture detection."""
    print(f"\n📂  Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = checkpoint["model_state_dict"]

    print(f"    Epoch  : {checkpoint.get('epoch', 'N/A')}")
    val_loss = checkpoint.get("val_loss", "N/A")
    print(f"    Val loss: {val_loss:.4f}" if isinstance(val_loss, float) else f"    Val loss: {val_loss}")

    tokenizer = KonkaniTokenizer(vocab_path)
    vocab_size = sd["ctc_head.weight"].shape[0]
    print(f"    Vocab size (from weights): {vocab_size}")

    if "encoder.layers.0.ff1.1.weight" in sd:
        d_model = sd["encoder.layers.0.ff1.1.weight"].shape[1]
    elif "encoder.input_proj.weight" in sd:
        d_model = sd["encoder.input_proj.weight"].shape[0]
    else:
        d_model = 256

    layer_indices = {
        int(k.split(".")[2])
        for k in sd
        if k.startswith("encoder.layers.") and k.split(".")[2].isdigit()
    }
    num_layers = max(layer_indices) + 1 if layer_indices else 12

    print(f"    d_model   : {d_model}")
    print(f"    num_layers: {num_layers}")

    model = ConformerCTC(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        input_dim=80,
        num_heads=4,
        conv_kernel_size=31,
        dropout=0.1,
    )
    model.load_state_dict(sd)
    model.to(device).eval()
    print("    ✅  Model loaded successfully!\n")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def get_log_probs(model, audio_processor, audio_path, device):
    """Return (log_probs [T, V], input_length) for one audio file."""
    audio_features, _ = audio_processor.process_audio_file(str(audio_path))
    T = audio_features.size(0)
    x = audio_features.unsqueeze(0).to(device)          # (1, T, 80)
    lengths = torch.tensor([T], device=device)

    with torch.no_grad():
        ctc_logits, _ = model(x, lengths)               # (1, T, V)

    log_probs = torch.nn.functional.log_softmax(ctc_logits[0], dim=-1)  # (T, V)
    return log_probs.cpu(), T


# ──────────────────────────────────────────────────────────────────────────────
# Greedy decode
# ──────────────────────────────────────────────────────────────────────────────

def greedy_decode(log_probs, tokenizer):
    """Standard CTC greedy (argmax + collapse).
    
    Note: the model uses <pad> (id 0) as a de-facto blank at most time steps;
    tokenizer.decode() already strips all special tokens, so greedy works fine.
    """
    # Collect all special token ids so we can exclude them from the prefix
    special_ids = {
        tokenizer.blank_id,
        tokenizer.pad_id,
        tokenizer.sos_id,
        tokenizer.eos_id,
        tokenizer.char2idx.get(KonkaniTokenizer.UNK_TOKEN, -1),
    }

    tokens = torch.argmax(log_probs, dim=-1).numpy()   # (T,)
    decoded, prev = [], None
    for t in tokens:
        if t not in special_ids and t != prev:
            decoded.append(t)
        prev = t
    return tokenizer.decode(decoded)


# ──────────────────────────────────────────────────────────────────────────────
# Beam search decode  (pure-Python, no external lib required)
# ──────────────────────────────────────────────────────────────────────────────

NEG_INF = float("-inf")


def beam_search_decode(
    log_probs: torch.Tensor,
    tokenizer,
    beam_width: int = 5,
    token_top_k: int = 40,
):
    """
    Prefix-beam-search CTC decoder with per-step token pruning.

    Key insight: this model outputs <pad> (id=0) with near-1 probability at most
    time steps instead of the CTC blank (id=1).  We treat ALL special tokens as
    blank-like — they collapse the frame without extending the prefix.

    At each time step:
      1. Keep only the top `token_top_k` tokens by log-prob to limit branching.
      2. For each beam prefix:
         a. Extend with any special/blank token → same prefix accumulates prob.
         b. Extend with each top-K real token → new prefix.
      3. Prune the resulting beam back to `beam_width` prefixes.
    """
    # Build set of all ids that act as blanks (special + CTC blank)
    special_ids = {
        tokenizer.blank_id,
        tokenizer.pad_id,
        tokenizer.sos_id,
        tokenizer.eos_id,
        tokenizer.char2idx.get(KonkaniTokenizer.UNK_TOKEN, -1),
    }

    lp = log_probs.numpy()       # (T, V) log-softmax
    T, V = lp.shape
    top_k = min(token_top_k, V)

    # For each time step, we'll aggregate blank mass as log-sum over all special ids
    def blank_mass(lp_t):
        """Log-sum of all special-token log-probs at time step t."""
        masses = [lp_t[sid] for sid in special_ids if 0 <= sid < V]
        result = masses[0]
        for m in masses[1:]:
            result = np.logaddexp(result, m)
        return result

    # Initial beam: empty prefix with p_blank=1 (log=0), p_non_blank=-inf
    # p_blank here means "last emitted token was blank/special"
    beam = {(): (0.0, NEG_INF)}

    def score(entry):
        pb, pnb = entry
        return np.logaddexp(pb, pnb)

    for t in range(T):
        # Pre-prune: keep top-beam_width prefixes
        beam = dict(
            sorted(beam.items(), key=lambda x: score(x[1]), reverse=True)[:beam_width]
        )

        # Find top-K non-special tokens at this time step
        lp_t = lp[t]
        # Sort all tokens descending, take top_k that are NOT special
        all_sorted = np.argsort(lp_t)[::-1]
        top_real_tokens = [int(c) for c in all_sorted if int(c) not in special_ids][:top_k]

        bm = blank_mass(lp_t)   # combined blank/special log-prob at this step

        next_beam: dict = {}

        for prefix, (p_b, p_nb) in beam.items():
            p_total = np.logaddexp(p_b, p_nb)

            # 1. Extend with blank/special  →  same prefix
            new_pb = p_total + bm
            old = next_beam.get(prefix, (NEG_INF, NEG_INF))
            next_beam[prefix] = (np.logaddexp(old[0], new_pb), old[1])

            # 2. Extend with each real (non-special) token
            for c in top_real_tokens:
                lp_c = lp_t[c]
                new_prefix = prefix + (c,)

                # If last real token in prefix equals c, only p_blank path can
                # extend without merging duplicate character occurrences
                if prefix and prefix[-1] == c:
                    new_pnb = p_b + lp_c
                else:
                    new_pnb = p_total + lp_c

                old = next_beam.get(new_prefix, (NEG_INF, NEG_INF))
                next_beam[new_prefix] = (old[0], np.logaddexp(old[1], new_pnb))

        beam = next_beam

    # Best prefix
    best_prefix = max(beam, key=lambda p: score(beam[p]))
    return tokenizer.decode(list(best_prefix))


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def metrics(preds, refs):
    wer = jiwer.wer(refs, preds) * 100
    cer = jiwer.cer(refs, preds) * 100
    return {"wer": wer, "cer": cer, "word_acc": 100 - wer, "char_acc": 100 - cer}


# ──────────────────────────────────────────────────────────────────────────────
# Main comparison
# ──────────────────────────────────────────────────────────────────────────────

def compare(checkpoint, test_manifest, vocab_path, device, beam_width, max_samples, show_examples):
    model, tokenizer = load_model(checkpoint, vocab_path, device)
    audio_processor = AudioProcessor()

    with open(test_manifest, "r", encoding="utf-8") as f:
        test_data = [json.loads(l) for l in f if l.strip()]

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"📊  Test samples: {len(test_data)}")
    print(f"🔭  Beam width  : {beam_width}\n")

    greedy_preds, beam_preds, refs = [], [], []
    greedy_times, beam_times = [], []
    errors = 0

    for i, item in enumerate(tqdm(test_data, desc="Decoding")):
        try:
            log_probs, _ = get_log_probs(model, audio_processor, item["audio_filepath"], device)
        except Exception as e:
            tqdm.write(f"  ⚠️  Skip {item['audio_filepath']}: {e}")
            errors += 1
            continue

        ref = item["text"]

        t0 = time.perf_counter()
        g = greedy_decode(log_probs, tokenizer)
        greedy_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        b = beam_search_decode(log_probs, tokenizer, beam_width)
        beam_times.append(time.perf_counter() - t0)

        greedy_preds.append(g)
        beam_preds.append(b)
        refs.append(ref)

        if i < show_examples:
            tqdm.write(f"\n  ── Sample {i+1} ───────────────────────────")
            tqdm.write(f"  Reference : {ref}")
            tqdm.write(f"  Greedy    : {g}")
            tqdm.write(f"  Beam (w={beam_width}): {b}")

    if not refs:
        print("❌  No samples decoded."); return

    gm = metrics(greedy_preds, refs)
    bm = metrics(beam_preds, refs)

    # ── Side-by-side table ─────────────────────────────────────────────────────
    W = 60
    print("\n" + "═" * W)
    print("  📊  GREEDY vs BEAM SEARCH — RESULTS")
    print("═" * W)
    ckpt_name = Path(checkpoint).name
    print(f"  Checkpoint : {ckpt_name}")
    print(f"  Samples    : {len(refs)}  (errors skipped: {errors})")
    print(f"  Beam width : {beam_width}")
    print("─" * W)
    print(f"  {'Metric':<28} {'Greedy':>10}  {'Beam':>10}  {'Δ (Beam−Greedy)':>14}")
    print("─" * W)

    def row(label, g_val, b_val, lower_better=True):
        delta = b_val - g_val
        arrow = "↓" if (delta < 0 and lower_better) or (delta > 0 and not lower_better) else "↑"
        marker = f"{arrow} {abs(delta):.2f}"
        print(f"  {label:<28} {g_val:>9.2f}%  {b_val:>9.2f}%  {marker:>14}")

    row("Word Error Rate (WER)",    gm["wer"],      bm["wer"],      lower_better=True)
    row("Char Error Rate (CER)",    gm["cer"],      bm["cer"],      lower_better=True)
    row("Word Accuracy",            gm["word_acc"], bm["word_acc"], lower_better=False)
    row("Char Accuracy",            gm["char_acc"], bm["char_acc"], lower_better=False)
    print("─" * W)

    # Speed
    avg_g = np.mean(greedy_times) * 1000
    avg_b = np.mean(beam_times) * 1000
    print(f"  {'Avg decode time (greedy)':<28} {avg_g:>9.1f}ms")
    print(f"  {'Avg decode time (beam)':<28} {avg_b:>9.1f}ms  ({avg_b/avg_g:.1f}×  slower)")
    print("═" * W)

    # Verdict
    wer_gain = gm["wer"] - bm["wer"]
    print("\n  💡  Verdict:")
    if wer_gain > 2:
        print(f"  ✅  Beam search is BETTER by {wer_gain:.2f}% WER — worth the extra compute.")
    elif wer_gain > 0:
        print(f"  ⚠️   Beam search is slightly better ({wer_gain:.2f}% WER) — marginal gain.")
    elif wer_gain == 0:
        print("  ➡️   Both produce identical results on this set.")
    else:
        print(f"  ⚠️   Greedy is BETTER by {abs(wer_gain):.2f}% WER — beam search not helping.")
    print()

    # Save JSON
    out_file = Path(checkpoint).stem + "_greedy_vs_beam.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint),
                "samples": len(refs),
                "beam_width": beam_width,
                "greedy": gm,
                "beam": bm,
                "avg_greedy_ms": avg_g,
                "avg_beam_ms": avg_b,
            },
            f,
            indent=2,
        )
    print(f"  💾  Results saved → {out_file}\n")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare greedy vs beam search on CTC model")
    parser.add_argument("--checkpoint", default="outputs/conformer_ctc_run1/best_conformer_ctc.pt")
    parser.add_argument("--test_manifest", default="data/konkani-10k/test_manifest.json")
    parser.add_argument("--vocab", default="data/konkani-10k/vocab.json")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width (default: 5)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit sample count for quick tests")
    parser.add_argument("--show_examples", type=int, default=5, help="Number of per-sample previews")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"🖥️   Device: {device}")
    compare(
        args.checkpoint,
        args.test_manifest,
        args.vocab,
        device,
        args.beam_width,
        args.max_samples,
        args.show_examples,
    )


if __name__ == "__main__":
    main()
