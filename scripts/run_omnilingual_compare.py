#!/usr/bin/env python3
"""
Run Meta Omnilingual ASR on Konkani validation set.
Outputs WER, CER, and a CSV with per‑utterance results.
Uses checkpointing to resume interrupted runs.
"""

import os
import json
import csv
import gc
import sys
from pathlib import Path

import torch
import librosa
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCTC
from jiwer import wer, cer

# ============================================================================
# CONFIGURATION – edit these paths as needed
# ============================================================================
MODEL_ID = "facebook/omniASR-CTC-300M"   # ~300M params, ~2GB VRAM
# Alternative: "facebook/omniASR-CTC-1B" or "facebook/omniASR-CTC-3B"

MANIFEST_PATH = "data/konkani-ultimate/val.json"   # your validation set
OUTPUT_CSV = "omnilingual_results.csv"
CHECKPOINT_FILE = "omnilingual_checkpoint.json"
SAVE_INTERVAL = 50    # save checkpoint every N files

# Device selection
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_manifest(path):
    """Load JSONL manifest (each line is a JSON object)."""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

def load_checkpoint():
    """Return processed count and results list if checkpoint exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("processed", 0), data.get("results", [])
    return 0, []

def save_checkpoint(processed, results):
    """Save current progress."""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"processed": processed, "results": results}, f, ensure_ascii=False, indent=2)

def transcribe_audio(model, processor, audio_path):
    """
    Load audio (16kHz mono) and transcribe with Omnilingual model.
    Returns transcribed text (string) or empty string on error.
    """
    try:
        # Load audio – librosa handles many formats
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        if len(audio) == 0:
            return ""

        # Process with processor
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]

        # Clear memory
        torch.mps.empty_cache() if DEVICE == "mps" else torch.cuda.empty_cache()
        gc.collect()

        return text.strip()
    except Exception as e:
        print(f"⚠️ Error processing {audio_path}: {e}", file=sys.stderr)
        return ""

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Meta Omnilingual ASR – Konkani Evaluation")
    print("=" * 60)

    # 1. Load model
    print(f"Loading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCTC.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    print("Model loaded.\n")

    # 2. Load manifest
    if not os.path.exists(MANIFEST_PATH):
        print(f"❌ Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    samples = load_manifest(MANIFEST_PATH)
    print(f"Total utterances: {len(samples)}\n")

    # 3. Load checkpoint
    processed, results = load_checkpoint()
    if processed > 0:
        print(f"🔄 Resuming from utterance {processed + 1} (saved {len(results)} results).")

    # 4. Process remaining samples
    start_idx = processed
    for idx in tqdm(range(start_idx, len(samples)), initial=start_idx, total=len(samples),
                    desc="Transcribing", unit="utt"):
        item = samples[idx]
        audio_path = item.get("audio_filepath", "")
        ref_text = item.get("text", "").strip()

        # Skip missing files
        if not os.path.exists(audio_path):
            print(f"⚠️ File not found: {audio_path}", file=sys.stderr)
            results.append({
                "audio_path": audio_path,
                "reference": ref_text,
                "hypothesis": "",
                "wer": 1.0,
                "cer": 1.0
            })
            continue

        # Transcribe
        hyp_text = transcribe_audio(model, processor, audio_path)

        # Compute per‑utterance WER / CER
        try:
            w = wer(ref_text, hyp_text) if ref_text and hyp_text else 1.0
            c = cer(ref_text, hyp_text) if ref_text and hyp_text else 1.0
        except Exception:
            w, c = 1.0, 1.0

        results.append({
            "audio_path": audio_path,
            "reference": ref_text,
            "hypothesis": hyp_text,
            "wer": w,
            "cer": c
        })

        # Save checkpoint periodically
        if (idx + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(idx + 1, results)
            print(f"\n💾 Checkpoint saved at {idx+1} files.")

    # 5. Final overall metrics
    all_refs = [r["reference"] for r in results if r["reference"]]
    all_hyps = [r["hypothesis"] for r in results if r["hypothesis"]]

    if all_refs and all_hyps:
        overall_wer = wer(all_refs, all_hyps) * 100
        overall_cer = cer(all_refs, all_hyps) * 100
    else:
        overall_wer = overall_cer = float("nan")

    # 6. Print summary
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Model       : {MODEL_ID}")
    print(f"Device      : {DEVICE}")
    print(f"Utterances  : {len(results)}")
    print(f"Overall WER : {overall_wer:.2f}%")
    print(f"Overall CER : {overall_cer:.2f}%")
    print("=" * 60)

    # 7. Save CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["audio_path", "reference", "hypothesis", "wer", "cer"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Detailed results saved to: {OUTPUT_CSV}")

    # 8. Final checkpoint
    save_checkpoint(len(samples), results)
    print("✅ Final checkpoint saved.")

    # 9. Cleanup (optional – remove checkpoint after success)
    # if os.path.exists(CHECKPOINT_FILE):
    #     os.remove(CHECKPOINT_FILE)
    #     print("🧹 Checkpoint file removed.")

if __name__ == "__main__":
    main()