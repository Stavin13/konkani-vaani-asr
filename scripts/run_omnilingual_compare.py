#!/usr/bin/env python3
"""
Run Meta Omnilingual ASR (local .pt) on Konkani validation set.
Uses the processor from the HuggingFace repo, but loads the model weights from the local .pt file.
Also sets the language code if needed (gom_Deva).
"""

import os
import json
import csv
import gc
import sys
from pathlib import Path

import torch
import librosa
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCTC, AutoConfig
from jiwer import wer, cer

# ============================================================================
# CONFIGURATION – adjust paths as needed
# ============================================================================
# Local model checkpoint
MODEL_PT_PATH = "/Volumes/data&proj/konkani/omniASR-CTC-300M-v2.pt"

# The HF repo ID (used for tokenizer/processor and config)
HF_MODEL_ID = "facebook/omniASR-CTC-300M"   # or "facebook/omniASR-CTC-300M-v2" if available

# Manifest and output paths
MANIFEST_PATH = "data/konkani-ultimate/val.json"   # your validation set
OUTPUT_CSV = "omnilingual_results.csv"
CHECKPOINT_FILE = "omnilingual_checkpoint.json"
SAVE_INTERVAL = 50

# Language hint (for models that support it; CTC-300M ignores it but we keep for reference)
LANGUAGE = "gom_Deva"   # Konkani (Devanagari)

# Device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# ============================================================================
# HELPERS
# ============================================================================

def load_manifest(path):
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("processed", 0), data.get("results", [])
    return 0, []

def save_checkpoint(processed, results):
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"processed": processed, "results": results}, f, ensure_ascii=False, indent=2)

def transcribe_audio(model, processor, audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        if len(audio) == 0:
            return ""

        # Process with the same tokenizer as the original HF model
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]

        if DEVICE == "mps":
            torch.mps.empty_cache()
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return text.strip()
    except Exception as e:
        print(f"⚠️ Error on {audio_path}: {e}", file=sys.stderr)
        return ""

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Meta Omnilingual ASR – Konkani Evaluation")
    print(f"Model file: {MODEL_PT_PATH}")
    print(f"Language:   {LANGUAGE}")
    print("=" * 60)

    # 1. Load processor and config from HF repo
    print(f"Loading processor/config from: {HF_MODEL_ID}")
    try:
        processor = AutoProcessor.from_pretrained(HF_MODEL_ID)
        config = AutoConfig.from_pretrained(HF_MODEL_ID)
    except Exception as e:
        print(f"❌ Failed to load from HF: {e}")
        print("🔄 Trying with trust_remote_code=True...")
        processor = AutoProcessor.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
        config = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)

    # 2. Load model from local .pt file
    print(f"Loading model weights from: {MODEL_PT_PATH}")
    if not os.path.exists(MODEL_PT_PATH):
        print(f"❌ Local checkpoint not found: {MODEL_PT_PATH}")
        print("🔄 Falling back to loading from HuggingFace...")
        model = AutoModelForCTC.from_pretrained(HF_MODEL_ID).to(DEVICE)
    else:
        # Instantiate model with the config (architecture) then load state dict
        model = AutoModelForCTC.from_config(config)
        state_dict = torch.load(MODEL_PT_PATH, map_location=DEVICE)
        # Sometimes the saved dict has a 'model' or 'state_dict' key – handle both
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE)

    model.eval()
    print("✅ Model loaded.\n")

    # 3. Load manifest
    if not os.path.exists(MANIFEST_PATH):
        print(f"❌ Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    samples = load_manifest(MANIFEST_PATH)
    print(f"Total utterances: {len(samples)}\n")

    # 4. Load checkpoint
    processed, results = load_checkpoint()
    if processed > 0:
        print(f"🔄 Resuming from utterance {processed + 1} (saved {len(results)} results).")

    # 5. Process
    start_idx = processed
    for idx in tqdm(range(start_idx, len(samples)), initial=start_idx, total=len(samples),
                    desc="Transcribing", unit="utt"):
        item = samples[idx]
        audio_path = item.get("audio_filepath", "")
        ref_text = item.get("text", "").strip()

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

        hyp_text = transcribe_audio(model, processor, audio_path)

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

        if (idx + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(idx + 1, results)
            print(f"\n💾 Checkpoint saved at {idx+1} files.")

    # 6. Final overall metrics
    all_refs = [r["reference"] for r in results if r["reference"]]
    all_hyps = [r["hypothesis"] for r in results if r["hypothesis"]]

    if all_refs and all_hyps:
        overall_wer = wer(all_refs, all_hyps) * 100
        overall_cer = cer(all_refs, all_hyps) * 100
    else:
        overall_wer = overall_cer = float("nan")

    # 7. Summary
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Model file  : {MODEL_PT_PATH}")
    print(f"Device      : {DEVICE}")
    print(f"Language    : {LANGUAGE}")
    print(f"Utterances  : {len(results)}")
    print(f"Overall WER : {overall_wer:.2f}%")
    print(f"Overall CER : {overall_cer:.2f}%")
    print("=" * 60)

    # 8. Save CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["audio_path", "reference", "hypothesis", "wer", "cer"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Detailed results saved to: {OUTPUT_CSV}")

    # 9. Final checkpoint
    save_checkpoint(len(samples), results)
    print("✅ Final checkpoint saved.")

if __name__ == "__main__":
    main()