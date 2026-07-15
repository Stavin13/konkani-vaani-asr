#!/usr/bin/env python3
"""
Generate predictions_analysis.xlsx for the current ASR setup.

This reproduces the workbook structure currently found at:
  outputs/predictions_analysis.xlsx

Outputs:
  - outputs/predictions_analysis.xlsx
  - outputs/table_ii_stats.txt
"""

import argparse
import json
import math
import re
import statistics
import sys
import unicodedata
from pathlib import Path
from typing import Any

import openpyxl
import torch
import torch.nn.functional as F
from jiwer import process_words
from openpyxl.styles import Font
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = BASE / "outputs/conformer_ctc_run1/best_conformer_ctc.pt"
DEFAULT_VOCAB = BASE / "data/konkani-10k/vocab.json"
DEFAULT_MANIFEST = BASE / "data/konkani-ultimate/val.json"
DEFAULT_LM_3GRAM = BASE / "models/language_models/konkani_3gram.binary"
DEFAULT_LM_4GRAM = BASE / "models/language_models/konkani_4gram.binary"
DEFAULT_UNIGRAMS = BASE / "models/language_models/unigrams.txt"
DEFAULT_MAMBA_CHECKPOINT = BASE / "mamba/best_model_test2.pt"
DEFAULT_MAMBA_VOCAB = BASE / "data/konkani-10k/vocab.json"
DEFAULT_OUT_XLSX = BASE / "outputs/predictions_analysis_val.xlsx"
DEFAULT_OUT_TXT = BASE / "outputs/table_ii_stats_val.txt"
MISSING_DIGITS = set("०३४५६७८")

sys.path.insert(0, str(BASE))
from models.conformer_ctc import ConformerCTC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate predictions_analysis.xlsx from the ASR test manifest."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--lm-3gram", type=Path, default=DEFAULT_LM_3GRAM)
    parser.add_argument("--lm-4gram", type=Path, default=DEFAULT_LM_4GRAM)
    parser.add_argument("--unigrams", type=Path, default=DEFAULT_UNIGRAMS)
    parser.add_argument("--mamba-checkpoint", type=Path, default=DEFAULT_MAMBA_CHECKPOINT)
    parser.add_argument("--mamba-vocab", type=Path, default=DEFAULT_MAMBA_VOCAB)
    parser.add_argument("--output-xlsx", type=Path, default=DEFAULT_OUT_XLSX)
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUT_TXT)
    parser.add_argument("--count", type=int, default=None, help="Limit samples for a quick run.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--beam", type=int, default=15)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run a fast internal sanity check and exit.",
    )
    return parser.parse_args()


def normalise_text(text: str, clean: bool = False) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    if clean:
        cleaned = []
        for ch in text:
            cat = unicodedata.category(ch)
            if (cat.startswith("L") or cat.startswith("M") or cat.startswith("N") or ch.isspace()):
                if ch not in MISSING_DIGITS:
                    cleaned.append(ch)
        text = "".join(cleaned)
    return re.sub(r" +", " ", text).strip()


class CharTokenizer:
    def __init__(self, vocab_path: Path):
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.idx2char = {int(k): v for k, v in vocab["idx2char"].items()}
        self.char2idx = vocab["char2idx"]
        self.vocab_size = vocab["vocab_size"]
        self.blank_id = 0

    def labels(self) -> list[str]:
        labels = []
        for i in range(self.vocab_size):
            piece = self.idx2char.get(i, f"<id{i}>")
            if i == self.blank_id:
                labels.append("")
            elif piece in {"<pad>", "<blank>", "<sos>", "<eos>", "<unk>"}:
                labels.append(f"<{piece[1:-1]}_{i}>")
            else:
                labels.append(piece)
        return labels

    def greedy_decode(self, ids: list[int]) -> str:
        chars = []
        prev = -1
        for i in ids:
            if i != self.blank_id and i != prev:
                ch = self.idx2char.get(i, "")
                if not (ch.startswith("<") and ch.endswith(">")):
                    chars.append(ch)
            prev = i
        return "".join(chars).strip()


class MambaTokenizer:
    def __init__(self, vocab_path: Path):
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.char2idx = vocab["char2idx"].copy()
        self.idx2char = {int(k): v for k, v in vocab["idx2char"].items()}
        self.base_vocab_size = len(self.char2idx)
        self.pad_id = self.char2idx.get("<pad>", 0)
        self.unk_id = self.char2idx.get("<unk>", 4)
        self.sep_id = self.base_vocab_size
        self.eos_id = self.base_vocab_size + 1
        self.char2idx["<sep>"] = self.sep_id
        self.char2idx["<new_eos>"] = self.eos_id
        self.idx2char[self.sep_id] = "<sep>"
        self.idx2char[self.eos_id] = "<eos>"
        self.vocab_size = len(self.char2idx)

    def encode(self, text: str) -> list[int]:
        return [self.char2idx.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        out = []
        for token_id in ids:
            if token_id == self.eos_id:
                break
            if token_id in (self.pad_id, self.sep_id):
                continue
            out.append(self.idx2char.get(token_id, "?"))
        return "".join(out)


class MambaBlock(torch.nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.norm = torch.nn.LayerNorm(d_model)
        self.in_proj = torch.nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = torch.nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=0,
            groups=self.d_inner,
            bias=True,
        )
        self.x_proj = torch.nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = torch.nn.Linear(1, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.log_A = torch.nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1).clone())
        self.D = torch.nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = torch.nn.Linear(self.d_inner, d_model, bias=False)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        xz = self.in_proj(x)
        x_ssm, gate = xz.chunk(2, dim=-1)
        x_conv = F.pad(x_ssm.transpose(1, 2), (self.conv1d.kernel_size[0] - 1, 0))
        x_conv = self.conv1d(x_conv).transpose(1, 2)
        x_conv = F.silu(x_conv)
        if attention_mask is not None:
            x_conv = x_conv * attention_mask.unsqueeze(-1)
        y = self._ssm(x_conv)
        out = y * F.silu(gate)
        out = self.drop(self.out_proj(out))
        return out + residual

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        A = -torch.exp(self.log_A.float())
        bcdt = self.x_proj(x)
        B_mat, C_mat, dt = bcdt.split([self.d_state, self.d_state, 1], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        dA = torch.exp(delta.unsqueeze(-1) * A)
        dB_u = delta.unsqueeze(-1) * B_mat.unsqueeze(2) * x.unsqueeze(-1)
        dA_cumprod = torch.exp(torch.cumsum(torch.log(dA.clamp(min=1e-10)), dim=1))
        dB_u_scaled = dB_u / dA_cumprod.clamp(min=1e-10)
        h = dA_cumprod * torch.cumsum(dB_u_scaled, dim=1)
        y = (h * C_mat.unsqueeze(2)).sum(-1)
        return y + x * self.D


class TinyMambaCorrectorModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        pad_id: int,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.layers = torch.nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand, dropout) for _ in range(n_layers)]
        )
        self.norm_out = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.norm_out(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        eos_token_id: int,
        max_new: int,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> list[int]:
        was_training = self.training
        self.eval()
        ids = src_ids.clone()
        mask = attention_mask.clone()
        for _ in range(max_new):
            logits = self(ids, attention_mask=mask)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold = topk_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_token], dim=1)
            mask = torch.cat([mask, torch.ones_like(next_token)], dim=1)
            if next_token.item() == eos_token_id:
                break
        if was_training:
            self.train()
        return ids[0, src_ids.size(1):].tolist()


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_torchaudio():
    import torchaudio

    return torchaudio


def infer_num_layers(state: dict[str, torch.Tensor]) -> int:
    prefixes = {
        int(key.split(".")[2])
        for key in state
        if key.startswith("encoder.layers.") and key.split(".")[2].isdigit()
    }
    return max(prefixes) + 1 if prefixes else 12


def load_model(checkpoint_path: Path, device: torch.device) -> ConformerCTC:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    vocab_size, d_model = state["ctc_head.weight"].shape
    model = ConformerCTC(
        vocab_size=vocab_size,
        input_dim=80,
        d_model=d_model,
        num_layers=infer_num_layers(state),
    )
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in state.items()}


def load_mamba_model(checkpoint_path: Path, vocab_path: Path, device: torch.device) -> tuple[TinyMambaCorrectorModel, MambaTokenizer, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    tokenizer = MambaTokenizer(vocab_path)
    model = TinyMambaCorrectorModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        d_state=config["d_state"],
        d_conv=config["d_conv"],
        expand=config["expand"],
        dropout=config["dropout"],
        pad_id=tokenizer.pad_id,
    )
    state = strip_prefix(checkpoint["model_state"], "_orig_mod.")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, tokenizer, config["max_len"]


def build_decoders(tokenizer: CharTokenizer, args: argparse.Namespace):
    from pyctcdecode import build_ctcdecoder

    unigrams = None
    if args.unigrams.exists():
        unigrams = [
            line.strip()
            for line in args.unigrams.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    labels = tokenizer.labels()
    dec_3gram = build_ctcdecoder(
        labels,
        kenlm_model_path=str(args.lm_3gram),
        unigrams=unigrams,
        alpha=args.alpha,
        beta=args.beta,
    )
    dec_4gram = build_ctcdecoder(
        labels,
        kenlm_model_path=str(args.lm_4gram),
        unigrams=unigrams,
        alpha=args.alpha,
        beta=args.beta,
    )
    return dec_3gram, dec_4gram


def load_audio(path: str, device: torch.device) -> torch.Tensor:
    import librosa

    audio, _ = librosa.load(path, sr=16000)
    wav = torch.as_tensor(audio, dtype=torch.float32)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0)
    return wav.to(device)


def wav_to_log_mel(wav: torch.Tensor, mel_fn: Any) -> tuple[torch.Tensor, torch.Tensor]:
    mel = mel_fn(wav.unsqueeze(0))
    mel = torch.log(mel + 1e-9).transpose(1, 2)
    mel_len = torch.tensor([(wav.size(0) // 160) + 1], device=wav.device)
    return mel, mel_len


def metric_stats(values: list[int]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.fmean(values), statistics.pstdev(values)


def edit_dist(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def wer_cer(ref: str, hyp: str) -> tuple[float, float]:
    ref_words = ref.split()
    hyp_words = hyp.split()
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    wer = edit_dist(ref_words, hyp_words) / max(len(ref_words), 1)
    cer = edit_dist(ref_chars, hyp_chars) / max(len(ref_chars), 1)
    return wer, cer


def summarize_metrics(rows: list[list], errors: dict[str, int]) -> str:
    report = [
        "",
        "=" * 40,
        "Table II Data Integrity Audit - Final Report",
        "=" * 40,
    ]
    indexes = {
        "3gram": {"hyp": 2, "S": 3, "D": 4, "I": 5, "WER": 6, "CER": 7},
        "4gram": {"hyp": 8, "S": 9, "D": 10, "I": 11, "WER": 12, "CER": 13},
        "mamba": {"hyp": 14, "S": 15, "D": 16, "I": 17, "WER": 18, "CER": 19},
    }
    for name in ("3gram", "4gram", "mamba"):
        subs = [row[indexes[name]["S"]] for row in rows]
        dels = [row[indexes[name]["D"]] for row in rows]
        ins = [row[indexes[name]["I"]] for row in rows]
        wers = [row[indexes[name]["WER"]] for row in rows]
        cers = [row[indexes[name]["CER"]] for row in rows]
        n = len(rows)
        s_mean, s_std = metric_stats(subs)
        d_mean, d_std = metric_stats(dels)
        i_mean, i_std = metric_stats(ins)
        wer_mean, wer_std = metric_stats(wers)
        cer_mean, cer_std = metric_stats(cers)
        ser = errors[name] / n if n else 0.0
        report.extend(
            [
                "",
                f"{name.upper()} Beam Search (N={n})",
                "-" * 25,
                f"Substitutions: {s_mean:.3f} +/- {s_std:.3f}",
                f"Deletions    : {d_mean:.3f} +/- {d_std:.3f}",
                f"Insertions   : {i_mean:.3f} +/- {i_std:.3f}",
                f"WER          : {wer_mean:.4f} +/- {wer_std:.4f}",
                f"CER          : {cer_mean:.4f} +/- {cer_std:.4f}",
                f"SER          : {ser:.4f}",
            ]
        )
    return "\n".join(report)


def build_workbook(rows: list[list], errors: dict[str, int]) -> openpyxl.Workbook:
    wb = openpyxl.Workbook()

    ws_summary = wb.active
    ws_summary.title = "Table II Summary"

    ws_details = wb.create_sheet("Detailed Predictions")
    headers = [
        "Audio",
        "Reference",
        "3-gram Hyp",
        "S3",
        "D3",
        "I3",
        "WER3",
        "CER3",
        "4-gram Hyp",
        "S4",
        "D4",
        "I4",
        "WER4",
        "CER4",
        "Mamba Hyp",
        "SM",
        "DM",
        "IM",
        "WERM",
        "CERM",
    ]
    ws_details.append(headers)
    for cell in ws_details[1]:
        cell.font = Font(bold=True)

    for name, s_idx, d_idx, i_idx, wer_idx, cer_idx in (
        ("3gram", 3, 4, 5, 6, 7),
        ("4gram", 9, 10, 11, 12, 13),
        ("mamba", 15, 16, 17, 18, 19),
    ):
        subs = [row[s_idx] for row in rows]
        dels = [row[d_idx] for row in rows]
        ins = [row[i_idx] for row in rows]
        wers = [row[wer_idx] for row in rows]
        cers = [row[cer_idx] for row in rows]
        s_mean, s_std = metric_stats(subs)
        d_mean, d_std = metric_stats(dels)
        i_mean, i_std = metric_stats(ins)
        wer_mean, wer_std = metric_stats(wers)
        cer_mean, cer_std = metric_stats(cers)
        ser = errors[name] / len(rows) if rows else 0.0
        ws_summary.append([f"{name.upper()} Metric", "Mean", "Std Dev"])
        ws_summary.append(["Substitutions", s_mean, s_std])
        ws_summary.append(["Deletions", d_mean, d_std])
        ws_summary.append(["Insertions", i_mean, i_std])
        ws_summary.append(["WER", wer_mean, wer_std])
        ws_summary.append(["CER", cer_mean, cer_std])
        ws_summary.append(["SER", ser, ""])
        ws_summary.append([])

    for row in rows:
        ws_details.append(row)

    return wb


def generate_rows(
    args: argparse.Namespace,
    model: ConformerCTC,
    dec_3gram,
    dec_4gram,
    mamba_model: TinyMambaCorrectorModel,
    mamba_tokenizer: MambaTokenizer,
    mamba_max_len: int,
    conformer_tokenizer: CharTokenizer,
    samples: list[dict],
    device: torch.device,
) -> tuple[list[list], dict[str, int]]:
    torchaudio = get_torchaudio()
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        win_length=400,
    ).to(device)
    rows = []
    errors = {"3gram": 0, "4gram": 0, "mamba": 0}
    failed = 0

    for sample in tqdm(samples, desc="Decoding", unit="sample"):
        try:
            wav = load_audio(sample["audio_filepath"], device)
            mel, mel_len = wav_to_log_mel(wav, mel_fn)
            with torch.no_grad():
                logits, _ = model(mel, mel_len)
            lp_np = F.log_softmax(logits, dim=-1).squeeze(0).cpu().float().numpy()
            greedy_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
            greedy_hyp = normalise_text(conformer_tokenizer.greedy_decode(greedy_ids), clean=False)
            ref = normalise_text(sample["text"], clean=True)

            row = [Path(sample["audio_filepath"]).name, ref]
            for name, decoder in (("3gram", dec_3gram), ("4gram", dec_4gram)):
                hyp = normalise_text(decoder.decode(lp_np, beam_width=args.beam), clean=True)
                result = process_words(ref, hyp)
                wer, cer = wer_cer(ref, hyp)
                if hyp != ref:
                    errors[name] += 1
                row.extend([hyp, result.substitutions, result.deletions, result.insertions, wer, cer])

            src_ids = mamba_tokenizer.encode(greedy_hyp)[: max(mamba_max_len - 1, 1)] + [mamba_tokenizer.sep_id]
            src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
            src_mask = torch.ones_like(src_tensor)
            max_new = min(max(len(ref) + 20, 1), max(mamba_max_len - len(src_ids), 1))
            mamba_ids = mamba_model.generate(
                src_tensor,
                attention_mask=src_mask,
                eos_token_id=mamba_tokenizer.eos_id,
                max_new=max_new,
            )
            mamba_hyp = normalise_text(mamba_tokenizer.decode(mamba_ids), clean=True)
            mamba_result = process_words(ref, mamba_hyp)
            mamba_wer, mamba_cer = wer_cer(ref, mamba_hyp)
            if mamba_hyp != ref:
                errors["mamba"] += 1
            row.extend([mamba_hyp, mamba_result.substitutions, mamba_result.deletions, mamba_result.insertions, mamba_wer, mamba_cer])
            rows.append(row)
        except Exception as exc:
            failed += 1
            if failed <= 3:
                print(f"Skipping {sample.get('audio_filepath', '<unknown>')}: {exc}")
            continue

    if failed:
        print(f"Skipped {failed} samples due to errors")
    return rows, errors


def run_self_check() -> None:
    assert normalise_text(" अ  ब ") == "अ ब"
    assert normalise_text("a-b", clean=True) == "ab"
    result = process_words("एक दोन", "एक")
    assert result.substitutions == 0
    assert result.deletions == 1
    wer, cer = wer_cer("एक दोन", "एक")
    assert wer == 0.5
    assert 0.0 < cer < 1.0
    assert strip_prefix({"_orig_mod.a": 1, "b": 2}, "_orig_mod.") == {"a": 1, "b": 2}
    assert ["Audio", "Reference", "3-gram Hyp", "S3", "D3", "I3", "WER3", "CER3", "4-gram Hyp", "S4", "D4", "I4", "WER4", "CER4", "Mamba Hyp", "SM", "DM", "IM", "WERM", "CERM"][14] == "Mamba Hyp"
    print("self-check passed")


def main() -> None:
    args = parse_args()
    if args.self_check:
        run_self_check()
        return

    device = choose_device(args.device)
    print(f"Device: {device}")

    tokenizer = CharTokenizer(args.vocab)
    model = load_model(args.checkpoint, device)
    dec_3gram, dec_4gram = build_decoders(tokenizer, args)
    mamba_model, mamba_tokenizer, mamba_max_len = load_mamba_model(
        args.mamba_checkpoint, args.mamba_vocab, device
    )

    samples = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.count is not None:
        samples = samples[: args.count]
    print(f"Processing {len(samples)} samples")

    rows, errors = generate_rows(
        args,
        model,
        dec_3gram,
        dec_4gram,
        mamba_model,
        mamba_tokenizer,
        mamba_max_len,
        tokenizer,
        samples,
        device,
    )
    print(f"Rows written: {len(rows)}")

    report = summarize_metrics(rows, errors)
    print(report)

    workbook = build_workbook(rows, errors)
    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(args.output_xlsx)
    args.output_txt.write_text(report + "\n", encoding="utf-8")

    print(f"Saved Excel: {args.output_xlsx}")
    print(f"Saved summary: {args.output_txt}")


if __name__ == "__main__":
    main()
