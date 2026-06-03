#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate or score fixed-prompt samples for LLM repetition/diversity audits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import torch
from tokenizers import Tokenizer

from llm_10x_common import ensure_src_import, read_json, write_json


ensure_src_import()
from src.config import GPTConfig  # noqa: E402
from src.model import GPTModel  # noqa: E402
from src.train import generate  # noqa: E402


FIXED_PROMPTS = ("이 영화는", "정말", "스토리는", "배우들의 연기는")
DEFAULT_DECODING_CONFIG = {
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 100,
    "num_samples_per_prompt": 3,
}
TOKEN_RE = re.compile(r"[가-힣]+|[A-Za-z0-9_]+|[^\s]")
MODEL_CONFIG_KEYS = (
    "vocab_size",
    "context_length",
    "emb_dim",
    "n_heads",
    "n_layers",
    "drop_rate",
    "qkv_bias",
    "ffn_mult",
    "norm_first",
    "norm_eps",
    "activation_name",
    "ffn_dropout_position",
    "attention_impl",
    "tie_embeddings",
    "init_std",
    "seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-jsonl", type=Path, default=None, help="Existing samples with prompt/text or prompt/generated_text fields.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional model checkpoint .pt to generate samples from.")
    parser.add_argument("--config-json", type=Path, default=None, help="Run config/result JSON used with --checkpoint.")
    parser.add_argument("--tokenizer-json", type=Path, default=None, help="HuggingFace tokenizer.json used with --checkpoint.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_DECODING_CONFIG["temperature"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_DECODING_CONFIG["top_k"])
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_DECODING_CONFIG["max_new_tokens"])
    parser.add_argument("--num-samples-per-prompt", type=int, default=DEFAULT_DECODING_CONFIG["num_samples_per_prompt"])
    parser.add_argument("--manual-quality-note", default="")
    parser.add_argument("--output-json", type=Path, default=Path("docs/llm_10x/generation_metrics.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("docs/llm_10x/generation_metrics.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/llm_10x/generation_metrics.md"))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def text_units(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def ngrams(units: list[str], n: int) -> list[tuple[str, ...]]:
    if len(units) < n:
        return []
    return [tuple(units[index : index + n]) for index in range(len(units) - n + 1)]


def distinct_n(text: str, n: int) -> float:
    grams = ngrams(text_units(text), n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def repetition_ratio_ngram(text: str, n: int = 3) -> float:
    grams = ngrams(text_units(text), n)
    if not grams:
        return 0.0
    counts: dict[tuple[str, ...], int] = {}
    for gram in grams:
        counts[gram] = counts.get(gram, 0) + 1
    repeated_positions = sum(count - 1 for count in counts.values() if count > 1)
    return repeated_positions / len(grams)


def sample_metrics(sample: dict[str, Any], manual_quality_note: str = "") -> dict[str, Any]:
    text = str(sample.get("generated_text", sample.get("text", "")))
    units = text_units(text)
    return {
        **sample,
        "generated_text": text,
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
        "repetition_ratio_3gram": repetition_ratio_ngram(text, 3),
        "generated_length_chars": len(text),
        "generated_length_units": len(units),
        "manual_quality_note": str(sample.get("manual_quality_note", manual_quality_note)),
    }


def aggregate_metrics(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        return {
            "distinct_1": math.nan,
            "distinct_2": math.nan,
            "repetition_ratio_3gram": math.nan,
            "average_generated_length": math.nan,
        }
    return {
        "distinct_1": sum(float(row["distinct_1"]) for row in samples) / len(samples),
        "distinct_2": sum(float(row["distinct_2"]) for row in samples) / len(samples),
        "repetition_ratio_3gram": sum(float(row["repetition_ratio_3gram"]) for row in samples) / len(samples),
        "average_generated_length": sum(float(row["generated_length_units"]) for row in samples) / len(samples),
    }


def read_samples_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_model_config(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    source = payload.get("config", payload)
    config = {key: source[key] for key in MODEL_CONFIG_KEYS if key in source}
    return GPTConfig(**config).to_dict()


def generate_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.checkpoint is None or args.config_json is None or args.tokenizer_json is None:
        raise ValueError("--checkpoint, --config-json, and --tokenizer-json are required when --samples-jsonl is not supplied")
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    model = GPTModel(load_model_config(args.config_json)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, Any]] = []
    for prompt in FIXED_PROMPTS:
        prompt_ids = tokenizer.encode(prompt).ids
        for sample_index in range(args.num_samples_per_prompt):
            idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            out = generate(
                model,
                idx,
                max_new_tokens=args.max_new_tokens,
                context_size=int(model.config["context_length"]),
                temperature=args.temperature,
                top_k=args.top_k,
            )
            full_text = tokenizer.decode(out[0].tolist())
            rows.append(
                {
                    "prompt": prompt,
                    "sample_index": sample_index,
                    "text": full_text,
                    "generated_text": full_text[len(prompt) :] if full_text.startswith(prompt) else full_text,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.4f}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# LLM Generation Metrics",
        "",
        "고정 prompt set에서 생성 반복과 다양성을 보는 sanity check입니다. `val_bits_per_char`가 좋아져도 반복률이 올라가면 채택을 보류합니다.",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in payload["aggregate"].items():
        lines.append(f"| `{key}` | {fmt(value)} |")
    lines.extend(
        [
            "",
            "## Samples",
            "",
            "| prompt | sample | distinct_1 | distinct_2 | repetition_3gram | length | note |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["samples"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("prompt", "")).replace("|", "\\|"),
                    str(row.get("sample_index", "")),
                    fmt(row.get("distinct_1")),
                    fmt(row.get("distinct_2")),
                    fmt(row.get("repetition_ratio_3gram")),
                    fmt(row.get("generated_length_units")),
                    str(row.get("manual_quality_note", "")).replace("|", "\\|"),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_samples = read_samples_jsonl(args.samples_jsonl) if args.samples_jsonl else generate_samples(args)
    samples = [sample_metrics(row, manual_quality_note=args.manual_quality_note) for row in raw_samples]
    payload = {
        "schema_version": 1,
        "prompt_set": list(FIXED_PROMPTS),
        "decoding_config": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "num_samples_per_prompt": args.num_samples_per_prompt,
        },
        "aggregate": aggregate_metrics(samples),
        "samples": samples,
    }
    write_json(args.output_json, payload)
    write_csv(args.output_csv, samples)
    write_markdown(args.output_md, payload)
    print(json.dumps({"output_json": str(args.output_json), "output_csv": str(args.output_csv), "output_md": str(args.output_md), "samples": len(samples)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
