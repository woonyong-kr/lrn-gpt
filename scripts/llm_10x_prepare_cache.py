#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare tokenizer and token-id caches for LLM 10x long-run experiments."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from llm_10x_common import CACHE_DIR, DEFAULT_TRAIN_TEXT, DEFAULT_VAL_TEXT, MATRIX_PATH, cache_dir_for, sha256_text, unique_tokenizer_settings, unique_vocab_sizes, write_json, write_u32


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
TOKEN_LENGTH_BINS = ("length_1", "length_2", "length_3_4", "length_5_8", "length_9_16", "length_17_plus")
PROFILE_CSV_FIELDS = (
    "corpus_sha256",
    "train_sha256",
    "val_sha256",
    "requested_vocab_size",
    "actual_vocab_size",
    "tokenizer_min_frequency",
    "bpe_merge_count",
    "train_chars",
    "val_chars",
    "train_tokens",
    "val_tokens",
    "train_tokens_per_char",
    "val_tokens_per_char",
    "train_chars_per_token",
    "val_chars_per_token",
    *TOKEN_LENGTH_BINS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--train-text", type=Path, default=DEFAULT_TRAIN_TEXT)
    parser.add_argument("--val-text", type=Path, default=DEFAULT_VAL_TEXT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--vocab-size", type=int, action="append", help="Vocab size to cache. Defaults to every tokenizer setting in the matrix.")
    parser.add_argument("--min-frequency", type=int, action="append", help="BPE merge min frequency. Defaults to every tokenizer setting in the matrix, or 2 when --vocab-size is supplied.")
    parser.add_argument("--char-limit", type=int, default=None, help="Optional per-split char limit for smoke tests.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_text(path: Path, char_limit: int | None) -> str:
    text = path.read_text(encoding="utf-8")
    if char_limit is not None:
        text = text[:char_limit]
    if not text.strip():
        raise ValueError(f"empty corpus split: {path}")
    return text


def cache_exists(output_dir: Path) -> bool:
    return all((output_dir / name).exists() for name in ("tokenizer.json", "train_ids.u32", "val_ids.u32", "manifest.json"))


def safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def combined_corpus_sha256(train_text: str, val_text: str) -> str:
    return sha256_text(f"{train_text}\n<VAL_SPLIT>\n{val_text}")


def token_length_histogram(vocab: dict[str, int]) -> dict[str, int]:
    histogram = {key: 0 for key in TOKEN_LENGTH_BINS}
    for token in vocab:
        length = len(token)
        if length <= 1:
            histogram["length_1"] += 1
        elif length == 2:
            histogram["length_2"] += 1
        elif length <= 4:
            histogram["length_3_4"] += 1
        elif length <= 8:
            histogram["length_5_8"] += 1
        elif length <= 16:
            histogram["length_9_16"] += 1
        else:
            histogram["length_17_plus"] += 1
    return histogram


def top_merge_rules(tokenizer_payload: dict[str, Any], limit: int = 50) -> list[str]:
    merges = tokenizer_payload.get("model", {}).get("merges", [])
    rules: list[str] = []
    for merge in merges[:limit]:
        if isinstance(merge, (list, tuple)):
            rules.append(" ".join(str(piece) for piece in merge))
        else:
            rules.append(str(merge))
    return rules


def profile_from_manifest(manifest: dict[str, Any], tokenizer_payload: dict[str, Any]) -> dict[str, Any]:
    vocab = tokenizer_payload.get("model", {}).get("vocab", {})
    histogram = token_length_histogram(vocab)
    requested_vocab_size = int(manifest.get("requested_vocab_size", manifest.get("vocab_size_requested", 0)))
    train_tokens = int(manifest.get("train_tokens", 0))
    val_tokens = int(manifest.get("val_tokens", 0))
    train_chars = int(manifest.get("train_chars", 0))
    val_chars = int(manifest.get("val_chars", 0))
    train_sha = str(manifest.get("train_sha256", manifest.get("train_text_sha256", "")))
    val_sha = str(manifest.get("val_sha256", manifest.get("val_text_sha256", "")))
    profile = {
        "corpus_sha256": manifest.get("corpus_sha256", ""),
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "requested_vocab_size": requested_vocab_size,
        "vocab_size_requested": requested_vocab_size,
        "actual_vocab_size": int(manifest.get("actual_vocab_size", len(vocab))),
        "tokenizer_min_frequency": int(manifest.get("tokenizer_min_frequency", manifest.get("min_frequency", 0))),
        "bpe_merge_count": int(manifest.get("bpe_merge_count", len(tokenizer_payload.get("model", {}).get("merges", [])))),
        "train_chars": train_chars,
        "val_chars": val_chars,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_tokens_per_char": float(manifest.get("train_tokens_per_char", safe_ratio(train_tokens, train_chars))),
        "val_tokens_per_char": float(manifest.get("val_tokens_per_char", safe_ratio(val_tokens, val_chars))),
        "train_chars_per_token": float(manifest.get("train_chars_per_token", safe_ratio(train_chars, train_tokens))),
        "val_chars_per_token": float(manifest.get("val_chars_per_token", safe_ratio(val_chars, val_tokens))),
        "token_length_histogram": histogram,
        "top_50_merge_rules": top_merge_rules(tokenizer_payload),
        "special_token_ids": manifest.get("special_token_ids", {}),
    }
    return profile


def write_profile_files(output_dir: Path, profile: dict[str, Any]) -> None:
    write_json(output_dir / "tokenizer_profile.json", profile)
    row = {field: profile.get(field, "") for field in PROFILE_CSV_FIELDS if field not in TOKEN_LENGTH_BINS}
    row.update(profile.get("token_length_histogram", {}))
    with (output_dir / "tokenizer_profile.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PROFILE_CSV_FIELDS))
        writer.writeheader()
        writer.writerow(row)


def sync_manifest_profile(output_dir: Path, manifest: dict[str, Any], tokenizer_payload: dict[str, Any], train_text: str, val_text: str) -> dict[str, Any]:
    train_tokens = int(manifest.get("train_tokens", 0))
    val_tokens = int(manifest.get("val_tokens", 0))
    train_chars = int(manifest.get("train_chars", len(train_text)))
    val_chars = int(manifest.get("val_chars", len(val_text)))
    train_sha = sha256_text(train_text)
    val_sha = sha256_text(val_text)
    special_token_ids = {
        token: tokenizer_payload.get("model", {}).get("vocab", {}).get(token)
        for token in SPECIAL_TOKENS
    }
    manifest.update(
        {
            "corpus_sha256": combined_corpus_sha256(train_text, val_text),
            "train_sha256": train_sha,
            "val_sha256": val_sha,
            "train_text_sha256": train_sha,
            "val_text_sha256": val_sha,
            "requested_vocab_size": int(manifest.get("requested_vocab_size", manifest.get("vocab_size_requested", 0))),
            "vocab_size_requested": int(manifest.get("requested_vocab_size", manifest.get("vocab_size_requested", 0))),
            "train_chars_per_token": safe_ratio(train_chars, train_tokens),
            "val_chars_per_token": safe_ratio(val_chars, val_tokens),
            "token_length_histogram": token_length_histogram(tokenizer_payload.get("model", {}).get("vocab", {})),
            "top_50_merge_rules": top_merge_rules(tokenizer_payload),
            "special_token_ids": special_token_ids,
        }
    )
    profile = profile_from_manifest(manifest, tokenizer_payload)
    write_json(output_dir / "manifest.json", manifest)
    write_profile_files(output_dir, profile)
    return manifest


def tokenizer_settings(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.vocab_size:
        min_frequencies = args.min_frequency or [2]
        return sorted({(vocab_size, min_frequency) for vocab_size in args.vocab_size for min_frequency in min_frequencies})
    if args.min_frequency:
        vocab_sizes = unique_vocab_sizes(args.matrix)
        return sorted({(vocab_size, min_frequency) for vocab_size in vocab_sizes for min_frequency in args.min_frequency})
    return unique_tokenizer_settings(args.matrix)


def build_cache(vocab_size: int, min_frequency: int, args: argparse.Namespace, train_text: str, val_text: str) -> dict[str, object]:
    output_dir = cache_dir_for(vocab_size, args.cache_dir, min_frequency)
    if cache_exists(output_dir) and not args.force:
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        tokenizer_payload = json.loads((output_dir / "tokenizer.json").read_text(encoding="utf-8"))
        sync_manifest_profile(output_dir, manifest, tokenizer_payload, train_text, val_text)
        return {"vocab_size": vocab_size, "min_frequency": min_frequency, "status": "exists", "path": str(output_dir)}

    if args.dry_run:
        return {"vocab_size": vocab_size, "min_frequency": min_frequency, "status": "would_build", "path": str(output_dir)}

    start = time.perf_counter()
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator([train_text], trainer=trainer, length=1)
    train_ids = tokenizer.encode(train_text).ids
    val_ids = tokenizer.encode(val_text).ids
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    train_count = write_u32(output_dir / "train_ids.u32", train_ids)
    val_count = write_u32(output_dir / "val_ids.u32", val_ids)
    elapsed = time.perf_counter() - start
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    merge_count = len(tokenizer_payload.get("model", {}).get("merges", []))
    actual_vocab_size = tokenizer.get_vocab_size()
    train_sha = sha256_text(train_text)
    val_sha = sha256_text(val_text)
    special_token_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_backend": "huggingface_tokenizers_bytelevel_bpe",
        "corpus_sha256": combined_corpus_sha256(train_text, val_text),
        "requested_vocab_size": vocab_size,
        "vocab_size_requested": vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "bpe_merge_count": merge_count,
        "min_frequency": min_frequency,
        "tokenizer_min_frequency": min_frequency,
        "char_limit": args.char_limit,
        "train_text_path": str(args.train_text),
        "val_text_path": str(args.val_text),
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "train_text_sha256": train_sha,
        "val_text_sha256": val_sha,
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "train_tokens": train_count,
        "val_tokens": val_count,
        "train_tokens_per_char": train_count / len(train_text),
        "val_tokens_per_char": val_count / len(val_text),
        "train_chars_per_token": len(train_text) / train_count,
        "val_chars_per_token": len(val_text) / val_count,
        "token_length_histogram": token_length_histogram(tokenizer_payload.get("model", {}).get("vocab", {})),
        "top_50_merge_rules": top_merge_rules(tokenizer_payload),
        "special_token_ids": special_token_ids,
        "elapsed_sec": elapsed,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_profile_files(output_dir, profile_from_manifest(manifest, tokenizer_payload))
    return {"vocab_size": vocab_size, "min_frequency": min_frequency, "status": "built", "path": str(output_dir), "elapsed_sec": round(elapsed, 3), "train_tokens": train_count, "val_tokens": val_count}


def main() -> None:
    args = parse_args()
    settings = tokenizer_settings(args)
    train_text = read_text(args.train_text, args.char_limit)
    val_text = read_text(args.val_text, args.char_limit)
    rows = [build_cache(vocab_size, min_frequency, args, train_text, val_text) for vocab_size, min_frequency in settings]
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
