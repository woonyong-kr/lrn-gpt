#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare tokenizer and token-id caches for LLM 10x long-run experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from llm_10x_common import CACHE_DIR, DEFAULT_TRAIN_TEXT, DEFAULT_VAL_TEXT, MATRIX_PATH, cache_dir_for, sha256_text, unique_tokenizer_settings, unique_vocab_sizes, write_json, write_u32


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


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
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_backend": "huggingface_tokenizers_bytelevel_bpe",
        "vocab_size_requested": vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "bpe_merge_count": merge_count,
        "min_frequency": min_frequency,
        "tokenizer_min_frequency": min_frequency,
        "char_limit": args.char_limit,
        "train_text_path": str(args.train_text),
        "val_text_path": str(args.val_text),
        "train_text_sha256": sha256_text(train_text),
        "val_text_sha256": sha256_text(val_text),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "train_tokens": train_count,
        "val_tokens": val_count,
        "train_tokens_per_char": train_count / len(train_text),
        "val_tokens_per_char": val_count / len(val_text),
        "elapsed_sec": elapsed,
    }
    write_json(output_dir / "manifest.json", manifest)
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
