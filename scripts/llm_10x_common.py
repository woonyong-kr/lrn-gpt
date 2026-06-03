#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the LLM 10x long-run experiment tooling."""

from __future__ import annotations

from array import array
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs" / "llm_10x"
LOCAL_DIR = ROOT / "local" / "llm_10x_isolated"
CACHE_DIR = LOCAL_DIR / "cache"
RUNS_DIR = LOCAL_DIR / "runs"
MATRIX_PATH = DOCS_DIR / "run_matrix.csv"
DEFAULT_TRAIN_TEXT = ROOT / "data" / "obsidian_llm_10x_lm_train.txt"
DEFAULT_VAL_TEXT = ROOT / "data" / "obsidian_llm_10x_lm_val.txt"


def ensure_src_import() -> None:
    src_parent = str(ROOT)
    if src_parent not in sys.path:
        sys.path.insert(0, src_parent)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_int(value: Any) -> int:
    return int(float(str(value).strip()))


def parse_float(value: Any) -> float:
    return float(str(value).strip())


def read_matrix(path: Path = MATRIX_PATH) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_u32(path: Path, values: Iterable[int]) -> int:
    data = array("I", (int(value) for value in values))
    if data.itemsize != 4:
        raise RuntimeError("array('I') is not 32-bit on this platform")
    if sys.byteorder != "little":
        data.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        data.tofile(handle)
    return len(data)


def read_u32(path: Path) -> array:
    data = array("I")
    with path.open("rb") as handle:
        data.fromfile(handle, path.stat().st_size // 4)
    if sys.byteorder != "little":
        data.byteswap()
    return data


def cache_dir_for(vocab_size: int, cache_root: Path = CACHE_DIR, min_frequency: int = 2) -> Path:
    return cache_root / f"vocab_{vocab_size}_minfreq_{min_frequency}"


def run_dir_for(run_number: int, runs_root: Path = RUNS_DIR) -> Path:
    return runs_root / f"run_{run_number:04d}"


def unique_vocab_sizes(matrix_path: Path = MATRIX_PATH) -> list[int]:
    rows = read_matrix(matrix_path)
    return sorted({parse_int(row["vocab_size"]) for row in rows})


def unique_tokenizer_settings(matrix_path: Path = MATRIX_PATH) -> list[tuple[int, int]]:
    rows = read_matrix(matrix_path)
    settings: set[tuple[int, int]] = set()
    for row in rows:
        min_frequency = parse_int(row.get("tokenizer_min_frequency", 2))
        settings.add((parse_int(row["vocab_size"]), min_frequency))
    return sorted(settings)


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)
