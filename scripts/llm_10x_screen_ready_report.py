#!/usr/bin/env python3
"""Build screen-ready plots and an interim report for isolated LLM 10x runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PLOT_AVAILABLE = True
except ModuleNotFoundError:
    plt = None
    np = None
    PLOT_AVAILABLE = False


MEANINGFUL_PHASE_MIN_READY = 2
MIN_TOKENIZER_FREQUENCY = 2
SPECIAL_TOKEN_COUNT = 4
BYTE_TOKEN_COUNT = 256
MERGE_TOKEN_START_ID = SPECIAL_TOKEN_COUNT + BYTE_TOKEN_COUNT
HANGUL_RE = re.compile(r"[가-힣]")
ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")
WORDLIKE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[가-힣]+")
PHASE_LABELS = {
    "phase1_lr": "Learning Rate",
    "phase3_vocab": "Vocab Size",
    "phase6_context": "Context Length",
    "phase11_ffn_mult": "FFN Mult",
}
PHASE_ORDER = ("phase1_lr", "phase3_vocab", "phase6_context", "phase11_ffn_mult")
PHASE_COLORS = {
    "phase1_lr": "#3b82f6",
    "phase3_vocab": "#10b981",
    "phase6_context": "#f59e0b",
    "phase11_ffn_mult": "#8b5cf6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=Path("docs/llm_10x/aggregate_summary.csv"))
    parser.add_argument("--meta-json", type=Path, default=Path("docs/llm_10x/aggregate_meta.json"))
    parser.add_argument("--all-results-jsonl", type=Path, default=Path("docs/llm_10x/all_run_results.jsonl"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/llm_10x/screen_ready_figures"))
    parser.add_argument("--report", type=Path, default=Path("docs/llm_10x/screen_ready_report.md"))
    parser.add_argument("--cache-dir", type=Path, default=Path("local/llm_10x_isolated/cache"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path("data/obsidian_llm_10x_manifest.json"))
    parser.add_argument("--dataset-comparison", type=Path, default=Path("data/obsidian_llm_10x_comparison.json"))
    parser.add_argument("--train-text", type=Path, default=Path("data/obsidian_llm_10x_lm_train.txt"))
    parser.add_argument("--val-text", type=Path, default=Path("data/obsidian_llm_10x_lm_val.txt"))
    return parser.parse_args()


def to_float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def warm_tokens_per_sec(row: dict[str, Any]) -> float:
    value = to_float(row.get("tokens_per_sec_after_warmup_median"))
    if value == value:
        return value
    return to_float(row.get("tokens_per_sec_median"))


def read_summary(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def read_ledger(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bytes_to_unicode_decoder() -> dict[str, int]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = byte_values[:]
    next_codepoint = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + next_codepoint)
            next_codepoint += 1
    return {chr(codepoint): byte for byte, codepoint in zip(byte_values, codepoints)}


BYTELEVEL_DECODER = bytes_to_unicode_decoder()


def bytelevel_decode(token: str) -> str:
    if token.startswith("<") and token.endswith(">"):
        return token
    pieces: list[bytes] = []
    for char in token:
        byte = BYTELEVEL_DECODER.get(char)
        pieces.append(bytes([byte]) if byte is not None else char.encode("utf-8"))
    return b"".join(pieces).decode("utf-8", errors="replace")


def read_u32_ids(path: Path) -> array:
    ids = array("I")
    with path.open("rb") as handle:
        ids.fromfile(handle, path.stat().st_size // 4)
    if sys.byteorder != "little":
        ids.byteswap()
    return ids


def percentile(values: list[float] | list[int], ratio: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * ratio
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] * (high - index) + ordered[high] * (index - low))


def text_corpus_stats(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    non_ws = sum(1 for char in text if not char.isspace())
    hangul_chars = sum(1 for char in text if "가" <= char <= "힣")
    ascii_alpha_chars = sum(1 for char in text if ("a" <= char <= "z") or ("A" <= char <= "Z"))
    digit_chars = sum(1 for char in text if char.isdigit())
    whitespace_words = text.split()
    wordlike_units = WORDLIKE_RE.findall(text)
    return {
        "path": str(path),
        "chars": len(text),
        "non_ws": non_ws,
        "hangul_chars": hangul_chars,
        "ascii_alpha_chars": ascii_alpha_chars,
        "digit_chars": digit_chars,
        "whitespace_words": len(whitespace_words),
        "unique_whitespace_words": len(set(whitespace_words)),
        "wordlike_units": len(wordlike_units),
        "unique_wordlike_units": len(set(wordlike_units)),
        "hangul_non_ws_ratio": hangul_chars / non_ws if non_ws else math.nan,
        "ascii_alpha_non_ws_ratio": ascii_alpha_chars / non_ws if non_ws else math.nan,
    }


def tokenizer_cache_dir(cache_dir: Path, vocab_size: int, min_frequency: int = MIN_TOKENIZER_FREQUENCY) -> Path:
    return cache_dir / f"vocab_{vocab_size}_minfreq_{min_frequency}"


def tokenizer_diagnostic(cache_dir: Path, vocab_size: int, min_frequency: int = MIN_TOKENIZER_FREQUENCY) -> dict[str, Any] | None:
    cache_path = tokenizer_cache_dir(cache_dir, vocab_size, min_frequency)
    required = ("manifest.json", "tokenizer.json", "train_ids.u32", "val_ids.u32")
    if not all((cache_path / name).exists() for name in required):
        return None

    manifest = read_json(cache_path / "manifest.json")
    tokenizer_payload = read_json(cache_path / "tokenizer.json")
    vocab_map = tokenizer_payload.get("model", {}).get("vocab", {})
    id_to_token = {int(token_id): token for token, token_id in vocab_map.items()}
    actual_vocab_size = to_int(manifest.get("actual_vocab_size"))
    decoded = {token_id: bytelevel_decode(id_to_token.get(token_id, "")) for token_id in range(actual_vocab_size)}

    train_ids = read_u32_ids(cache_path / "train_ids.u32")
    val_ids = read_u32_ids(cache_path / "val_ids.u32")
    train_counts = Counter(train_ids)
    val_counts = Counter(val_ids)
    total_train = len(train_ids)
    total_val = len(val_ids)

    all_ids = set(range(actual_vocab_size))
    used_train_ids = set(train_counts)
    used_val_ids = set(val_counts)
    special_ids = set(range(SPECIAL_TOKEN_COUNT))
    byte_ids = set(range(SPECIAL_TOKEN_COUNT, MERGE_TOKEN_START_ID))
    merge_ids = set(range(MERGE_TOKEN_START_ID, actual_vocab_size))
    hangul_ids = {token_id for token_id in all_ids if HANGUL_RE.search(decoded[token_id])}
    ascii_ids = {token_id for token_id in all_ids if ASCII_ALPHA_RE.search(decoded[token_id])}

    def mass(token_ids: set[int], counts: Counter[int], total: int) -> float:
        return sum(counts.get(token_id, 0) for token_id in token_ids) / total if total else math.nan

    def train_type_count_at_most(threshold: int, token_ids: set[int] | None = None) -> int:
        candidates = all_ids if token_ids is None else token_ids
        return sum(1 for token_id in candidates if train_counts.get(token_id, 0) <= threshold)

    def appeared_train_type_count_at_most(threshold: int) -> int:
        return sum(1 for token_id in used_train_ids if train_counts[token_id] <= threshold)

    def appeared_train_mass_at_most(threshold: int) -> float:
        return sum(train_counts[token_id] for token_id in used_train_ids if train_counts[token_id] <= threshold) / total_train

    sorted_counts = train_counts.most_common()
    top1_mass = sorted_counts[0][1] / total_train if sorted_counts else math.nan
    top10_mass = sum(count for _, count in sorted_counts[:10]) / total_train if total_train else math.nan
    top100_mass = sum(count for _, count in sorted_counts[:100]) / total_train if total_train else math.nan
    merge_lengths = [len(decoded[token_id]) for token_id in merge_ids]
    used_merge_lengths = [len(decoded[token_id]) for token_id in used_train_ids & merge_ids]
    hangul_merge_lengths = [len(decoded[token_id]) for token_id in hangul_ids & merge_ids]
    merge_position_count = sum(train_counts[token_id] for token_id in used_train_ids & merge_ids)

    top_tokens = [
        {
            "id": token_id,
            "freq": count,
            "share": count / total_train,
            "decoded": decoded[token_id].replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r"),
        }
        for token_id, count in sorted_counts[:10]
    ]
    top_hangul: list[dict[str, Any]] = []
    for token_id, count in sorted_counts:
        if token_id in hangul_ids:
            top_hangul.append(
                {
                    "id": token_id,
                    "freq": count,
                    "share": count / total_train,
                    "decoded": decoded[token_id].replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r"),
                }
            )
        if len(top_hangul) >= 10:
            break

    return {
        "vocab_size": vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "bpe_merge_count": len(tokenizer_payload.get("model", {}).get("merges", [])),
        "train_chars": to_int(manifest.get("train_chars")),
        "val_chars": to_int(manifest.get("val_chars")),
        "train_tokens": total_train,
        "val_tokens": total_val,
        "train_tokens_per_char": to_float(manifest.get("train_tokens_per_char")),
        "val_tokens_per_char": to_float(manifest.get("val_tokens_per_char")),
        "train_chars_per_token": to_int(manifest.get("train_chars")) / total_train if total_train else math.nan,
        "val_chars_per_token": to_int(manifest.get("val_chars")) / total_val if total_val else math.nan,
        "used_train_types": len(used_train_ids),
        "used_val_types": len(used_val_ids),
        "unused_train_types": actual_vocab_size - len(used_train_ids),
        "unused_val_types": actual_vocab_size - len(used_val_ids),
        "used_merge_types": len(used_train_ids & merge_ids),
        "unused_merge_types": len(merge_ids - used_train_ids),
        "vocab_types_freq0": train_type_count_at_most(0),
        "vocab_types_freq1": train_type_count_at_most(1),
        "vocab_types_freq5": train_type_count_at_most(5),
        "vocab_types_freq20": train_type_count_at_most(20),
        "appeared_types_freq1": appeared_train_type_count_at_most(1),
        "appeared_types_freq5": appeared_train_type_count_at_most(5),
        "appeared_types_freq20": appeared_train_type_count_at_most(20),
        "appeared_mass_freq1": appeared_train_mass_at_most(1),
        "appeared_mass_freq5": appeared_train_mass_at_most(5),
        "appeared_mass_freq20": appeared_train_mass_at_most(20),
        "merge_types_freq0": train_type_count_at_most(0, merge_ids),
        "merge_types_freq20": train_type_count_at_most(20, merge_ids),
        "merge_train_mass": mass(merge_ids, train_counts, total_train),
        "byte_train_mass": mass(byte_ids, train_counts, total_train),
        "special_train_mass": mass(special_ids, train_counts, total_train),
        "merge_val_mass": mass(merge_ids, val_counts, total_val),
        "byte_val_mass": mass(byte_ids, val_counts, total_val),
        "top1_mass": top1_mass,
        "top10_mass": top10_mass,
        "top100_mass": top100_mass,
        "weighted_decoded_chars_per_token": sum(train_counts[token_id] * len(decoded[token_id]) for token_id in used_train_ids) / total_train,
        "weighted_merge_decoded_chars": (
            sum(train_counts[token_id] * len(decoded[token_id]) for token_id in used_train_ids & merge_ids) / merge_position_count
            if merge_position_count
            else math.nan
        ),
        "merge_len_mean": statistics.mean(merge_lengths) if merge_lengths else math.nan,
        "merge_len_p50": percentile(merge_lengths, 0.5),
        "merge_len_p95": percentile(merge_lengths, 0.95),
        "merge_len_max": max(merge_lengths) if merge_lengths else 0,
        "used_merge_len_p95": percentile(used_merge_lengths, 0.95),
        "used_merge_len_max": max(used_merge_lengths) if used_merge_lengths else 0,
        "hangul_vocab_types": len(hangul_ids),
        "hangul_merge_types": len(hangul_ids & merge_ids),
        "hangul_used_types": len(used_train_ids & hangul_ids),
        "hangul_train_mass": mass(hangul_ids, train_counts, total_train),
        "hangul_val_mass": mass(hangul_ids, val_counts, total_val),
        "hangul_types_freq20": train_type_count_at_most(20, hangul_ids),
        "hangul_merge_len_p50": percentile(hangul_merge_lengths, 0.5),
        "hangul_merge_len_p95": percentile(hangul_merge_lengths, 0.95),
        "hangul_merge_len_max": max(hangul_merge_lengths) if hangul_merge_lengths else 0,
        "ascii_vocab_types": len(ascii_ids),
        "ascii_train_mass": mass(ascii_ids, train_counts, total_train),
        "top_tokens": top_tokens,
        "top_hangul": top_hangul,
    }


def ready_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in summary_rows:
        if to_int(row.get("completed_runs")) >= 3 and to_float(row.get("final_val_bits_per_char_median")) == to_float(row.get("final_val_bits_per_char_median")):
            rows.append(row)
    return rows


def meaningful_phases(rows: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["phase"])] += 1
    phases = [phase for phase in PHASE_ORDER if counts.get(phase, 0) >= MEANINGFUL_PHASE_MIN_READY]
    phases.extend(sorted(phase for phase, count in counts.items() if count >= MEANINGFUL_PHASE_MIN_READY and phase not in phases))
    return phases


def axis_value(row: dict[str, Any]) -> float:
    return to_float(row.get("axis_value"))


def raw_results_by_condition(ledger_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger_rows:
        result = row.get("result", {})
        condition_id = str(result.get("condition_id") or row.get("condition_id"))
        if condition_id:
            grouped[condition_id].append(result)
    return grouped


def metric_values(raw_by_condition: dict[str, list[dict[str, Any]]], condition_id: str, metric: str) -> list[float]:
    values = [to_float(row.get(metric)) for row in raw_by_condition.get(condition_id, [])]
    return [value for value in values if value == value]


def write_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_ranking(rows: list[dict[str, Any]], output_path: Path) -> None:
    ranked = sorted(rows, key=lambda row: to_float(row["final_val_bits_per_char_median"]))
    labels = [str(row["condition_id"]) for row in ranked]
    values = [to_float(row["final_val_bits_per_char_median"]) for row in ranked]
    errors = [to_float(row.get("final_val_bits_per_char_iqr"), 0.0) / 2 for row in ranked]
    colors = [PHASE_COLORS.get(str(row["phase"]), "#64748b") for row in ranked]

    fig_height = max(4.8, 0.36 * len(labels) + 1.2)
    plt.figure(figsize=(9.2, fig_height))
    y = np.arange(len(labels))
    plt.barh(y, values, xerr=errors, color=colors, alpha=0.88, error_kw={"elinewidth": 1.0, "capsize": 2})
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Median validation bits/char (lower is better)")
    plt.title("Screen-ready condition ranking")
    plt.grid(axis="x", alpha=0.22)
    for index, row in enumerate(ranked):
        phase_label = PHASE_LABELS.get(str(row["phase"]), str(row["phase"]))
        plt.text(values[index] + 0.012, index, phase_label, va="center", fontsize=8, color="#334155")
    write_fig(output_path)


def plot_speed_quality(rows: list[dict[str, Any]], output_path: Path) -> None:
    plt.figure(figsize=(8.6, 5.4))
    for phase in sorted({str(row["phase"]) for row in rows}):
        phase_rows = [row for row in rows if row["phase"] == phase]
        x = [to_float(row.get("elapsed_sec_median")) / 60.0 for row in phase_rows]
        y = [to_float(row.get("final_val_bits_per_char_median")) for row in phase_rows]
        labels = [str(row["condition_id"]) for row in phase_rows]
        plt.scatter(x, y, s=58, color=PHASE_COLORS.get(phase, "#64748b"), label=PHASE_LABELS.get(phase, phase), alpha=0.9)
        for xi, yi, label in zip(x, y, labels):
            plt.text(xi + 0.015, yi, label, fontsize=8, color="#334155")
    plt.xlabel("Median elapsed minutes per run (lower is faster)")
    plt.ylabel("Median validation bits/char (lower is better)")
    plt.title("Quality vs runtime among screen-ready conditions")
    plt.grid(alpha=0.22)
    plt.legend(fontsize=8)
    write_fig(output_path)


def plot_phase_metric(
    phase: str,
    rows: list[dict[str, Any]],
    raw_by_condition: dict[str, list[dict[str, Any]]],
    metric: str,
    median_column: str,
    iqr_column: str,
    output_path: Path,
    ylabel: str,
    title_suffix: str,
    zero_line: bool = False,
) -> None:
    phase_rows = sorted([row for row in rows if row["phase"] == phase], key=axis_value)
    if not phase_rows:
        return
    x = np.array([axis_value(row) for row in phase_rows], dtype=float)
    y = np.array([to_float(row.get(median_column)) for row in phase_rows], dtype=float)
    err = np.array([to_float(row.get(iqr_column), 0.0) / 2 for row in phase_rows], dtype=float)

    plt.figure(figsize=(8.4, 5.0))
    color = PHASE_COLORS.get(phase, "#64748b")
    plt.errorbar(x, y, yerr=err, marker="o", color=color, linewidth=2, capsize=3, label="median +/- IQR/2")
    for row in phase_rows:
        values = metric_values(raw_by_condition, str(row["condition_id"]), metric)
        if not values:
            continue
        jitter_width = 0.008 * (max(x) - min(x) if len(x) > 1 else max(abs(x[0]), 1.0))
        jitter = np.linspace(-jitter_width, jitter_width, num=len(values)) if len(values) > 1 else np.array([0.0])
        plt.scatter(np.full(len(values), axis_value(row)) + jitter, values, s=24, color="#0f172a", alpha=0.52, zorder=3)
    for xi, yi, row in zip(x, y, phase_rows):
        plt.text(xi, yi, f" {row['condition_id']}", fontsize=8, color="#334155", va="bottom")
    if zero_line:
        plt.axhline(0.0, color="#94a3b8", linewidth=1, linestyle="--")
    if phase == "phase1_lr":
        plt.xscale("log")
    plt.xlabel(f"{PHASE_LABELS.get(phase, phase)} value")
    plt.ylabel(ylabel)
    plt.title(f"{PHASE_LABELS.get(phase, phase)} - {title_suffix}")
    plt.grid(alpha=0.22)
    write_fig(output_path)


def plot_phase_speed(phase: str, rows: list[dict[str, Any]], output_path: Path) -> None:
    phase_rows = sorted([row for row in rows if row["phase"] == phase], key=axis_value)
    if not phase_rows:
        return
    labels = [str(row["axis_value"]) for row in phase_rows]
    x = np.arange(len(labels))
    elapsed_min = [to_float(row.get("elapsed_sec_median")) / 60.0 for row in phase_rows]
    tokens_sec = [warm_tokens_per_sec(row) for row in phase_rows]
    color = PHASE_COLORS.get(phase, "#64748b")

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.6), sharex=True)
    axes[0].bar(x, elapsed_min, color=color, alpha=0.82)
    axes[0].set_ylabel("Median minutes/run")
    axes[0].set_title(f"{PHASE_LABELS.get(phase, phase)} - runtime and throughput")
    axes[0].grid(axis="y", alpha=0.22)
    axes[1].plot(x, tokens_sec, marker="o", color="#0f172a", linewidth=2)
    axes[1].set_ylabel("Median warm tok/s")
    axes[1].set_xlabel(f"{PHASE_LABELS.get(phase, phase)} value")
    axes[1].grid(alpha=0.22)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    write_fig(output_path)


def phase_best_row(rows: list[dict[str, Any]], phase: str) -> dict[str, Any] | None:
    phase_rows = [row for row in rows if row["phase"] == phase]
    if not phase_rows:
        return None
    return min(phase_rows, key=lambda row: to_float(row["final_val_bits_per_char_median"]))


def available_tokenizer_vocab_sizes(cache_dir: Path, min_frequency: int = MIN_TOKENIZER_FREQUENCY) -> list[int]:
    vocab_sizes: list[int] = []
    pattern = f"vocab_*_minfreq_{min_frequency}"
    for path in cache_dir.glob(pattern):
        if not path.is_dir():
            continue
        try:
            vocab_sizes.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(vocab_sizes))


def svg_text(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_fmt_float(value: Any, digits: int = 2) -> str:
    number = to_float(value)
    if number != number:
        return ""
    return f"{number:.{digits}f}"


def svg_fmt_pct(value: Any, digits: int = 1) -> str:
    number = to_float(value)
    if number != number:
        return ""
    return f"{number * 100:.{digits}f}%"


def svg_card(x: int, y: int, width: int, height: int, title: str, body: list[str], accent: str) -> str:
    lines = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" fill="#ffffff" stroke="#d8dee8"/>',
        f'<rect x="{x}" y="{y}" width="6" height="{height}" rx="3" fill="{accent}"/>',
        f'<text x="{x + 22}" y="{y + 32}" font-size="19" font-weight="700" fill="#0f172a">{svg_text(title)}</text>',
    ]
    for index, text in enumerate(body):
        lines.append(
            f'<text x="{x + 22}" y="{y + 64 + 25 * index}" font-size="15" fill="#334155">{svg_text(text)}</text>'
        )
    return "\n".join(lines)


def write_svg(path: Path, body: str, width: int = 1100, height: int = 620) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="#f8fafc"/>
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans KR", Arial, sans-serif; }}
  </style>
{body}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def linear_scale(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if domain_max == domain_min:
        return (range_min + range_max) / 2
    return range_min + (value - domain_min) * (range_max - range_min) / (domain_max - domain_min)


def draw_axes(x: int, y: int, width: int, height: int, y_label: str, x_label: str = "vocab size") -> list[str]:
    return [
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" stroke="#94a3b8"/>',
        f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" stroke="#94a3b8"/>',
        f'<text x="{x}" y="{y - 14}" font-size="13" fill="#475569">{svg_text(y_label)}</text>',
        f'<text x="{x + width - 80}" y="{y + height + 44}" font-size="13" fill="#475569">{svg_text(x_label)}</text>',
    ]


def polyline(points: list[tuple[float, float]], color: str) -> str:
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>'


def paper_figures(
    figures_dir: Path,
    ready: list[dict[str, Any]],
    tokenizer_diags: list[dict[str, Any]],
    train_stats: dict[str, Any],
    val_stats: dict[str, Any],
    dataset_manifest: dict[str, Any],
) -> dict[str, Path]:
    output_dir = figures_dir.parent / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "paper_claim_map": output_dir / "paper_claim_map.svg",
        "paper_ranking": output_dir / "paper_screen_ready_ranking.svg",
        "paper_dataset": output_dir / "paper_dataset_composition.svg",
        "paper_vocab_tradeoff": output_dir / "paper_vocab_tradeoff.svg",
        "paper_token_distribution": output_dir / "paper_token_distribution.svg",
    }
    write_claim_map(paths["paper_claim_map"], ready, tokenizer_diags, train_stats, dataset_manifest)
    write_screen_ready_ranking(paths["paper_ranking"], ready)
    write_dataset_composition(paths["paper_dataset"], train_stats, val_stats, dataset_manifest)
    write_vocab_tradeoff(paths["paper_vocab_tradeoff"], ready, tokenizer_diags)
    write_token_distribution(paths["paper_token_distribution"], tokenizer_diags)
    return paths


def write_claim_map(path: Path, ready: list[dict[str, Any]], tokenizer_diags: list[dict[str, Any]], train_stats: dict[str, Any], dataset_manifest: dict[str, Any]) -> None:
    ranked = sorted(ready, key=lambda row: to_float(row.get("final_val_bits_per_char_median")))
    best = ranked[0] if ranked else {}
    vocab10000 = next((diag for diag in tokenizer_diags if to_int(diag.get("vocab_size")) == 10000), {})
    counts = dataset_manifest.get("counts", {})
    body = [
        '<text x="48" y="58" font-size="30" font-weight="800" fill="#0f172a">LLM 10x 실험의 현재 결론 지도</text>',
        '<text x="48" y="91" font-size="16" fill="#475569">이 그림은 최종 결론이 아니라, screen-ready 조건에서 지금 말할 수 있는 주장과 보류해야 할 주장을 분리한다.</text>',
        svg_card(
            48,
            130,
            500,
            155,
            "기준 데이터셋",
            [
                f"{fmt_int(counts.get('documents'))}개 문서, {fmt_int(counts.get('chunks'))}개 chunk",
                f"train {fmt_int(train_stats.get('chars'))} 문자, 공백 기준 {fmt_int(train_stats.get('whitespace_words'))} 단어",
                f"한글 비중 {fmt_pct(train_stats.get('hangul_non_ws_ratio'))}, ASCII 비중 {fmt_pct(train_stats.get('ascii_alpha_non_ws_ratio'))}",
            ],
            "#2563eb",
        ),
        svg_card(
            590,
            130,
            452,
            155,
            "현재 최상위 조건",
            [
                f"{best.get('condition_id', '')}: validation bits/char {fmt_float(best.get('final_val_bits_per_char_median'), 5)}",
                "단, claim-ready 반복 수에는 아직 도달하지 않음",
                "따라서 실험 우선순위용 결론으로 제한",
            ],
            "#059669",
        ),
        svg_card(
            48,
            330,
            500,
            155,
            "Vocab 해석",
            [
                "6000 -> 10000에서 validation bits/char는 개선",
                f"하지만 V10000의 20회 이하 type은 {fmt_int(vocab10000.get('vocab_types_freq20'))}개",
                "압축 이득과 tail token 위험을 함께 판단해야 함",
            ],
            "#ea580c",
        ),
        svg_card(
            590,
            330,
            452,
            155,
            "말하면 안 되는 결론",
            [
                "큰 vocab일수록 무조건 좋다",
                "NSMC 결과가 Obsidian LLM 10x를 대체한다",
                "token-level loss만으로 tokenizer를 비교한다",
            ],
            "#7c3aed",
        ),
    ]
    write_svg(path, "\n".join(body))


def write_screen_ready_ranking(path: Path, ready: list[dict[str, Any]]) -> None:
    ranked = sorted(ready, key=lambda row: to_float(row.get("final_val_bits_per_char_median")))[:10]
    if not ranked:
        write_svg(path, "")
        return
    values = [to_float(row.get("final_val_bits_per_char_median")) for row in ranked]
    min_value = min(values)
    max_value = max(values)
    chart_x = 250
    chart_y = 112
    chart_w = 760
    row_h = 39
    body = [
        '<text x="48" y="58" font-size="29" font-weight="800" fill="#0f172a">Figure 1. Screen-ready 조건 순위</text>',
        '<text x="48" y="88" font-size="15" fill="#475569">질문: 현재 완료된 반복 n>=3 조건 중 validation bits/char가 낮은 조건은 무엇인가?</text>',
        '<text x="250" y="106" font-size="13" fill="#64748b">quality score: lower validation loss = longer bar</text>',
    ]
    for index, row in enumerate(ranked):
        y = chart_y + index * row_h
        value = values[index]
        bar_w = linear_scale(value, min_value, max_value, 660, 280)
        color = PHASE_COLORS.get(str(row.get("phase")), "#64748b")
        body.extend(
            [
                f'<text x="48" y="{y + 24}" font-size="15" font-weight="700" fill="#0f172a">{index + 1}. {svg_text(row.get("condition_id"))}</text>',
                f'<text x="150" y="{y + 24}" font-size="13" fill="#64748b">{svg_text(PHASE_LABELS.get(str(row.get("phase")), str(row.get("phase"))))}</text>',
                f'<rect x="{chart_x}" y="{y + 7}" width="{bar_w:.1f}" height="24" rx="6" fill="{color}" opacity="0.88"/>',
                f'<text x="{chart_x + bar_w + 12:.1f}" y="{y + 25}" font-size="14" fill="#0f172a">{svg_fmt_float(value, 5)}</text>',
            ]
        )
    body.extend(
        [
            '<rect x="48" y="535" width="1004" height="48" rx="8" fill="#fff7ed" stroke="#fed7aa"/>',
            '<text x="70" y="565" font-size="15" fill="#9a3412">해석: 이 순위는 최종 논문 결론이 아니라 다음 반복 보강 대상을 고르는 근거다. claim-ready 반복 수가 아직 0개다.</text>',
        ]
    )
    write_svg(path, "\n".join(body))


def write_dataset_composition(path: Path, train_stats: dict[str, Any], val_stats: dict[str, Any], dataset_manifest: dict[str, Any]) -> None:
    def composition(stats: dict[str, Any]) -> list[tuple[str, float, str]]:
        non_ws = max(to_float(stats.get("non_ws")), 1.0)
        hangul = to_float(stats.get("hangul_chars")) / non_ws
        ascii_alpha = to_float(stats.get("ascii_alpha_chars")) / non_ws
        digit = to_float(stats.get("digit_chars")) / non_ws
        other = max(0.0, 1.0 - hangul - ascii_alpha - digit)
        return [("한글", hangul, "#2563eb"), ("ASCII", ascii_alpha, "#059669"), ("숫자", digit, "#f59e0b"), ("기호/기타", other, "#7c3aed")]

    body = [
        '<text x="48" y="58" font-size="29" font-weight="800" fill="#0f172a">Figure 2. 이 실험은 어떤 언어/문자 분포를 학습하는가?</text>',
        '<text x="48" y="88" font-size="15" fill="#475569">질문: 한글 tokenizer 논의를 이 데이터셋에 그대로 적용할 수 있는가?</text>',
    ]
    for row_index, (label, stats) in enumerate((("train", train_stats), ("validation", val_stats))):
        y = 155 + row_index * 118
        body.append(f'<text x="70" y="{y - 14}" font-size="18" font-weight="700" fill="#0f172a">{label}</text>')
        x = 210
        for name, ratio, color in composition(stats):
            width = 680 * ratio
            body.append(f'<rect x="{x:.1f}" y="{y - 38}" width="{width:.1f}" height="34" fill="{color}" opacity="0.9"/>')
            if width > 150:
                body.append(f'<text x="{x + 8:.1f}" y="{y - 15}" font-size="12" fill="#ffffff">{svg_text(name)} {svg_fmt_pct(ratio, 1)}</text>')
            x += width
        body.append(f'<text x="920" y="{y - 14}" font-size="14" fill="#334155">{fmt_int(stats.get("chars"))} 문자</text>')
    legend_y = 360
    x = 70
    for name, _, color in composition(train_stats):
        body.extend(
            [
                f'<rect x="{x}" y="{legend_y}" width="16" height="16" rx="3" fill="{color}"/>',
                f'<text x="{x + 24}" y="{legend_y + 13}" font-size="14" fill="#334155">{svg_text(name)}</text>',
            ]
        )
        x += 132
    body.extend(
        [
            '<rect x="48" y="430" width="1004" height="76" rx="8" fill="#eff6ff" stroke="#bfdbfe"/>',
            '<text x="70" y="462" font-size="15" fill="#1e3a8a">관찰: train과 validation 모두 한글은 약 6%대이고, ASCII가 약 75%다.</text>',
            '<text x="70" y="488" font-size="15" fill="#1e3a8a">해석: 이 실험의 주 결론은 한글 전용 tokenizer가 아니라 영어/코드/LLM 자료 중심 혼합 corpus tokenizer에 대한 결론이다.</text>',
        ]
    )
    write_svg(path, "\n".join(body))


def write_vocab_tradeoff(path: Path, ready: list[dict[str, Any]], tokenizer_diags: list[dict[str, Any]]) -> None:
    ready_vocab = {
        to_int(row.get("axis_value")): to_float(row.get("final_val_bits_per_char_median"))
        for row in ready
        if row.get("phase") == "phase3_vocab"
    }
    vocabs = [vocab for vocab in (6000, 8000, 10000) if vocab in ready_vocab]
    diag_by_vocab = {to_int(diag.get("vocab_size")): diag for diag in tokenizer_diags}
    body = [
        '<text x="48" y="58" font-size="29" font-weight="800" fill="#0f172a">Figure 3. Vocab을 키우면 무엇을 얻고 무엇을 잃는가?</text>',
        '<text x="48" y="88" font-size="15" fill="#475569">질문: V6000 -> V10000 개선을 tokenizer 권고로 바로 말해도 되는가?</text>',
    ]
    if not vocabs:
        write_svg(path, "\n".join(body))
        return
    left_x, top_y, chart_w, chart_h = 80, 150, 420, 300
    right_x = 610
    body.extend(draw_axes(left_x, top_y, chart_w, chart_h, "validation bits/char"))
    bits = [ready_vocab[vocab] for vocab in vocabs]
    bit_min, bit_max = min(bits), max(bits)
    bit_pad = max(0.02, (bit_max - bit_min) * 0.18)
    points = []
    for index, vocab in enumerate(vocabs):
        x = linear_scale(index, 0, max(len(vocabs) - 1, 1), left_x + 25, left_x + chart_w - 25)
        y = linear_scale(ready_vocab[vocab], bit_min - bit_pad, bit_max + bit_pad, top_y + chart_h, top_y)
        points.append((x, y))
        body.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#059669"/>',
                f'<text x="{x - 28:.1f}" y="{top_y + chart_h + 28}" font-size="13" fill="#334155">V{vocab // 1000}k</text>',
                f'<text x="{x - 30:.1f}" y="{y - 14:.1f}" font-size="13" font-weight="700" fill="#065f46">{svg_fmt_float(ready_vocab[vocab], 3)}</text>',
            ]
        )
    body.append(polyline(points, "#059669"))
    body.append('<text x="92" y="515" font-size="15" font-weight="700" fill="#065f46">loss 관찰: V6000 -> V10000은 좋아진다.</text>')

    body.extend(draw_axes(right_x, top_y, chart_w, chart_h, "train 20회 이하 vocab type"))
    tails = [to_float(diag_by_vocab[vocab].get("vocab_types_freq20")) for vocab in vocabs if vocab in diag_by_vocab]
    if tails:
        tail_min, tail_max = min(tails), max(tails)
        tail_pad = max(30, (tail_max - tail_min) * 0.18)
        points = []
        for index, vocab in enumerate(vocabs):
            if vocab not in diag_by_vocab:
                continue
            tail = to_float(diag_by_vocab[vocab].get("vocab_types_freq20"))
            x = linear_scale(index, 0, max(len(vocabs) - 1, 1), right_x + 25, right_x + chart_w - 25)
            y = linear_scale(tail, tail_min - tail_pad, tail_max + tail_pad, top_y + chart_h, top_y)
            points.append((x, y))
            body.extend(
                [
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#ea580c"/>',
                    f'<text x="{x - 28:.1f}" y="{top_y + chart_h + 28}" font-size="13" fill="#334155">V{vocab // 1000}k</text>',
                    f'<text x="{x - 20:.1f}" y="{y - 14:.1f}" font-size="13" font-weight="700" fill="#9a3412">{fmt_int(tail)}</text>',
                ]
            )
        body.append(polyline(points, "#ea580c"))
    body.append('<text x="622" y="515" font-size="15" font-weight="700" fill="#9a3412">risk 관찰: rare/tail type도 함께 늘어난다.</text>')
    body.extend(
        [
            '<rect x="48" y="545" width="1004" height="50" rx="8" fill="#fff7ed" stroke="#fed7aa"/>',
            '<text x="70" y="576" font-size="15" fill="#9a3412">결론: V10000은 유망하지만, 더 큰 vocab 권고 전에는 tail token, top-token 쏠림, validation gap을 같이 확인해야 한다.</text>',
        ]
    )
    write_svg(path, "\n".join(body))


def write_token_distribution(path: Path, tokenizer_diags: list[dict[str, Any]]) -> None:
    selected = [diag for diag in sorted(tokenizer_diags, key=lambda row: to_int(row.get("vocab_size"))) if to_int(diag.get("vocab_size")) in (6000, 8000, 10000, 12000)]
    body = [
        '<text x="48" y="58" font-size="29" font-weight="800" fill="#0f172a">Figure 4. Merge token이 한쪽으로 쏠렸는가?</text>',
        '<text x="48" y="88" font-size="15" fill="#475569">질문: vocab 증가가 하나의 거대 token 집중으로 나타나는가, 아니면 tail token 증가로 나타나는가?</text>',
    ]
    if not selected:
        write_svg(path, "\n".join(body))
        return
    chart_x, chart_y, chart_w, chart_h = 90, 145, 900, 320
    body.extend(draw_axes(chart_x, chart_y, chart_w, chart_h, "train token 비중", "vocab size"))
    series = [
        ("top1", "top1_mass", "#2563eb"),
        ("top10", "top10_mass", "#059669"),
        ("top100", "top100_mass", "#7c3aed"),
        ("merge token", "merge_train_mass", "#ea580c"),
    ]
    for name, key, color in series:
        points = []
        for index, diag in enumerate(selected):
            value = to_float(diag.get(key))
            x = linear_scale(index, 0, max(len(selected) - 1, 1), chart_x + 35, chart_x + chart_w - 35)
            y = linear_scale(value, 0.0, 0.70, chart_y + chart_h, chart_y)
            points.append((x, y))
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
            if key in ("top1_mass", "merge_train_mass"):
                body.append(f'<text x="{x - 30:.1f}" y="{y - 12:.1f}" font-size="12" fill="{color}">{svg_fmt_pct(value, 1)}</text>')
        body.append(polyline(points, color))
        legend_x = 110 + series.index((name, key, color)) * 190
        body.extend(
            [
                f'<line x1="{legend_x}" y1="510" x2="{legend_x + 32}" y2="510" stroke="{color}" stroke-width="4"/>',
                f'<text x="{legend_x + 42}" y="515" font-size="14" fill="#334155">{svg_text(name)}</text>',
            ]
        )
    for index, diag in enumerate(selected):
        x = linear_scale(index, 0, max(len(selected) - 1, 1), chart_x + 35, chart_x + chart_w - 35)
        body.append(f'<text x="{x - 28:.1f}" y="{chart_y + chart_h + 28}" font-size="13" fill="#334155">V{to_int(diag.get("vocab_size")) // 1000}k</text>')
    body.extend(
        [
            '<rect x="48" y="545" width="1004" height="50" rx="8" fill="#eef2ff" stroke="#c7d2fe"/>',
            '<text x="70" y="576" font-size="15" fill="#3730a3">해석: top1은 약 3%대라 한 token 독점은 아니다. 더 큰 문제는 20회 이하 tail type 증가와 긴 token row의 학습 부족 가능성이다.</text>',
        ]
    )
    write_svg(path, "\n".join(body))


def fmt_float(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    if number != number:
        return ""
    return f"{number:.{digits}f}"


def fmt_int(value: Any) -> str:
    return f"{to_int(value):,}"


def fmt_pct(value: Any, digits: int = 2) -> str:
    number = to_float(value)
    if number != number:
        return ""
    return f"{number * 100:.{digits}f}%"


def relative_to_report(path: Path, report_path: Path) -> str:
    return str(path.relative_to(report_path.parent))


def write_report(
    report_path: Path,
    figures: dict[str, Path],
    ready: list[dict[str, Any]],
    phases: list[str],
    meta: dict[str, Any],
    dataset_manifest: dict[str, Any],
    dataset_comparison: dict[str, Any],
    train_stats: dict[str, Any],
    val_stats: dict[str, Any],
    tokenizer_diags: list[dict[str, Any]],
) -> None:
    ranked = sorted(ready, key=lambda row: to_float(row["final_val_bits_per_char_median"]))
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# LLM 10x Screen-Ready 중간 보고서")
    lines.append("")
    lines.append(f"- generated_at_utc: `{generated_at}`")
    lines.append(f"- completed_physical_runs: `{meta.get('completed_physical_runs')}` / `{meta.get('planned_physical_runs')}`")
    lines.append(f"- pending_physical_runs: `{meta.get('pending_physical_runs')}`")
    lines.append(f"- screen_ready_conditions: `{meta.get('screen_ready_conditions')}`")
    lines.append(f"- claim_ready_conditions: `{meta.get('claim_ready_conditions')}`")
    lines.append(f"- result_ledger_status: `{meta.get('result_ledger_status')}`")
    lines.append(f"- all_results_jsonl: `{meta.get('all_results_jsonl')}`")
    lines.append("")
    lines.append("## 범위")
    lines.append("")
    lines.append("이 문서는 중간 선별용 보고서입니다. 실제 실행이 3회 이상 완료된 조건을 screen-ready로 보고 그래프를 만들었습니다. 아직 claim-ready 깊이에 도달한 조건은 없으므로, 아래 결과는 최종 통계 주장용이 아니라 다음 실험 우선순위를 정하기 위한 근거입니다.")
    lines.append("")
    lines.append("현재 최소 2개 이상의 screen-ready 조건이 있어 축 방향성을 볼 수 있는 항목은 다음과 같습니다.")
    for phase in phases:
        count = sum(1 for row in ready if row["phase"] == phase)
        lines.append(f"- {PHASE_LABELS.get(phase, phase)}: `{count}` screen-ready conditions")
    lines.append("")
    lines.append("## 기준 데이터셋과 규모")
    lines.append("")
    dataset_name = dataset_manifest.get("dataset_name", "obsidian_llm_10x")
    counts = dataset_manifest.get("counts", {})
    sources = dataset_manifest.get("sources", [])
    lines.append(
        f"이 보고서는 `{dataset_name}`를 기준으로 합니다. 하나의 LM 학습 데이터셋이지만, 원천은 `{len(sources)}`개 source 묶음에서 온 "
        f"`{fmt_int(counts.get('documents'))}`개 문서와 `{fmt_int(counts.get('chunks'))}`개 chunk입니다. "
        "`NSMC`는 이 screen-ready 실험의 학습 기준이 아니라, 별도 한글 stress test 후보로만 봐야 합니다."
    )
    llm_vs_nsmc = dataset_comparison.get("multipliers", {}).get("llm_10x_lm_file_vs_nsmc_chars")
    if llm_vs_nsmc is not None:
        lines.append(f"- LM 파일 문자 수는 기존 NSMC 파일 대비 약 `{fmt_float(llm_vs_nsmc, 2)}배`입니다.")
    lines.append("")
    lines.append("| split | 파일 | 문자 | 공백 제외 문자 | 공백 기준 단어 | 고유 공백 단어 | word-like 단위 | 고유 word-like | 한글/공백제외 | ASCII/공백제외 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for split, stats in (("train", train_stats), ("val", val_stats)):
        lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    f"`{stats.get('path')}`",
                    fmt_int(stats.get("chars")),
                    fmt_int(stats.get("non_ws")),
                    fmt_int(stats.get("whitespace_words")),
                    fmt_int(stats.get("unique_whitespace_words")),
                    fmt_int(stats.get("wordlike_units")),
                    fmt_int(stats.get("unique_wordlike_units")),
                    fmt_pct(stats.get("hangul_non_ws_ratio")),
                    fmt_pct(stats.get("ascii_alpha_non_ws_ratio")),
                ]
            )
            + " |"
        )
    lines.append("")
    paper_figure_specs = [
        (
            "paper_claim_map",
            "Figure 0. 결론 지도",
            "질문: 지금 이 실험에서 말할 수 있는 주장과 보류해야 하는 주장은 무엇인가?",
            "읽는 법: 왼쪽은 데이터셋 규모, 오른쪽은 현재 결론의 강도와 제한이다.",
            "핵심: 이 보고서는 최종 통계 결론이 아니라 screen-ready 중간 의사결정 문서다.",
        ),
        (
            "paper_ranking",
            "Figure 1. Screen-ready 조건 순위",
            "질문: 완료된 조건 중 validation bits/char가 가장 낮은 후보는 무엇인가?",
            "읽는 법: 막대가 길수록 validation bits/char가 낮다. 낮을수록 좋다.",
            "핵심: CTX0256, LR0300, FFN03, V10000이 현재 상위 후보지만 claim-ready 반복 수에는 아직 도달하지 않았다.",
        ),
        (
            "paper_dataset",
            "Figure 2. 데이터셋 문자 구성",
            "질문: 이 실험을 한글 중심 tokenizer 실험으로 해석해도 되는가?",
            "읽는 법: train/validation의 공백 제외 문자 구성을 비교한다.",
            "핵심: 한글은 약 6%대이고 ASCII가 약 75%이므로, 한글 전용 결론이 아니라 혼합 corpus 결론이다.",
        ),
        (
            "paper_vocab_tradeoff",
            "Figure 3. Vocab 압축 이득과 tail 비용",
            "질문: V6000 -> V10000 개선을 큰 vocab 권고로 바로 말해도 되는가?",
            "읽는 법: 왼쪽은 validation bits/char, 오른쪽은 train 20회 이하 vocab type이다.",
            "핵심: loss는 개선되지만 rare/tail type도 증가한다. 따라서 V10000은 유망 후보이지 최종 권고가 아니다.",
        ),
        (
            "paper_token_distribution",
            "Figure 4. Token 쏠림과 분포",
            "질문: vocab 증가 문제가 한 token 독점인지, tail token 증가인지 어떻게 구분하는가?",
            "읽는 법: top1/top10/top100/merge token 비중을 vocab별로 본다.",
            "핵심: top1 독점보다는 tail token과 긴 merge token의 학습 부족 가능성이 더 중요한 위험이다.",
        ),
    ]
    lines.append("## 발표/논문용 Figure 요약")
    lines.append("")
    lines.append("아래 figure들은 원시 dashboard가 아니라 발표 본문에 바로 쓸 수 있도록 `질문 -> 읽는 법 -> 핵심 해석` 순서로 구성했습니다.")
    lines.append("")
    for figure_key, title, question, how_to_read, takeaway in paper_figure_specs:
        figure_path = figures.get(figure_key)
        if not figure_path:
            continue
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"- {question}")
        lines.append(f"- {how_to_read}")
        lines.append(f"- {takeaway}")
        lines.append("")
        lines.append(f"![{title}]({relative_to_report(figure_path, report_path)})")
        lines.append("")
    lines.append("")
    lines.append("## 현재 상위 Screen-Ready 조건")
    lines.append("")
    lines.append("| 순위 | 조건 | 축 그룹 | 축 | 값 | n | 검증 bits/char 중앙값 | IQR | 중앙 실행 시간(분) | 중앙 warm tok/s |")
    lines.append("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for index, row in enumerate(ranked[:15], start=1):
        elapsed_min = to_float(row.get("elapsed_sec_median")) / 60.0
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    str(row["condition_id"]),
                    PHASE_LABELS.get(str(row["phase"]), str(row["phase"])),
                    str(row["sweep_axis"]),
                    str(row["axis_value"]),
                    str(row["completed_runs"]),
                    fmt_float(row.get("final_val_bits_per_char_median"), 5),
                    fmt_float(row.get("final_val_bits_per_char_iqr"), 5),
                    f"{elapsed_min:.2f}",
                    fmt_float(warm_tokens_per_sec(row), 0),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## 축별 보고")
    lines.append("")
    for phase in phases:
        best = phase_best_row(ready, phase)
        label = PHASE_LABELS.get(phase, phase)
        lines.append(f"### {label}")
        lines.append("")
        if best is not None:
            lines.append(
                f"- 현재 최상위 screen-ready 조건: `{best['condition_id']}` "
                f"({best['sweep_axis']}={best['axis_value']}), "
                f"검증 bits/char 중앙값 `{fmt_float(best.get('final_val_bits_per_char_median'), 5)}`."
            )
        phase_rows = sorted([row for row in ready if row["phase"] == phase], key=axis_value)
        if len(phase_rows) >= 2:
            first = phase_rows[0]
            last = phase_rows[-1]
            lines.append(
                f"- 현재 커버 범위: `{first['axis_value']}`부터 `{last['axis_value']}`까지, screen-ready 조건 `{len(phase_rows)}`개."
            )
        if phase == "phase3_vocab":
            lines.append("- 해석: vocab 축은 loss 그래프만 보면 V10000이 좋아 보이지만, tokenizer 진단에서는 tail type 증가가 동시에 나타납니다. 본문 Figure 3과 Figure 4를 기준으로 읽어야 합니다.")
        elif phase == "phase1_lr":
            lines.append("- 해석: learning rate는 현재 `2e-4`부터 `3e-4` 구간이 우선 후보입니다. 다만 다른 축과 상호작용할 수 있으므로 최종값은 상위 후보 재반복에서 확인해야 합니다.")
        elif phase == "phase6_context":
            lines.append("- 해석: 현재 `0.1 epoch` 짧은 예산에서는 짧은 context가 유리합니다. 이는 긴 context가 본질적으로 나쁘다는 뜻이 아니라, 제한된 step에서 학습 밀도가 더 높다는 신호일 수 있습니다.")
        elif phase == "phase11_ffn_mult":
            lines.append("- 해석: `ffn_mult=3`은 효율 후보입니다. 하지만 `ffn_mult=4` 기본값과 더 큰 조건의 screen-ready 반복이 필요합니다.")
        lines.append("")
    if tokenizer_diags:
        lines.append("## Vocab Size Tokenizer 진단")
        lines.append("")
        screen_vocab_values = sorted(
            {
                to_int(row.get("axis_value"))
                for row in ready
                if row.get("phase") == "phase3_vocab" and to_int(row.get("axis_value")) > 0
            }
        )
        if screen_vocab_values:
            lines.append(
                "아래 표는 screen-ready vocab 조건과 주변 cache를 직접 읽은 것입니다. "
                f"현재 3회 이상 학습 결과가 있는 vocab 조건은 `{', '.join(str(value) for value in screen_vocab_values)}`입니다. "
                "따라서 loss 그래프만으로 tokenizer 권고를 확정하지 않고, 실제 token 사용 분포를 함께 봅니다."
            )
        else:
            lines.append("아래 표는 사용 가능한 tokenizer cache를 직접 읽은 것입니다. 아직 screen-ready vocab 조건은 충분하지 않습니다.")
        lines.append("")
        lines.append("| vocab | merge rule | train token | val token | train 문자/token | val 문자/token | train 사용 type | train 미사용 type | train 20회 이하 type | merge 20회 이하 type | merge token 비중 | top1 | top10 | top100 | 한글 vocab type | 한글 token 비중 | merge 길이 p95/max |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for diag in sorted(tokenizer_diags, key=lambda row: to_int(row.get("vocab_size"))):
            lines.append(
                "| "
                + " | ".join(
                    [
                        fmt_int(diag.get("vocab_size")),
                        fmt_int(diag.get("bpe_merge_count")),
                        fmt_int(diag.get("train_tokens")),
                        fmt_int(diag.get("val_tokens")),
                        fmt_float(diag.get("train_chars_per_token"), 3),
                        fmt_float(diag.get("val_chars_per_token"), 3),
                        fmt_int(diag.get("used_train_types")),
                        fmt_int(diag.get("unused_train_types")),
                        fmt_int(diag.get("vocab_types_freq20")),
                        fmt_int(diag.get("merge_types_freq20")),
                        fmt_pct(diag.get("merge_train_mass")),
                        fmt_pct(diag.get("top1_mass")),
                        fmt_pct(diag.get("top10_mass")),
                        fmt_pct(diag.get("top100_mass")),
                        fmt_int(diag.get("hangul_vocab_types")),
                        fmt_pct(diag.get("hangul_train_mass")),
                        f"{fmt_float(diag.get('used_merge_len_p95'), 1)} / {fmt_int(diag.get('used_merge_len_max'))}",
                    ]
                )
                + " |"
            )
        lines.append("")
        diag_by_vocab = {to_int(diag.get("vocab_size")): diag for diag in tokenizer_diags}
        if 6000 in diag_by_vocab and 10000 in diag_by_vocab:
            low = diag_by_vocab[6000]
            high = diag_by_vocab[10000]
            token_reduction = 1.0 - to_float(high.get("train_tokens")) / to_float(low.get("train_tokens"))
            low_tail = to_int(low.get("vocab_types_freq20"))
            high_tail = to_int(high.get("vocab_types_freq20"))
            low_unused = to_int(low.get("unused_train_types"))
            high_unused = to_int(high.get("unused_train_types"))
            lines.append(
                f"- 압축 이득: `V6000 -> V10000`에서 train token은 `{fmt_int(low.get('train_tokens'))}`개에서 "
                f"`{fmt_int(high.get('train_tokens'))}`개로 줄어 약 `{fmt_pct(token_reduction)}` 감소합니다. "
                f"val 문자/token도 `{fmt_float(low.get('val_chars_per_token'), 3)}`에서 `{fmt_float(high.get('val_chars_per_token'), 3)}`로 늘어납니다."
            )
            lines.append(
                f"- tail 비용: 같은 구간에서 train 미사용 vocab type은 `{low_unused}`개에서 `{high_unused}`개로, "
                f"train 20회 이하 vocab type은 `{low_tail}`개에서 `{high_tail}`개로 늘어납니다. "
                "이는 embedding/LM head row 일부가 충분히 update되지 않을 수 있다는 신호입니다."
            )
            lines.append(
                f"- 쏠림 판단: top1 비중은 `{fmt_pct(low.get('top1_mass'))}`에서 `{fmt_pct(high.get('top1_mass'))}`, "
                f"top10 비중은 `{fmt_pct(low.get('top10_mass'))}`에서 `{fmt_pct(high.get('top10_mass'))}`로 조금 늘지만, "
                f"top100 비중은 `{fmt_pct(low.get('top100_mass'))}`에서 `{fmt_pct(high.get('top100_mass'))}`입니다. "
                "현재 수치만 보면 한 token이 전체를 독점한다기보다, 개행/경로 문자/숫자/구두점 같은 구조 token이 상위권을 차지합니다."
            )
        if 10000 in diag_by_vocab:
            chosen = diag_by_vocab[10000]
        else:
            chosen = max(tokenizer_diags, key=lambda row: to_int(row.get("vocab_size")))
        lines.append("")
        lines.append(f"### Top Token 예시: V{to_int(chosen.get('vocab_size'))}")
        lines.append("")
        lines.append("| rank | id | token | train 등장 | train 비중 |")
        lines.append("| ---: | ---: | --- | ---: | ---: |")
        for rank, token in enumerate(chosen.get("top_tokens", [])[:10], start=1):
            lines.append(
                f"| {rank} | {fmt_int(token.get('id'))} | `{token.get('decoded')}` | "
                f"{fmt_int(token.get('freq'))} | {fmt_pct(token.get('share'))} |"
            )
        lines.append("")
        lines.append("### 한글 Token 예시")
        lines.append("")
        lines.append("| rank | id | token | train 등장 | train 비중 |")
        lines.append("| ---: | ---: | --- | ---: | ---: |")
        for rank, token in enumerate(chosen.get("top_hangul", [])[:10], start=1):
            lines.append(
                f"| {rank} | {fmt_int(token.get('id'))} | `{token.get('decoded')}` | "
                f"{fmt_int(token.get('freq'))} | {fmt_pct(token.get('share'))} |"
            )
        lines.append("")
        lines.append(
            "한글 token 비중은 약 7%대로, train corpus의 한글/공백제외 문자 비중과 비슷합니다. "
            "즉 이 실험은 한글 중심 실험이 아니라 영어/코드/LLM 자료 중심의 혼합 corpus 실험입니다. "
            "한글 인식 BPE 권고는 이 표에 더해 한글 전용 validation 또는 Obsidian 한글 subset 결과를 별도로 붙여야 합니다."
        )
        lines.append("")
    lines.append("## 중간 해석")
    lines.append("")
    lines.append("- Learning rate: 현재 screen-ready 기준 sweet spot은 `2e-4`부터 `3e-4` 근처이며, 지금까지는 `3e-4`가 가장 좋습니다.")
    lines.append("- Context length: 현재 `0.1 epoch` 예산에서는 짧은 context가 앞서며, 특히 `CTX0256`이 강합니다. 다만 이는 예산 민감 신호로 봐야 하고, 긴 context 자체가 나쁘다는 최종 결론은 아닙니다.")
    lines.append("- Vocab size: `6000 -> 8000 -> 10000`은 검증 bits/char만 보면 개선되지만, 동시에 미사용 type과 20회 이하 tail type도 증가합니다. 따라서 현재 결론은 `V10000 유망`이지 `큰 vocab일수록 좋음`이 아닙니다.")
    lines.append("- FFN multiplier: `ffn_mult=3`이 `2`보다 약간 앞서며 효율 후보로 좋습니다. 다만 기본값과 더 큰 FFN 조건은 아직 실행이 필요합니다.")
    lines.append("")
    lines.append("## 다음 의사결정")
    lines.append("")
    lines.append("큐는 백그라운드에서 계속 실행하면 됩니다. 더 빠른 실행 가능한 결론이 목표라면, 모든 탐색 조건 완료를 기다리기보다 현재 상위 후보를 n=10으로 보강하는 쪽이 좋습니다.")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_rows = read_summary(args.summary_csv)
    meta = read_json(args.meta_json)
    ledger_rows = read_ledger(args.all_results_jsonl)
    ready = ready_rows(summary_rows)
    phases = meaningful_phases(ready)
    scoped_ready = [row for row in ready if row["phase"] in phases]
    raw_by_condition = raw_results_by_condition(ledger_rows)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    figures: dict[str, Path] = {}
    figures["ranking"] = args.figures_dir / "screen_ready_quality_rank.png"
    figures["speed_quality"] = args.figures_dir / "screen_ready_speed_quality.png"
    if PLOT_AVAILABLE:
        plot_ranking(scoped_ready, figures["ranking"])
        plot_speed_quality(scoped_ready, figures["speed_quality"])

        for phase in phases:
            figures[f"{phase}_quality"] = args.figures_dir / f"{phase}_quality.png"
            plot_phase_metric(
                phase,
                scoped_ready,
                raw_by_condition,
                "final_val_bits_per_char",
                "final_val_bits_per_char_median",
                "final_val_bits_per_char_iqr",
                figures[f"{phase}_quality"],
                "Validation bits/char (lower is better)",
                "validation quality",
            )
            figures[f"{phase}_gap"] = args.figures_dir / f"{phase}_generalization_gap.png"
            plot_phase_metric(
                phase,
                scoped_ready,
                raw_by_condition,
                "final_generalization_gap",
                "final_generalization_gap_median",
                "final_generalization_gap_iqr",
                figures[f"{phase}_gap"],
                "Final generalization gap",
                "generalization gap",
                zero_line=True,
            )
            figures[f"{phase}_speed"] = args.figures_dir / f"{phase}_speed.png"
            plot_phase_speed(phase, scoped_ready, figures[f"{phase}_speed"])
    else:
        for phase in phases:
            for suffix, filename in (
                ("quality", f"{phase}_quality.png"),
                ("gap", f"{phase}_generalization_gap.png"),
                ("speed", f"{phase}_speed.png"),
            ):
                path = args.figures_dir / filename
                if path.exists():
                    figures[f"{phase}_{suffix}"] = path

    dataset_manifest = read_optional_json(args.dataset_manifest)
    dataset_comparison = read_optional_json(args.dataset_comparison)
    train_stats = text_corpus_stats(args.train_text)
    val_stats = text_corpus_stats(args.val_text)
    tokenizer_diags = [
        diag
        for vocab_size in available_tokenizer_vocab_sizes(args.cache_dir)
        if (diag := tokenizer_diagnostic(args.cache_dir, vocab_size)) is not None
    ]
    figures.update(paper_figures(args.figures_dir, scoped_ready, tokenizer_diags, train_stats, val_stats, dataset_manifest))

    write_report(
        args.report,
        figures,
        scoped_ready,
        phases,
        meta,
        dataset_manifest,
        dataset_comparison,
        train_stats,
        val_stats,
        tokenizer_diags,
    )
    print(
        {
            "report": str(args.report),
            "figures_dir": str(args.figures_dir),
            "figure_count": len(figures),
            "meaningful_phases": phases,
            "screen_ready_conditions": len(scoped_ready),
        }
    )


if __name__ == "__main__":
    main()
