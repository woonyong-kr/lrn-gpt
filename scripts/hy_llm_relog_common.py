# -*- coding: utf-8 -*-
"""Utilities for relogging REPORT.md HY experiments with LLM-style metrics."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REPORT_MD = ROOT / "REPORT.md"
HY_RESULT_DIR = ROOT / "docs" / "HY" / "testresult"
RELOG_DIR = ROOT / "docs" / "HY" / "llm_relog"
RUNS_DIR = RELOG_DIR / "runs"
FIGURE_DIR = ROOT / "docs" / "HY" / "figures_llm_seq"
LOG2_E = math.log(2)

GROUPS: list[dict[str, Any]] = [
    {"section": "5 epoch baseline", "group": "baseline_5ep", "axis": "baseline", "baseline": "E00", "ids": ["E00"]},
    {"section": "context_length", "group": "context_length", "axis": "context_length", "baseline": "E00", "ids": ["E01", "E00", "E02", "E03"]},
    {"section": "vocab_size", "group": "vocab_size", "axis": "vocab_size", "baseline": "E00", "ids": ["E04", "E00", "E05", "E06"]},
    {"section": "emb_dim", "group": "emb_dim", "axis": "emb_dim", "baseline": "E00", "ids": ["E07", "E00", "E08"]},
    {"section": "n_heads 5ep", "group": "n_heads_5ep", "axis": "n_heads", "baseline": "E00", "ids": ["E09", "E00", "E10"]},
    {"section": "n_layers 5ep", "group": "n_layers_5ep", "axis": "n_layers", "baseline": "E00", "ids": ["E11", "E00", "E12", "E13"]},
    {"section": "ffn_multiplier 5ep", "group": "ffn_multiplier_5ep", "axis": "ffn_mult", "baseline": "E00", "ids": ["E14", "E00", "E15"]},
    {"section": "drop_rate 5ep", "group": "dropout_5ep", "axis": "drop_rate", "baseline": "E00", "ids": ["E16", "E17", "E00", "E18"]},
    {"section": "qkv_bias 5ep", "group": "qkv_bias_5ep", "axis": "qkv_bias", "baseline": "E00", "ids": ["E00", "E19"]},
    {"section": "weight_tying 5ep", "group": "weight_tying_5ep", "axis": "tie_embeddings", "baseline": "E00", "ids": ["E00", "E20"]},
    {"section": "10 epoch baseline", "group": "baseline_10ep", "axis": "baseline", "baseline": "E28", "ids": ["E28"]},
    {"section": "activation 10ep", "group": "activation_10ep", "axis": "activation", "baseline": "E28", "ids": ["E25", "E28", "E26"]},
    {"section": "norm_first 10ep", "group": "norm_first_10ep", "axis": "norm_first", "baseline": "E28", "ids": ["E28", "E27"]},
    {"section": "stride 10ep", "group": "stride_10ep", "axis": "stride", "baseline": "E28", "ids": ["E28", "E29"]},
    {"section": "n_heads 10ep", "group": "n_heads_10ep", "axis": "n_heads", "baseline": "E28", "ids": ["E31", "E32", "E28", "E33", "E34"]},
    {"section": "ffn_multiplier 10ep", "group": "ffn_multiplier_10ep", "axis": "ffn_mult", "baseline": "E28", "ids": ["E35", "E28", "E36"]},
    {"section": "depth/norm 10ep", "group": "depth_norm_10ep", "axis": "n_layers_norm_first", "baseline": "E28", "ids": ["E28", "E37", "E38"]},
    {"section": "qkv_bias 10ep", "group": "qkv_bias_10ep", "axis": "qkv_bias", "baseline": "E28", "ids": ["E28", "E39", "E37", "E40"]},
    {"section": "weight_tying 10/20/50ep", "group": "weight_tying_long", "axis": "tie_embeddings_epochs", "baseline": "E28", "ids": ["E28", "E41", "E42", "E43", "E44"]},
    {"section": "long dropout 20ep", "group": "dropout_long", "axis": "drop_rate", "baseline": "E23", "ids": ["E21", "E22", "E23", "E24"]},
    {"section": "depth/norm 20ep", "group": "depth_norm_20ep", "axis": "n_layers_norm_first", "baseline": "E23", "ids": ["E23", "E45", "E46", "E47", "E48"]},
    {"section": "activation seed repeat", "group": "activation_seed", "axis": "activation_seed", "baseline": "E52", "ids": ["E49", "E50", "E51", "E52", "E53", "E54", "E55", "E56", "E57"]},
    {"section": "regularized depth/norm", "group": "regularized_depth_norm", "axis": "n_layers_norm_first", "baseline": "E60", "ids": ["E60", "E61", "E62", "E63"]},
]

TARGET_IDS = sorted({experiment_id for group in GROUPS for experiment_id in group["ids"]}, key=lambda value: int(value[1:]))

MANIFEST_FIELDS = [
    "experiment_id",
    "source_md",
    "report_section",
    "comparison_group",
    "baseline_id",
    "changed_variable",
    "changed_value",
    "seed",
    "vocab_size",
    "context_length",
    "emb_dim",
    "n_heads",
    "n_layers",
    "ffn_mult",
    "activation",
    "drop_rate",
    "norm_first",
    "qkv_bias",
    "weight_tying",
    "stride",
    "num_epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "rerun_required",
    "rerun_reason",
]

METRIC_FIELDS = [
    "experiment_id",
    "title",
    "source_md",
    "comparison_group",
    "changed_variable",
    "changed_value",
    "seed",
    "vocab_size",
    "context_length",
    "emb_dim",
    "n_heads",
    "n_layers",
    "ffn_mult",
    "activation",
    "drop_rate",
    "norm_first",
    "qkv_bias",
    "weight_tying",
    "stride",
    "num_epochs",
    "parameter_count",
    "actual_vocab_size",
    "bpe_merge_count",
    "train_char_count",
    "val_char_count",
    "train_token_count",
    "val_token_count",
    "train_tokens_per_char",
    "val_tokens_per_char",
    "train_chars_per_token",
    "val_chars_per_token",
    "best_epoch",
    "best_step",
    "best_tokens_seen",
    "best_val_loss",
    "best_val_bits_per_char",
    "final_epoch",
    "final_step",
    "final_tokens_seen",
    "final_train_loss",
    "final_val_loss",
    "final_train_bits_per_char",
    "final_val_bits_per_char",
    "final_minus_best_val_loss",
    "final_minus_best_val_bits_per_char",
    "final_generalization_gap",
    "gap_slope_last_quarter",
    "overfit_score",
    "tokens_per_parameter",
    "tokens_per_sec_after_warmup",
    "compute_proxy_param_tokens",
    "estimated_train_flops",
]

STEP_FIELDS = [
    "experiment_id",
    "epoch",
    "step",
    "tokens_seen",
    "estimated_chars_seen",
    "train_loss",
    "val_loss",
    "test_loss",
    "train_nats_per_char",
    "val_nats_per_char",
    "train_bits_per_char",
    "val_bits_per_char",
    "train_val_gap",
    "best_val_loss_so_far",
    "best_val_bits_per_char_so_far",
    "val_loss_minus_best_so_far",
    "val_bits_minus_best_so_far",
    "lr",
    "grad_norm",
    "elapsed_sec",
    "tokens_per_sec",
    "tokens_per_sec_after_warmup",
    "parameter_count",
    "compute_proxy_param_tokens",
    "estimated_train_flops",
]


@dataclass
class ExperimentRecord:
    experiment_id: str
    title: str
    source_path: Path
    metrics: dict[str, Any]
    series: list[dict[str, Any]]


def load_records(result_dir: Path = HY_RESULT_DIR) -> dict[str, ExperimentRecord]:
    if not result_dir.is_absolute():
        result_dir = ROOT / result_dir
    candidates: dict[str, list[ExperimentRecord]] = {}
    for path in sorted(result_dir.glob("E*_result.md")):
        record = parse_hy_markdown(path)
        candidates.setdefault(record.experiment_id, []).append(record)
    return {experiment_id: choose_record(records) for experiment_id, records in candidates.items()}


def choose_record(records: list[ExperimentRecord]) -> ExperimentRecord:
    def priority(record: ExperimentRecord) -> tuple[int, int, int]:
        simple_name = f"{record.experiment_id}_result.md"
        descriptive = 0 if record.source_path.name == simple_name else 1
        has_series = 1 if len(record.series) >= 2 else 0
        has_chars = 1 if number(record.metrics, "train_char_count") and number(record.metrics, "val_char_count") else 0
        return (descriptive, has_series + has_chars, len(record.source_path.name))

    return max(records, key=priority)


def parse_hy_markdown(path: Path) -> ExperimentRecord:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"(E\d+)", path.name)
    if not match:
        raise ValueError(f"cannot find experiment id in {path}")
    experiment_id = match.group(1)
    metrics = parse_two_column_tables(text)
    metrics["experiment_id"] = experiment_id
    metrics["source_md"] = str(path.relative_to(ROOT))
    title = first_heading(text) or path.stem
    metrics["title"] = title
    series = parse_csv_blocks(text)
    normalize_metrics(metrics, series)
    return ExperimentRecord(experiment_id=experiment_id, title=title, source_path=path, metrics=metrics, series=series)


def first_heading(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def parse_two_column_tables(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in text.splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 2:
            continue
        key, value = cells
        if key in {"항목", "지표"} or value == "값":
            continue
        data[normalize_key(key)] = parse_value(value)
    return data


def parse_csv_blocks(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in re.finditer(r"```csv\n(.*?)```", text, flags=re.DOTALL):
        block = match.group(1).strip()
        if not block:
            continue
        reader = csv.DictReader(block.splitlines())
        for row in reader:
            rows.append({normalize_key(key): parse_value(value) for key, value in row.items()})
    return rows


def normalize_key(key: str | None) -> str:
    if key is None:
        return ""
    normalized = key.strip().strip("`").lower()
    normalized = normalized.replace(" ", "_").replace("-", "_")
    normalized = normalized.replace("final_test_loss", "final_test_loss")
    return normalized


def parse_value(value: Any) -> Any:
    if value is None:
        return ""
    if not isinstance(value, str):
        return value
    cleaned = value.strip().strip("`")
    if cleaned == "":
        return ""
    if cleaned in {"True", "true"}:
        return True
    if cleaned in {"False", "false"}:
        return False
    if cleaned.lower() in {"none", "null"}:
        return ""
    numeric = cleaned.replace(",", "")
    if re.fullmatch(r"[-+]?\d+", numeric):
        try:
            return int(numeric)
        except ValueError:
            return cleaned
    if re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)(e[-+]?\d+)?", numeric, flags=re.IGNORECASE):
        try:
            return float(numeric)
        except ValueError:
            return cleaned
    return cleaned


def normalize_metrics(metrics: dict[str, Any], series: list[dict[str, Any]]) -> None:
    aliases = {
        "train_chars": "train_char_count",
        "val_chars": "val_char_count",
        "train_tokens": "train_token_count",
        "val_tokens": "val_token_count",
        "num_epochs": "num_epochs",
        "epochs": "num_epochs",
        "activation_name": "activation",
        "weight_tying": "weight_tying",
        "tie_embeddings": "weight_tying",
        "lr": "lr",
        "learning_rate": "lr",
    }
    for old, new in aliases.items():
        if old in metrics and new not in metrics:
            metrics[new] = metrics[old]
    if "activation" not in metrics and "activation_name" in metrics:
        metrics["activation"] = metrics["activation_name"]
    if "ffn_multiplier" in metrics and "ffn_mult" not in metrics:
        metrics["ffn_mult"] = metrics["ffn_multiplier"]
    if "final_loss_gap" in metrics and "final_generalization_gap" not in metrics:
        metrics["final_generalization_gap"] = metrics["final_loss_gap"]
    if "final_val_loss" not in metrics and series:
        metrics["final_val_loss"] = series[-1].get("val_loss", "")
    if "final_train_loss" not in metrics and series:
        metrics["final_train_loss"] = series[-1].get("train_loss", "")
    if "best_val_loss" not in metrics and series:
        metrics["best_val_loss"] = min(number(row, "val_loss", float("inf")) for row in series)


def build_manifest(records: dict[str, ExperimentRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for group in GROUPS:
        for experiment_id in group["ids"]:
            key = (group["group"], experiment_id)
            if key in seen:
                continue
            seen.add(key)
            record = records.get(experiment_id)
            metrics = record.metrics if record else {}
            changed_value = changed_value_for_axis(metrics, group["axis"])
            rerun_required, rerun_reason = rerun_status(record)
            row = {
                "experiment_id": experiment_id,
                "source_md": "" if record is None else str(record.source_path.relative_to(ROOT)),
                "report_section": group["section"],
                "comparison_group": group["group"],
                "baseline_id": group["baseline"],
                "changed_variable": group["axis"],
                "changed_value": changed_value,
                "seed": metrics.get("seed", ""),
                "vocab_size": metrics.get("vocab_size", ""),
                "context_length": metrics.get("context_length", ""),
                "emb_dim": metrics.get("emb_dim", ""),
                "n_heads": metrics.get("n_heads", ""),
                "n_layers": metrics.get("n_layers", ""),
                "ffn_mult": metrics.get("ffn_mult", ""),
                "activation": metrics.get("activation", ""),
                "drop_rate": metrics.get("drop_rate", ""),
                "norm_first": metrics.get("norm_first", ""),
                "qkv_bias": metrics.get("qkv_bias", ""),
                "weight_tying": metrics.get("weight_tying", ""),
                "stride": metrics.get("stride", ""),
                "num_epochs": metrics.get("num_epochs", ""),
                "batch_size": metrics.get("batch_size", ""),
                "lr": metrics.get("lr", ""),
                "weight_decay": metrics.get("weight_decay", ""),
                "rerun_required": rerun_required,
                "rerun_reason": rerun_reason,
            }
            rows.append(row)
    return rows


def changed_value_for_axis(metrics: dict[str, Any], axis: str) -> Any:
    if axis == "baseline":
        return "baseline"
    if axis == "n_layers_norm_first":
        return f"layers={metrics.get('n_layers', '')},norm_first={metrics.get('norm_first', '')}"
    if axis == "tie_embeddings_epochs":
        return f"weight_tying={metrics.get('weight_tying', '')},epochs={metrics.get('num_epochs', '')}"
    if axis == "activation_seed":
        return f"activation={metrics.get('activation', '')},seed={metrics.get('seed', '')}"
    return metrics.get(axis, "")


def rerun_status(record: ExperimentRecord | None) -> tuple[bool, str]:
    if record is None:
        return True, "source markdown not found"
    required = ["train_char_count", "val_char_count", "train_token_count", "val_token_count", "parameter_count"]
    missing = [key for key in required if number(record.metrics, key) <= 0]
    if missing:
        return True, "missing metrics: " + ",".join(missing)
    if len(record.series) < 2:
        return True, "missing step-level csv history"
    if not any(number(row, "tokens_seen") > 0 for row in record.series):
        return True, "missing tokens_seen in history"
    return False, ""


def write_manifest(rows: list[dict[str, Any]], path: Path = RELOG_DIR / "experiment_manifest.csv") -> None:
    write_csv(rows, path, MANIFEST_FIELDS)


def relog_records(records: dict[str, ExperimentRecord], manifest_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    row_by_id = {row["experiment_id"]: row for row in manifest_rows}
    for experiment_id in TARGET_IDS:
        record = records[experiment_id]
        manifest_row = row_by_id.get(experiment_id, {})
        run_dir = RUNS_DIR / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        history = build_history(record)
        metrics = build_run_metrics(record, history)
        metrics.update(
            {
                "comparison_group": manifest_row.get("comparison_group", ""),
                "changed_variable": manifest_row.get("changed_variable", ""),
                "changed_value": manifest_row.get("changed_value", ""),
            }
        )
        write_json(run_dir / "config.json", build_config(record))
        write_json(run_dir / "tokenizer_profile.json", build_tokenizer_profile(record))
        write_json(run_dir / "metrics.json", metrics)
        write_jsonl(run_dir / "history.jsonl", history)
        write_jsonl(run_dir / "samples.jsonl", build_samples(record))
        (run_dir / "result.md").write_text(render_run_result(record, metrics), encoding="utf-8")
        all_metrics.append(metrics)
        all_history.extend(history)
    return all_metrics, all_history


def build_config(record: ExperimentRecord) -> dict[str, Any]:
    metrics = record.metrics
    keys = [
        "experiment_id",
        "title",
        "source_md",
        "seed",
        "vocab_size",
        "context_length",
        "emb_dim",
        "n_heads",
        "n_layers",
        "ffn_mult",
        "activation",
        "drop_rate",
        "qkv_bias",
        "weight_tying",
        "norm_first",
        "batch_size",
        "num_epochs",
        "lr",
        "weight_decay",
        "eval_freq",
        "eval_iter",
        "stride",
        "amp",
        "device",
        "torch",
        "parameter_count",
    ]
    return {key: metrics.get(key, "") for key in keys}


def build_tokenizer_profile(record: ExperimentRecord) -> dict[str, Any]:
    metrics = record.metrics
    vocab_size = int(number(metrics, "vocab_size"))
    train_chars = number(metrics, "train_char_count")
    val_chars = number(metrics, "val_char_count")
    train_tokens = number(metrics, "train_token_count")
    val_tokens = number(metrics, "val_token_count")
    actual_vocab_size = int(number(metrics, "actual_vocab_size", vocab_size))
    estimated_merge_count = max(0, actual_vocab_size - 260)
    return {
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "bpe_merge_count": int(number(metrics, "bpe_merge_count", estimated_merge_count)),
        "special_token_count": 4,
        "byte_token_count": 256,
        "train_chars": train_chars,
        "val_chars": val_chars,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_tokens_per_char": safe_ratio(train_tokens, train_chars),
        "val_tokens_per_char": safe_ratio(val_tokens, val_chars),
        "train_chars_per_token": safe_ratio(train_chars, train_tokens),
        "val_chars_per_token": safe_ratio(val_chars, val_tokens),
        "token_length_histogram": {},
        "top_50_merge_rules": [],
        "byte_fallback_ratio": None,
        "profile_note": "Tokenizer JSON was not present in this repository; size/count fields are reconstructed from HY result logs.",
    }


def build_history(record: ExperimentRecord) -> list[dict[str, Any]]:
    metrics = record.metrics
    train_tpc = safe_ratio(number(metrics, "train_token_count"), number(metrics, "train_char_count"))
    val_tpc = safe_ratio(number(metrics, "val_token_count"), number(metrics, "val_char_count"))
    train_cpt = safe_ratio(number(metrics, "train_char_count"), number(metrics, "train_token_count"))
    parameter_count = number(metrics, "parameter_count")
    best_val_loss = float("inf")
    best_val_bits = float("inf")
    rows: list[dict[str, Any]] = []
    for raw in sorted(record.series, key=lambda row: (number(row, "tokens_seen"), number(row, "step"))):
        train_loss = number(raw, "train_loss")
        val_loss = number(raw, "val_loss")
        test_loss = number(raw, "test_loss", val_loss)
        tokens_seen = number(raw, "tokens_seen")
        train_nats = train_loss * train_tpc
        val_nats = val_loss * val_tpc
        train_bits = safe_ratio(train_nats, LOG2_E)
        val_bits = safe_ratio(val_nats, LOG2_E)
        best_val_loss = min(best_val_loss, val_loss)
        best_val_bits = min(best_val_bits, val_bits)
        gap = number(raw, "loss_gap", val_loss - train_loss)
        row = {
            "experiment_id": record.experiment_id,
            "epoch": number(raw, "epoch"),
            "step": number(raw, "step"),
            "tokens_seen": tokens_seen,
            "estimated_chars_seen": tokens_seen * train_cpt,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "train_nats_per_char": train_nats,
            "val_nats_per_char": val_nats,
            "train_bits_per_char": train_bits,
            "val_bits_per_char": val_bits,
            "train_val_gap": gap,
            "best_val_loss_so_far": best_val_loss,
            "best_val_bits_per_char_so_far": best_val_bits,
            "val_loss_minus_best_so_far": val_loss - best_val_loss,
            "val_bits_minus_best_so_far": val_bits - best_val_bits,
            "lr": raw.get("lr", metrics.get("lr", "")),
            "grad_norm": raw.get("grad_norm", ""),
            "elapsed_sec": number(raw, "elapsed_sec"),
            "tokens_per_sec": number(raw, "tokens_per_sec"),
            "tokens_per_sec_after_warmup": number(raw, "tokens_per_sec"),
            "parameter_count": parameter_count,
            "compute_proxy_param_tokens": parameter_count * tokens_seen,
            "estimated_train_flops": 6 * parameter_count * tokens_seen,
        }
        rows.append(row)
    return rows


def build_run_metrics(record: ExperimentRecord, history: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = record.metrics
    final = max(history, key=lambda row: (number(row, "tokens_seen"), number(row, "step")))
    best = min(history, key=lambda row: number(row, "val_loss", float("inf")))
    train_tpc = safe_ratio(number(metrics, "train_token_count"), number(metrics, "train_char_count"))
    val_tpc = safe_ratio(number(metrics, "val_token_count"), number(metrics, "val_char_count"))
    train_cpt = safe_ratio(number(metrics, "train_char_count"), number(metrics, "train_token_count"))
    val_cpt = safe_ratio(number(metrics, "val_char_count"), number(metrics, "val_token_count"))
    final_minus_best_loss = number(final, "val_loss") - number(best, "val_loss")
    final_minus_best_bits = number(final, "val_bits_per_char") - number(best, "val_bits_per_char")
    gap_slope = slope_last_quarter(history, "train_val_gap")
    final_gap = number(final, "train_val_gap")
    return {
        "experiment_id": record.experiment_id,
        "title": record.title,
        "source_md": str(record.source_path.relative_to(ROOT)),
        "seed": metrics.get("seed", ""),
        "vocab_size": metrics.get("vocab_size", ""),
        "context_length": metrics.get("context_length", ""),
        "emb_dim": metrics.get("emb_dim", ""),
        "n_heads": metrics.get("n_heads", ""),
        "n_layers": metrics.get("n_layers", ""),
        "ffn_mult": metrics.get("ffn_mult", ""),
        "activation": metrics.get("activation", ""),
        "drop_rate": metrics.get("drop_rate", ""),
        "norm_first": metrics.get("norm_first", ""),
        "qkv_bias": metrics.get("qkv_bias", ""),
        "weight_tying": metrics.get("weight_tying", ""),
        "stride": metrics.get("stride", ""),
        "num_epochs": metrics.get("num_epochs", ""),
        "parameter_count": number(metrics, "parameter_count"),
        "actual_vocab_size": int(number(metrics, "actual_vocab_size", number(metrics, "vocab_size"))),
        "bpe_merge_count": int(max(0, number(metrics, "actual_vocab_size", number(metrics, "vocab_size")) - 260)),
        "train_char_count": number(metrics, "train_char_count"),
        "val_char_count": number(metrics, "val_char_count"),
        "train_token_count": number(metrics, "train_token_count"),
        "val_token_count": number(metrics, "val_token_count"),
        "train_tokens_per_char": train_tpc,
        "val_tokens_per_char": val_tpc,
        "train_chars_per_token": train_cpt,
        "val_chars_per_token": val_cpt,
        "best_epoch": number(best, "epoch"),
        "best_step": number(best, "step"),
        "best_tokens_seen": number(best, "tokens_seen"),
        "best_val_loss": number(best, "val_loss"),
        "best_val_bits_per_char": number(best, "val_bits_per_char"),
        "final_epoch": number(final, "epoch"),
        "final_step": number(final, "step"),
        "final_tokens_seen": number(final, "tokens_seen"),
        "final_train_loss": number(final, "train_loss"),
        "final_val_loss": number(final, "val_loss"),
        "final_train_bits_per_char": number(final, "train_bits_per_char"),
        "final_val_bits_per_char": number(final, "val_bits_per_char"),
        "final_minus_best_val_loss": final_minus_best_loss,
        "final_minus_best_val_bits_per_char": final_minus_best_bits,
        "final_generalization_gap": final_gap,
        "gap_slope_last_quarter": gap_slope,
        "overfit_score": max(0.0, final_gap) + max(0.0, final_minus_best_loss) + max(0.0, gap_slope * 1_000_000),
        "tokens_per_parameter": safe_ratio(number(final, "tokens_seen"), number(metrics, "parameter_count")),
        "tokens_per_sec_after_warmup": number(final, "tokens_per_sec_after_warmup"),
        "compute_proxy_param_tokens": number(final, "compute_proxy_param_tokens"),
        "estimated_train_flops": number(final, "estimated_train_flops"),
    }


def build_samples(record: ExperimentRecord) -> list[dict[str, Any]]:
    text = record.source_path.read_text(encoding="utf-8")
    prompt_match = re.search(r"prompt:\s*`([^`]+)`", text)
    prompt = prompt_match.group(1) if prompt_match else "이 영화는"
    sample = ""
    if "## 5. 생성 샘플" in text:
        after = text.split("## 5. 생성 샘플", 1)[1]
        block = re.search(r"```text\n(.*?)```", after, flags=re.DOTALL)
        if block:
            sample = block.group(1).strip()
    return [{"prompt": prompt, "sample": sample, "source_md": str(record.source_path.relative_to(ROOT))}]


def render_run_result(record: ExperimentRecord, metrics: dict[str, Any]) -> str:
    return f"""# {record.experiment_id} LLM relog result

source: `{record.source_path.relative_to(ROOT)}`

| metric | value |
| --- | ---: |
| final_val_loss | {format_float(metrics.get("final_val_loss"))} |
| final_val_bits_per_char | {format_float(metrics.get("final_val_bits_per_char"))} |
| best_val_loss | {format_float(metrics.get("best_val_loss"))} |
| best_val_bits_per_char | {format_float(metrics.get("best_val_bits_per_char"))} |
| final_minus_best_val_loss | {format_float(metrics.get("final_minus_best_val_loss"))} |
| final_generalization_gap | {format_float(metrics.get("final_generalization_gap"))} |
| final_tokens_seen | {format_float(metrics.get("final_tokens_seen"))} |
| compute_proxy_param_tokens | {format_float(metrics.get("compute_proxy_param_tokens"))} |

This file is generated from the existing HY step CSV and normalized to the LLM metric schema.
"""


def aggregate_outputs(metrics: list[dict[str, Any]], history: list[dict[str, Any]]) -> None:
    write_csv(metrics, RELOG_DIR / "report_llm_metrics.csv", METRIC_FIELDS)
    write_csv(history, RELOG_DIR / "report_llm_metrics_by_step.csv", STEP_FIELDS)
    write_csv(build_summary(metrics), RELOG_DIR / "report_llm_summary_by_condition.csv")
    write_json(RELOG_DIR / "aggregate_meta.json", {"run_count": len(metrics), "history_rows": len(history), "target_ids": TARGET_IDS})


def build_summary(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in metrics:
        groups.setdefault(str(row.get("comparison_group", "")), []).append(row)
    summary: list[dict[str, Any]] = []
    for group, rows in sorted(groups.items()):
        vals = [number(row, "final_val_bits_per_char") for row in rows if number(row, "final_val_bits_per_char") > 0]
        gaps = [number(row, "final_generalization_gap") for row in rows]
        summary.append(
            {
                "comparison_group": group,
                "run_count": len(rows),
                "best_experiment_id": min(rows, key=lambda row: number(row, "final_val_bits_per_char", float("inf"))).get("experiment_id"),
                "mean_final_val_bits_per_char": mean(vals) if vals else 0.0,
                "std_final_val_bits_per_char": pstdev(vals) if len(vals) > 1 else 0.0,
                "mean_final_generalization_gap": mean(gaps) if gaps else 0.0,
            }
        )
    return summary


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as file:
        return [{key: parse_value(value) for key, value in row.items()} for row in csv.DictReader(file)]


def write_csv(rows: list[dict[str, Any]], path: Path, fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def number(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    return to_float(mapping.get(key), default)


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        try:
            if math.isnan(float(value)):
                return default
        except ValueError:
            return default
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return default


def safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def slope_last_quarter(history: list[dict[str, Any]], field: str) -> float:
    if len(history) < 4:
        return 0.0
    ordered = sorted(history, key=lambda row: number(row, "tokens_seen"))
    subset = ordered[max(0, int(len(ordered) * 0.75)) :]
    if len(subset) < 2:
        return 0.0
    x = [number(row, "tokens_seen") for row in subset]
    y = [number(row, field) for row in subset]
    x_mean = mean(x)
    y_mean = mean(y)
    denom = sum((value - x_mean) ** 2 for value in x)
    if denom == 0:
        return 0.0
    return sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x, y)) / denom


def format_float(value: Any, digits: int = 6) -> str:
    if value == "" or value is None:
        return ""
    numeric = to_float(value, float("nan"))
    if math.isnan(numeric):
        return str(value)
    if abs(numeric) >= 1_000_000:
        return f"{numeric:.3e}"
    return f"{numeric:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], fields: list[str], max_rows: int | None = None) -> str:
    selected = rows[: max_rows or len(rows)]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in selected:
        cells = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                cells.append(format_float(value, 4))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
