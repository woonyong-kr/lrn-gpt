#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate LLM 10x repeated-run results into statistical reports."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from collections import Counter
import json
import math
from pathlib import Path
import random
from statistics import mean, median, stdev
import time
from typing import Any

from llm_10x_common import DOCS_DIR, MATRIX_PATH, RUNS_DIR, parse_int, read_json, read_matrix, write_json, write_jsonl


DEFAULT_SUMMARY_CSV = DOCS_DIR / "aggregate_summary.csv"
DEFAULT_REPORT = DOCS_DIR / "aggregate_report.md"
DEFAULT_META = DOCS_DIR / "aggregate_meta.json"
DEFAULT_ALL_RESULTS_JSONL = DOCS_DIR / "all_run_results.jsonl"
METRICS = (
    "final_val_bits_per_char",
    "final_val_nats_per_char",
    "final_val_loss",
    "best_val_bits_per_char",
    "best_val_nats_per_char",
    "best_val_loss",
    "final_minus_best_val_loss",
    "final_generalization_gap",
    "generalization_gap_delta",
    "train_val_improvement_gap",
    "overfit_score",
    "tokens_seen",
    "estimated_chars_seen",
    "parameter_count",
    "tokens_per_param",
    "compute_proxy",
    "estimated_train_flops",
    "elapsed_sec",
    "seconds_per_epoch",
    "tokens_per_sec",
    "tokens_per_sec_after_warmup",
)
LN2 = math.log(2.0)
TRAINING_FLOPS_FACTOR = 6
HIGHER_IS_BETTER = {"tokens_per_sec", "tokens_per_sec_after_warmup"}
BASELINES = {
    "phase1_lr": "LR0300",
    "phase2_epoch": "E0025",
    "phase3_vocab": "V12000",
    "phase4_tokenizer_granularity": "MF002",
    "phase5_capacity": "M031",
    "phase6_context": "CTX0512",
    "phase7_batch_size": "BS004",
    "phase8_dropout": "DO100",
    "phase9_weight_decay": "WD050",
    "phase10_grad_clip": "GC100",
    "phase11_ffn_mult": "FFN04",
    "phase12_activation": "ACT_GELU",
    "phase13_init_std": "INIT020",
    "phase14_structure": "STRUCT_BASE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--meta-json", type=Path, default=DEFAULT_META)
    parser.add_argument("--all-results-jsonl", type=Path, default=DEFAULT_ALL_RESULTS_JSONL)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def read_result_with_retry(path: Path, attempts: int = 3) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return read_json(path)
        except Exception as exc:  # pragma: no cover - defensive against concurrent writes.
            last_exc = exc
            if attempt + 1 < attempts:
                time.sleep(0.1)
    assert last_exc is not None
    raise last_exc


def load_results(runs_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    load_errors: list[dict[str, str]] = []
    result_file_count = 0
    non_completed_result_files = 0
    for result_path in sorted(runs_dir.glob("run_*/result.json")):
        result_file_count += 1
        try:
            payload = read_result_with_retry(result_path)
        except Exception as exc:
            load_errors.append({"path": str(result_path), "error": str(exc)})
            continue
        if payload.get("status") == "completed":
            payload["_result_path"] = str(result_path)
            payload["_run_dir"] = str(result_path.parent)
            rows.append(payload)
        else:
            non_completed_result_files += 1
    return rows, {
        "result_file_count": result_file_count,
        "result_load_error_count": len(load_errors),
        "result_load_errors": load_errors,
        "non_completed_result_files": non_completed_result_files,
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def token_scale_from_result(row: dict[str, Any], split: str) -> tuple[float, float]:
    manifest = row.get("cache_vocab_manifest", {}) if isinstance(row.get("cache_vocab_manifest"), dict) else {}
    tokens_per_char = to_float(row.get(f"{split}_tokens_per_char"), default=0.0)
    chars_per_token = to_float(row.get(f"{split}_chars_per_token"), default=0.0)
    if tokens_per_char == 0.0:
        tokens_per_char = to_float(manifest.get(f"{split}_tokens_per_char"), default=0.0)
    if chars_per_token == 0.0:
        chars_per_token = to_float(manifest.get(f"{split}_chars_per_token"), default=0.0)
    tokens = to_float(manifest.get(f"{split}_tokens"), default=0.0)
    chars = to_float(manifest.get(f"{split}_chars"), default=0.0)
    if tokens_per_char == 0.0:
        tokens_per_char = safe_ratio(tokens, chars)
    if chars_per_token == 0.0:
        chars_per_token = safe_ratio(chars, tokens)
    return tokens_per_char, chars_per_token


def enrich_result_metrics(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    train_tpc, train_cpt = token_scale_from_result(enriched, "train")
    val_tpc, _ = token_scale_from_result(enriched, "val")
    final_train_loss = to_float(enriched.get("final_train_loss"), default=float("nan"))
    final_val_loss = to_float(enriched.get("final_val_loss"), default=float("nan"))
    best_val_loss = to_float(enriched.get("best_val_loss"), default=final_val_loss)
    tokens_seen = to_float(enriched.get("tokens_seen"), default=0.0)
    parameter_count = to_float(enriched.get("parameter_count"), default=0.0)
    compute_proxy = to_float(enriched.get("compute_proxy"), default=0.0)
    if compute_proxy == 0.0:
        compute_proxy = to_float(enriched.get("compute_proxy_param_tokens"), default=0.0)
    if compute_proxy == 0.0:
        compute_proxy = parameter_count * tokens_seen
    estimated_train_flops = to_float(enriched.get("estimated_train_flops"), default=0.0)
    if estimated_train_flops == 0.0:
        estimated_train_flops = TRAINING_FLOPS_FACTOR * compute_proxy
    enriched.setdefault("final_train_nats_per_char", final_train_loss * train_tpc)
    enriched.setdefault("final_val_nats_per_char", final_val_loss * val_tpc)
    enriched.setdefault("final_train_bits_per_char", final_train_loss * train_tpc / LN2)
    enriched.setdefault("final_val_bits_per_char", final_val_loss * val_tpc / LN2)
    enriched.setdefault("best_val_nats_per_char", best_val_loss * val_tpc)
    enriched.setdefault("best_val_bits_per_char", best_val_loss * val_tpc / LN2)
    enriched.setdefault("final_minus_best_val_loss", final_val_loss - best_val_loss)
    enriched.setdefault("best_tokens_seen", int(enriched.get("best_step", 0) or 0) * int(enriched.get("batch_size", 0) or 0) * int(enriched.get("context_length", 0) or 0))
    enriched.setdefault("estimated_chars_seen", tokens_seen * train_cpt)
    enriched.setdefault("tokens_per_param", safe_ratio(tokens_seen, parameter_count))
    enriched.setdefault("tokens_per_parameter", safe_ratio(tokens_seen, parameter_count))
    enriched.setdefault("compute_proxy", compute_proxy)
    enriched.setdefault("compute_proxy_param_tokens", compute_proxy)
    enriched.setdefault("estimated_train_flops", estimated_train_flops)
    enriched.setdefault("tokens_per_sec_after_warmup", enriched.get("tokens_per_sec", 0.0))
    return enriched


def run_number_from_result(row: dict[str, Any]) -> int:
    return parse_int(row.get("run_number", Path(str(row["_run_dir"])).name.split("_")[-1]))


def build_result_audit(matrix_rows: list[dict[str, str]], physical_results: list[dict[str, Any]], scan_meta: dict[str, Any]) -> dict[str, Any]:
    matrix_by_run = {parse_int(row["run_number"]): row for row in matrix_rows}
    matrix_run_numbers = set(matrix_by_run)
    completed_run_numbers = [run_number_from_result(row) for row in physical_results]
    completed_run_set = set(completed_run_numbers)
    duplicates = sorted(run_number for run_number, count in Counter(completed_run_numbers).items() if count > 1)
    without_matrix = sorted(run_number for run_number in completed_run_set if run_number not in matrix_run_numbers)
    pending_run_numbers = sorted(matrix_run_numbers - completed_run_set)
    mismatches: list[dict[str, Any]] = []
    for result in physical_results:
        run_number = run_number_from_result(result)
        matrix_row = matrix_by_run.get(run_number)
        if matrix_row is None:
            continue
        expected_condition = matrix_row.get("condition_id")
        expected_seed = parse_int(matrix_row.get("seed", 0))
        observed_condition = str(result.get("condition_id"))
        observed_seed = parse_int(result.get("seed", 0))
        if observed_condition != expected_condition or observed_seed != expected_seed:
            mismatches.append(
                {
                    "run_number": run_number,
                    "result_path": str(result.get("_result_path", "")),
                    "expected_condition_id": expected_condition,
                    "observed_condition_id": observed_condition,
                    "expected_seed": expected_seed,
                    "observed_seed": observed_seed,
                }
            )
    return {
        **scan_meta,
        "completed_physical_runs": len(physical_results),
        "completed_physical_run_numbers": sorted(completed_run_set),
        "pending_physical_runs": len(pending_run_numbers),
        "pending_physical_run_numbers": pending_run_numbers,
        "duplicate_completed_run_numbers": duplicates,
        "completed_results_without_matrix": without_matrix,
        "matrix_result_mismatch_count": len(mismatches),
        "matrix_result_mismatches": mismatches,
    }


def write_all_results_jsonl(path: Path, matrix_rows: list[dict[str, str]], physical_results: list[dict[str, Any]]) -> int:
    matrix_by_run = {parse_int(row["run_number"]): row for row in matrix_rows}
    ledger_rows: list[dict[str, Any]] = []
    for result in sorted(physical_results, key=run_number_from_result):
        run_number = run_number_from_result(result)
        result_payload = {key: value for key, value in result.items() if not key.startswith("_")}
        ledger_rows.append(
            {
                "schema_version": 1,
                "record_type": "completed_physical_run_result",
                "run_number": run_number,
                "condition_id": result_payload.get("condition_id"),
                "seed": result_payload.get("seed"),
                "status": result_payload.get("status"),
                "result_path": result.get("_result_path"),
                "run_dir": result.get("_run_dir"),
                "matrix_row": matrix_by_run.get(run_number, {}),
                "result": result_payload,
            }
        )
    return write_jsonl(path, ledger_rows)


def parse_epoch_milestones(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    return [int(part) for part in str(value).replace(",", "|").split("|") if part.strip()]


def read_history(path: Path) -> list[dict[str, Any]]:
    history_path = path / "history.jsonl"
    if not history_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with history_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json_loads(line))
            except Exception:
                continue
    return rows


def json_loads(line: str) -> dict[str, Any]:
    import json

    return json.loads(line)


def compute_overfit_metrics(initial_train_loss: float, initial_val_loss: float, final_train_loss: float, final_val_loss: float) -> dict[str, Any]:
    train_loss_delta = initial_train_loss - final_train_loss
    val_loss_delta = initial_val_loss - final_val_loss
    initial_generalization_gap = initial_val_loss - initial_train_loss
    final_generalization_gap = final_val_loss - final_train_loss
    generalization_gap_delta = final_generalization_gap - initial_generalization_gap
    train_val_improvement_gap = train_loss_delta - val_loss_delta
    overfit_score = max(0.0, final_generalization_gap) + max(0.0, generalization_gap_delta) + max(0.0, train_val_improvement_gap)

    if val_loss_delta < -0.01:
        fit_status = "val_regressed"
    elif train_loss_delta < 0.005 and val_loss_delta < 0.005:
        fit_status = "underfit_or_too_short"
    elif generalization_gap_delta > 0.05 and train_val_improvement_gap > 0.02:
        fit_status = "overfit_risk"
    elif train_loss_delta > 0.01 and val_loss_delta > 0.01:
        fit_status = "generalizing"
    else:
        fit_status = "mixed"

    return {
        "train_loss_delta": train_loss_delta,
        "val_loss_delta": val_loss_delta,
        "initial_generalization_gap": initial_generalization_gap,
        "final_generalization_gap": final_generalization_gap,
        "generalization_gap_delta": generalization_gap_delta,
        "train_val_improvement_gap": train_val_improvement_gap,
        "overfit_score": overfit_score,
        "fit_status": fit_status,
    }


def milestone_event(history: list[dict[str, Any]], milestone: int) -> dict[str, Any] | None:
    eligible = [row for row in history if float(row.get("epoch", 0.0)) >= milestone]
    if not eligible:
        return None
    return min(eligible, key=lambda row: float(row.get("epoch", 0.0)))


def expand_epoch_milestones(matrix_rows: list[dict[str, str]], results: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    epoch_rows = [row for row in matrix_rows if row.get("phase") == "phase2_epoch" and row.get("epoch_milestones")]
    if not epoch_rows:
        return matrix_rows, results

    milestones = parse_epoch_milestones(epoch_rows[0].get("epoch_milestones"))
    if not milestones:
        return matrix_rows, results

    expanded_matrix = [row for row in matrix_rows if not (row.get("phase") == "phase2_epoch" and row.get("epoch_milestones"))]
    for row in epoch_rows:
        for milestone in milestones:
            virtual = dict(row)
            virtual["condition_id"] = f"E{milestone:04d}"
            virtual["purpose"] = f"epoch milestone {milestone} from 1500-epoch long-run"
            virtual["sweep_axis"] = "epochs"
            virtual["axis_value"] = str(milestone)
            virtual["epochs"] = str(milestone)
            virtual["epoch_milestones"] = ""
            expanded_matrix.append(virtual)

    expanded_results = [row for row in results if not (row.get("phase") == "phase2_epoch" and row.get("epoch_milestones"))]
    epoch_result_ids = {row["condition_id"] for row in epoch_rows}
    for result in results:
        if result.get("phase") != "phase2_epoch" or result.get("condition_id") not in epoch_result_ids:
            continue
        history = read_history(Path(str(result["_run_dir"])))
        if not history:
            continue
        train_tpc = float(result.get("cache_vocab_manifest", {}).get("train_tokens_per_char", 1.0))
        val_tpc = float(result.get("cache_vocab_manifest", {}).get("val_tokens_per_char", 1.0))
        train_cpt = float(result.get("cache_vocab_manifest", {}).get("train_chars_per_token", safe_ratio(1.0, train_tpc)))
        parameter_count = float(result.get("parameter_count", 0.0))
        for milestone in milestones:
            event = milestone_event(history, milestone)
            if event is None:
                continue
            events_to_milestone = [row for row in history if float(row.get("epoch", 0.0)) <= float(event.get("epoch", 0.0))]
            best_event = min(events_to_milestone, key=lambda row: float(row.get("val_loss", float("inf"))))
            train_loss = float(event["train_loss"])
            val_loss = float(event["val_loss"])
            best_val_loss = float(best_event.get("val_loss", val_loss))
            tokens_seen = int(event.get("tokens_seen", result.get("tokens_seen", 0)))
            compute_proxy = parameter_count * tokens_seen
            virtual_result = dict(result)
            virtual_result.update(
                {
                    "condition_id": f"E{milestone:04d}",
                    "purpose": f"epoch milestone {milestone} from 1500-epoch long-run",
                    "sweep_axis": "epochs",
                    "axis_value": str(milestone),
                    "epochs": milestone,
                    "resolved_epochs": float(event.get("epoch", milestone)),
                    "final_train_loss": train_loss,
                    "final_val_loss": val_loss,
                    "final_train_nats_per_char": train_loss * train_tpc,
                    "final_val_nats_per_char": val_loss * val_tpc,
                    "final_train_bits_per_char": train_loss * train_tpc / math.log(2),
                    "final_val_bits_per_char": val_loss * val_tpc / math.log(2),
                    "best_val_loss": best_val_loss,
                    "best_val_nats_per_char": best_val_loss * val_tpc,
                    "best_val_bits_per_char": best_val_loss * val_tpc / math.log(2),
                    "best_epoch": float(best_event.get("epoch", milestone)),
                    "best_tokens_seen": int(best_event.get("tokens_seen", tokens_seen)),
                    "final_minus_best_val_loss": val_loss - best_val_loss,
                    "elapsed_sec": float(event.get("elapsed_sec", result.get("elapsed_sec", 0.0))),
                    "seconds_per_epoch": float(event.get("elapsed_sec", result.get("elapsed_sec", 0.0))) / max(1, milestone),
                    "tokens_seen": tokens_seen,
                    "estimated_chars_seen": tokens_seen * train_cpt,
                    "tokens_per_param": safe_ratio(tokens_seen, parameter_count),
                    "tokens_per_parameter": safe_ratio(tokens_seen, parameter_count),
                    "compute_proxy": compute_proxy,
                    "compute_proxy_param_tokens": compute_proxy,
                    "estimated_train_flops": TRAINING_FLOPS_FACTOR * compute_proxy,
                    "tokens_per_sec": float(event.get("tokens_per_sec", result.get("tokens_per_sec", 0.0))),
                    "tokens_per_sec_after_warmup": float(event.get("tokens_per_sec_after_warmup", result.get("tokens_per_sec_after_warmup", result.get("tokens_per_sec", 0.0)))),
                    "epoch_milestones": "",
                }
            )
            virtual_result.update(
                compute_overfit_metrics(
                    float(result["initial_train_loss"]),
                    float(result["initial_val_loss"]),
                    train_loss,
                    val_loss,
                )
            )
            expanded_results.append(virtual_result)

    return expanded_matrix, expanded_results


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    frac = pos - lower
    return ordered[lower] * (1 - frac) + ordered[upper] * frac


def bootstrap_ci(values: list[float], iterations: int, rng: random.Random) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    means: list[float] = []
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in values]
        means.append(mean(sample))
    return percentile(means, 0.025), percentile(means, 0.975)


def stats_for(values: list[float], iterations: int, rng: random.Random) -> dict[str, float]:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return {"n": 0}
    ci_low, ci_high = bootstrap_ci(clean, iterations, rng)
    return {
        "n": len(clean),
        "mean": mean(clean),
        "median": median(clean),
        "std": 0.0 if len(clean) < 2 else stdev(clean),
        "iqr": percentile(clean, 0.75) - percentile(clean, 0.25),
        "min": min(clean),
        "max": max(clean),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def matrix_lookup(matrix_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in matrix_rows:
        lookup.setdefault(row["condition_id"], row)
    return lookup


def paired_deltas(results: list[dict[str, Any]], metric: str) -> dict[tuple[str, str], list[float]]:
    by_phase_condition_seed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in results:
        by_phase_condition_seed[(row["phase"], row["condition_id"], parse_int(row["seed"]))] = row

    deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in results:
        phase = row["phase"]
        baseline = BASELINES.get(phase)
        if baseline is None or row["condition_id"] == baseline:
            continue
        base_row = by_phase_condition_seed.get((phase, baseline, parse_int(row["seed"])))
        if base_row is None or metric not in base_row or metric not in row:
            continue
        deltas[(phase, row["condition_id"])].append(float(row[metric]) - float(base_row[metric]))
    return deltas


def paired_delta_details(results: list[dict[str, Any]], metric: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    by_phase_condition_seed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in results:
        by_phase_condition_seed[(row["phase"], row["condition_id"], parse_int(row["seed"]))] = row

    details: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        phase = row["phase"]
        baseline = BASELINES.get(phase)
        if baseline is None or row["condition_id"] == baseline:
            continue
        seed = parse_int(row["seed"])
        base_row = by_phase_condition_seed.get((phase, baseline, seed))
        if base_row is None or metric not in base_row or metric not in row:
            continue
        baseline_value = float(base_row[metric])
        condition_value = float(row[metric])
        details[(phase, row["condition_id"])].append(
            {
                "seed": seed,
                "baseline_condition_id": baseline,
                "baseline_value": baseline_value,
                "condition_value": condition_value,
                "delta": condition_value - baseline_value,
            }
        )
    return details


def better_than_baseline(delta: float, metric: str) -> bool:
    if metric in HIGHER_IS_BETTER:
        return delta > 0
    return delta < 0


def aggregate(matrix_rows: list[dict[str, str]], results: list[dict[str, Any]], bootstrap: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["condition_id"]].append(row)

    lookup = matrix_lookup(matrix_rows)
    planned_by_condition: dict[str, int] = defaultdict(int)
    for row in matrix_rows:
        planned_by_condition[row["condition_id"]] += 1
    delta_details_by_metric = {metric: paired_delta_details(results, metric) for metric in METRICS}
    rows: list[dict[str, Any]] = []
    for condition_id, matrix_row in sorted(lookup.items(), key=lambda item: parse_int(item[1]["run_number"])):
        condition_results = grouped.get(condition_id, [])
        screen_ready = len(condition_results) >= 3
        claim_ready = len(condition_results) >= 10
        summary: dict[str, Any] = {
            "phase": matrix_row["phase"],
            "condition_id": condition_id,
            "purpose": matrix_row["purpose"],
            "stage": matrix_row.get("stage", "exploratory"),
            "sweep_axis": matrix_row.get("sweep_axis", ""),
            "axis_value": matrix_row.get("axis_value", ""),
            "planned_runs": planned_by_condition[condition_id],
            "completed_runs": len(condition_results),
            "screen_ready": screen_ready,
            "claim_ready": claim_ready,
            "ready_for_screen": len(condition_results) >= min(3, planned_by_condition[condition_id]) and len(condition_results) >= planned_by_condition[condition_id],
            "ready_for_claim": len(condition_results) >= 10 and len(condition_results) >= planned_by_condition[condition_id],
        }
        for metric in METRICS:
            metric_stats = stats_for([float(row[metric]) for row in condition_results if metric in row], bootstrap, rng)
            for key, value in metric_stats.items():
                summary[f"{metric}_{key}"] = value
            metric_details = delta_details_by_metric[metric].get((matrix_row["phase"], condition_id), [])
            metric_deltas = [float(detail["delta"]) for detail in metric_details]
            delta_stats = stats_for(metric_deltas, bootstrap, rng)
            for key, value in delta_stats.items():
                summary[f"{metric}_paired_delta_to_baseline_{key}"] = value
            if metric_details:
                wins = sum(1 for detail in metric_details if better_than_baseline(float(detail["delta"]), metric))
                summary[f"{metric}_baseline_win_rate"] = wins / len(metric_details)
            else:
                summary[f"{metric}_baseline_win_rate"] = ""
        primary_details = delta_details_by_metric["final_val_bits_per_char"].get((matrix_row["phase"], condition_id), [])
        deltas = [float(detail["delta"]) for detail in primary_details]
        delta_stats = stats_for(deltas, bootstrap, rng)
        for key, value in delta_stats.items():
            summary[f"paired_delta_bits_vs_phase_baseline_{key}"] = value
        summary["paired_delta_to_baseline_median"] = delta_stats.get("median", "")
        summary["paired_delta_to_baseline_by_seed"] = json.dumps(primary_details, ensure_ascii=False, sort_keys=True) if primary_details else ""
        if deltas:
            summary["beats_baseline_count"] = sum(1 for value in deltas if better_than_baseline(value, "final_val_bits_per_char"))
            summary["beats_baseline_ratio"] = summary["beats_baseline_count"] / len(deltas)
            summary["baseline_win_rate"] = summary["beats_baseline_ratio"]
        else:
            summary["beats_baseline_count"] = ""
            summary["beats_baseline_ratio"] = ""
            summary["baseline_win_rate"] = ""
        rows.append(summary)

    meta = {
        "planned_conditions": len(lookup),
        "planned_runs": len(matrix_rows),
        "completed_runs": len(results),
        "screen_ready_conditions": sum(1 for row in rows if row["ready_for_screen"]),
        "claim_ready_conditions": sum(1 for row in rows if row["ready_for_claim"]),
    }
    return rows, meta


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    if value == "" or value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.6g}"


def write_report(path: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    ledger_status = str(meta.get("result_ledger_status", "UNKNOWN"))
    lines = [
        "# LLM 10x 집계 보고서",
        "",
        "## 현재 상태",
        "",
        f"- 계획된 조건 수: `{meta['planned_conditions']}`",
        f"- 계획된 분석 행 수: `{meta['planned_runs']}`",
        f"- 계획된 실제 실행 수: `{meta.get('planned_physical_runs', meta['planned_runs'])}`",
        f"- 완료된 실행 수: `{meta['completed_runs']}`",
        f"- 완료된 실제 실행 수: `{meta.get('completed_physical_runs', meta['completed_runs'])}`",
        f"- 화면 검토 가능 조건(`탐색 조건 n >= 3`): `{meta['screen_ready_conditions']}`",
        f"- 주장 근거 가능 조건(`n >= 10`): `{meta['claim_ready_conditions']}`",
        f"- 전체 실행 결과 원장: `{meta.get('all_results_jsonl', '')}`",
        f"- 원장 행 수: `{meta.get('all_results_ledger_rows', 0)}`",
        f"- 원장 누락 감사: `{ledger_status}`",
        f"- result.json 읽기 실패: `{meta.get('result_load_error_count', 0)}`",
        f"- 중복 완료 run_number: `{len(meta.get('duplicate_completed_run_numbers', []))}`",
        f"- matrix 밖 완료 결과: `{len(meta.get('completed_results_without_matrix', []))}`",
        f"- matrix/result 불일치: `{meta.get('matrix_result_mismatch_count', 0)}`",
        "",
        "탐색 조건은 3회 반복이 끝나면 화면 검토는 가능하지만, 최종 주장은 선별된 후보에 대해 10회 반복이 필요합니다.",
        "",
        "원장 누락 감사가 `PASS`가 아니면 완료된 개별 결과가 집계 원장에 빠졌거나, 읽을 수 없는 결과 파일/중복/계획표 불일치가 있다는 뜻입니다.",
        "",
        "## 조건별 요약",
        "",
        "| phase | 조건 | 단계 | 축 | 값 | n | 화면 검토 가능 | 주장 가능 | 검증 bits/char 중앙값 | 검증 IQR | 과적합 중앙값 | gap 중앙값 | 시간 h 중앙값 | warm tok/s 중앙값 | paired delta 중앙값 | 기준선 승률 |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        warm_tokens_per_sec = row.get("tokens_per_sec_after_warmup_median")
        if warm_tokens_per_sec in ("", None):
            warm_tokens_per_sec = row.get("tokens_per_sec_median")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["phase"]),
                    str(row["condition_id"]),
                    str(row.get("stage", "")),
                    str(row.get("sweep_axis", "")),
                    str(row.get("axis_value", "")),
                    fmt(row["completed_runs"]),
                    fmt(row["ready_for_screen"]),
                    fmt(row["ready_for_claim"]),
                    fmt(row.get("final_val_bits_per_char_median")),
                    fmt(row.get("final_val_bits_per_char_iqr")),
                    fmt(row.get("overfit_score_median")),
                    fmt(row.get("final_generalization_gap_median")),
                    fmt(None if row.get("elapsed_sec_median") is None else float(row.get("elapsed_sec_median", float("nan"))) / 3600),
                    fmt(warm_tokens_per_sec),
                    fmt(row.get("paired_delta_to_baseline_median")),
                    fmt(row.get("baseline_win_rate")),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    physical_matrix_rows = read_matrix(args.matrix)
    physical_results, scan_meta = load_results(args.runs_dir)
    physical_results = [enrich_result_metrics(row) for row in physical_results]
    audit = build_result_audit(physical_matrix_rows, physical_results, scan_meta)
    ledger_rows = write_all_results_jsonl(args.all_results_jsonl, physical_matrix_rows, physical_results)
    audit["all_results_jsonl"] = str(args.all_results_jsonl)
    audit["all_results_ledger_rows"] = ledger_rows
    audit["ledger_matches_completed_physical_runs"] = ledger_rows == len(physical_results)
    audit["result_ledger_status"] = (
        "PASS"
        if ledger_rows == len(physical_results)
        and audit["result_load_error_count"] == 0
        and not audit["duplicate_completed_run_numbers"]
        and not audit["completed_results_without_matrix"]
        and audit["matrix_result_mismatch_count"] == 0
        else "FAIL"
    )
    matrix_rows, results = expand_epoch_milestones(physical_matrix_rows, physical_results)
    results = [enrich_result_metrics(row) for row in results]
    rows, meta = aggregate(matrix_rows, results, bootstrap=args.bootstrap, seed=args.seed)
    meta["planned_physical_runs"] = len(physical_matrix_rows)
    meta["planned_analysis_rows"] = len(matrix_rows)
    meta.update(audit)
    write_summary_csv(args.summary_csv, rows)
    write_json(args.meta_json, meta)
    write_report(args.report, rows, meta)
    print(
        {
            "summary_csv": str(args.summary_csv),
            "report": str(args.report),
            "all_results_jsonl": str(args.all_results_jsonl),
            "planned_conditions": meta["planned_conditions"],
            "planned_runs": meta["planned_runs"],
            "planned_physical_runs": meta["planned_physical_runs"],
            "completed_runs": meta["completed_runs"],
            "completed_physical_runs": meta["completed_physical_runs"],
            "pending_physical_runs": meta["pending_physical_runs"],
            "screen_ready_conditions": meta["screen_ready_conditions"],
            "claim_ready_conditions": meta["claim_ready_conditions"],
            "all_results_ledger_rows": meta["all_results_ledger_rows"],
            "result_ledger_status": meta["result_ledger_status"],
            "result_load_error_count": meta["result_load_error_count"],
            "duplicate_completed_run_numbers": meta["duplicate_completed_run_numbers"],
            "matrix_result_mismatch_count": meta["matrix_result_mismatch_count"],
        }
    )
    if meta["result_ledger_status"] != "PASS":
        raise SystemExit("result ledger audit failed; see aggregate_report.md")


if __name__ == "__main__":
    main()
