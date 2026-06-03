#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the LLM 10x experiment matrix as a resumable queue."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from llm_10x_common import CACHE_DIR, DOCS_DIR, LOCAL_DIR, MATRIX_PATH, RUNS_DIR, cache_dir_for, parse_float, parse_int, read_json, read_matrix, run_dir_for, write_json


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUEUE_LOG = LOCAL_DIR / "queue_events.jsonl"
DEFAULT_STATUS = LOCAL_DIR / "queue_status.json"
DEFAULT_SUMMARY_CSV = DOCS_DIR / "aggregate_summary.csv"
DEFAULT_REPORT = DOCS_DIR / "aggregate_report.md"
DEFAULT_META = DOCS_DIR / "aggregate_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--queue-log", type=Path, default=DEFAULT_QUEUE_LOG)
    parser.add_argument("--status-json", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--meta-json", type=Path, default=DEFAULT_META)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-steps-per-run", type=int, default=None, help="Cap optimizer steps for each physical run.")
    parser.add_argument("--epoch-cap-per-run", type=float, default=None, help="Cap each physical run by authored epoch fraction.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--aggregate-every", type=int, default=1)
    parser.add_argument("--order", choices=["matrix", "cost_ascending"], default="matrix", help="Pending-run selection order.")
    parser.add_argument("--with-checkpoints", action="store_true", help="Store checkpoints. Disabled by default for broad sweeps.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time": now(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def completed_run_numbers(runs_dir: Path) -> set[int]:
    completed: set[int] = set()
    if not runs_dir.exists():
        return completed
    for result_path in runs_dir.glob("run_*/result.json"):
        try:
            payload = read_json(result_path)
        except Exception:
            continue
        if payload.get("status") == "completed":
            completed.add(parse_int(payload.get("run_number", result_path.parent.name.split("_")[-1])))
    return completed


def estimate_dataset_length(token_count: int, context_length: int) -> int:
    available = token_count - context_length - 1
    return 0 if available < 0 else available // context_length + 1


def estimate_steps_per_epoch(token_count: int, context_length: int, batch_size: int) -> int:
    return max(1, (estimate_dataset_length(token_count, context_length) + batch_size - 1) // batch_size)


def cached_train_tokens(row: dict[str, str], cache_root: Path) -> int:
    vocab_size = parse_int(row["vocab_size"])
    min_frequency = parse_int(row.get("tokenizer_min_frequency", 2))
    manifest_path = cache_dir_for(vocab_size, cache_root, min_frequency) / "manifest.json"
    if manifest_path.exists():
        try:
            return parse_int(read_json(manifest_path).get("train_tokens", 6_633_675))
        except Exception:
            return 6_633_675
    # The cached 12k tokenizer has 6,633,675 train tokens. Use it as a
    # stable proxy until the tokenizer-specific cache has been prepared.
    return 6_633_675


def estimated_run_cost(row: dict[str, str], args: argparse.Namespace) -> float:
    context_length = parse_int(row["context_length"])
    batch_size = parse_int(row.get("batch_size", 4))
    emb_dim = parse_int(row["emb_dim"])
    n_layers = parse_int(row["n_layers"])
    ffn_mult = parse_int(row["ffn_mult"])
    vocab_size = parse_int(row["vocab_size"])

    if args.max_steps_per_run is not None and args.epoch_cap_per_run is None:
        total_steps = args.max_steps_per_run
    else:
        epochs = args.epoch_cap_per_run if args.epoch_cap_per_run is not None else parse_float(row["epochs"])
        steps_per_epoch = estimate_steps_per_epoch(cached_train_tokens(row, args.cache_dir), context_length, batch_size)
        total_steps = max(1, int(steps_per_epoch * epochs + 0.999999))

    attention_cost = n_layers * batch_size * context_length * context_length * emb_dim
    block_projection_cost = n_layers * batch_size * context_length * emb_dim * emb_dim * (4 + 2 * ffn_mult)
    softmax_cost = batch_size * context_length * emb_dim * vocab_size
    implementation_penalty = 1.35 if row.get("attention_impl") == "manual" else 1.0
    cache_penalty = 0.0
    cache_dir = cache_dir_for(vocab_size, args.cache_dir, parse_int(row.get("tokenizer_min_frequency", 2)))
    if not all((cache_dir / name).exists() for name in ("tokenizer.json", "train_ids.u32", "val_ids.u32", "manifest.json")):
        cache_penalty = 0.05 * (attention_cost + block_projection_cost + softmax_cost)
    return total_steps * (attention_cost + block_projection_cost + softmax_cost) * implementation_penalty + cache_penalty


def pending_rows(rows: list[dict[str, str]], runs_dir: Path) -> list[dict[str, str]]:
    completed = completed_run_numbers(runs_dir)
    pending: list[dict[str, str]] = []
    for row in rows:
        run_number = parse_int(row["run_number"])
        if run_number in completed:
            continue
        result_path = run_dir_for(run_number, runs_dir) / "result.json"
        if result_path.exists():
            continue
        pending.append(row)
    return pending


def next_pending_row(rows: list[dict[str, str]], args: argparse.Namespace) -> dict[str, str] | None:
    candidates = pending_rows(rows, args.runs_dir)
    if not candidates:
        return None
    if args.order == "cost_ascending":
        return min(candidates, key=lambda row: (estimated_run_cost(row, args), parse_int(row["run_number"])))
    return candidates[0]


def run_command(command: list[str], event_path: Path, label: str) -> None:
    append_event(event_path, {"event": "command_start", "label": label, "command": command})
    start = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT)
    elapsed = time.perf_counter() - start
    append_event(event_path, {"event": "command_done", "label": label, "returncode": process.returncode, "elapsed_sec": elapsed})
    if process.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {process.returncode}")


def ensure_cache(row: dict[str, str], args: argparse.Namespace) -> None:
    vocab_size = parse_int(row["vocab_size"])
    min_frequency = parse_int(row.get("tokenizer_min_frequency", 2))
    cache_dir = cache_dir_for(vocab_size, args.cache_dir, min_frequency)
    required = ("tokenizer.json", "train_ids.u32", "val_ids.u32", "manifest.json")
    if all((cache_dir / name).exists() for name in required):
        append_event(args.queue_log, {"event": "cache_exists", "vocab_size": vocab_size, "tokenizer_min_frequency": min_frequency, "path": str(cache_dir)})
        return
    command = [
        sys.executable,
        "scripts/llm_10x_prepare_cache.py",
        "--matrix",
        str(args.matrix),
        "--cache-dir",
        str(args.cache_dir),
        "--vocab-size",
        str(vocab_size),
        "--min-frequency",
        str(min_frequency),
    ]
    run_command(command, args.queue_log, f"cache_vocab_{vocab_size}_minfreq_{min_frequency}")


def run_one(row: dict[str, str], args: argparse.Namespace) -> None:
    run_number = parse_int(row["run_number"])
    command = [
        sys.executable,
        "scripts/llm_10x_run_next.py",
        "--matrix",
        str(args.matrix),
        "--cache-dir",
        str(args.cache_dir),
        "--runs-dir",
        str(args.runs_dir),
        "--run-number",
        str(run_number),
        "--device",
        args.device,
        "--eval-batches",
        str(args.eval_batches),
        "--queue-status-json",
        str(args.status_json),
    ]
    if args.epoch_cap_per_run is not None:
        command.extend(["--epochs", str(args.epoch_cap_per_run)])
    elif args.max_steps_per_run is not None:
        command.extend(["--max-steps", str(args.max_steps_per_run)])
    if not args.with_checkpoints:
        command.append("--no-checkpoints")
    run_command(command, args.queue_log, f"run_{run_number:04d}_{row['condition_id']}")


def aggregate(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "scripts/llm_10x_aggregate.py",
        "--matrix",
        str(args.matrix),
        "--runs-dir",
        str(args.runs_dir),
        "--summary-csv",
        str(args.summary_csv),
        "--report",
        str(args.report),
        "--meta-json",
        str(args.meta_json),
    ]
    run_command(command, args.queue_log, "aggregate")


def write_status(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    write_json(args.status_json, {"updated_at": now(), **payload})


def main() -> None:
    args = parse_args()
    rows = read_matrix(args.matrix)
    completed_at_start = len(completed_run_numbers(args.runs_dir))
    append_event(
        args.queue_log,
        {
            "event": "queue_start",
            "planned_physical_runs": len(rows),
            "completed_at_start": completed_at_start,
            "max_runs": args.max_runs,
            "max_steps_per_run": args.max_steps_per_run,
            "epoch_cap_per_run": args.epoch_cap_per_run,
            "order": args.order,
            "device": args.device,
            "eval_batches": args.eval_batches,
            "with_checkpoints": args.with_checkpoints,
        },
    )
    write_status(args, {"status": "running", "planned_physical_runs": len(rows), "completed_runs": completed_at_start, "max_steps_per_run": args.max_steps_per_run, "epoch_cap_per_run": args.epoch_cap_per_run, "order": args.order})

    runs_done = 0
    try:
        while args.max_runs is None or runs_done < args.max_runs:
            row = next_pending_row(rows, args)
            if row is None:
                aggregate(args)
                write_status(args, {"status": "completed", "planned_physical_runs": len(rows), "completed_runs": len(completed_run_numbers(args.runs_dir))})
                append_event(args.queue_log, {"event": "queue_completed", "runs_done_this_session": runs_done})
                return

            run_number = parse_int(row["run_number"])
            selection_payload = {"event": "run_selected", "run_number": run_number, "condition_id": row["condition_id"], "seed": parse_int(row["seed"]), "order": args.order}
            if args.order == "cost_ascending":
                selection_payload["estimated_cost"] = estimated_run_cost(row, args)
            append_event(args.queue_log, selection_payload)
            write_status(args, {"status": "running", "stage": "selected", "current_run_number": run_number, "current_condition_id": row["condition_id"], "runs_done_this_session": runs_done, "max_steps_per_run": args.max_steps_per_run, "epoch_cap_per_run": args.epoch_cap_per_run, "order": args.order})

            if args.dry_run:
                append_event(args.queue_log, {"event": "dry_run_selected", "run_number": run_number})
                runs_done += 1
                continue

            ensure_cache(row, args)
            run_one(row, args)
            runs_done += 1

            if args.aggregate_every > 0 and runs_done % args.aggregate_every == 0:
                aggregate(args)

        write_status(args, {"status": "paused_after_max_runs", "runs_done_this_session": runs_done, "completed_runs": len(completed_run_numbers(args.runs_dir))})
        append_event(args.queue_log, {"event": "queue_paused_after_max_runs", "runs_done_this_session": runs_done})
    except Exception as exc:
        write_status(args, {"status": "failed", "error": str(exc), "runs_done_this_session": runs_done, "completed_runs": len(completed_run_numbers(args.runs_dir))})
        append_event(args.queue_log, {"event": "queue_failed", "error": str(exc), "runs_done_this_session": runs_done})
        raise


if __name__ == "__main__":
    main()
