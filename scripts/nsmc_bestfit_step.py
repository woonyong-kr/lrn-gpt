#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one rule-selected NSMC best-fit experiment step."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from nsmc_bestfit_rules import CACHE_DIR, DECISION_REPORT_PATH, DOCS_DIR, MATRIX_PATH, NEXT_PLAN_PATH, ROOT, RUNS_DIR, TRAIN_TEXT, VAL_TEXT, apply_rules, read_json


LOCAL_DIR = ROOT / "local" / "nsmc_bestfit"
QUEUE_LOG = LOCAL_DIR / "queue_events.jsonl"
STATUS_JSON = LOCAL_DIR / "queue_status.json"
AGGREGATE_CSV = DOCS_DIR / "aggregate_summary.csv"
AGGREGATE_REPORT = DOCS_DIR / "aggregate_report.md"
AGGREGATE_META = DOCS_DIR / "aggregate_meta.json"
LOCK_PATH = LOCAL_DIR / "step.lock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--train-text", type=Path, default=TRAIN_TEXT)
    parser.add_argument("--val-text", type=Path, default=VAL_TEXT)
    parser.add_argument("--queue-log", type=Path, default=QUEUE_LOG)
    parser.add_argument("--status-json", type=Path, default=STATUS_JSON)
    parser.add_argument("--summary-csv", type=Path, default=AGGREGATE_CSV)
    parser.add_argument("--report", type=Path, default=AGGREGATE_REPORT)
    parser.add_argument("--meta-json", type=Path, default=AGGREGATE_META)
    parser.add_argument("--max-runs", type=int, default=1)
    parser.add_argument("--max-steps-per-run", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--eval-batches", type=int, default=20)
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


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"updated_at": now(), **payload}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def step_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", 0))
        except Exception:
            pid = 0
        if pid and process_alive(pid):
            raise SystemExit(f"NSMC best-fit step is already running with pid={pid}")
    path.write_text(json.dumps({"pid": os.getpid(), "started_at": now()}, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        yield
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def run_command(command: list[str], label: str, log_path: Path, dry_run: bool = False) -> None:
    append_event(log_path, {"event": "command_start", "label": label, "command": command, "dry_run": dry_run})
    if dry_run:
        return
    process = subprocess.run(command, cwd=ROOT)
    append_event(log_path, {"event": "command_done", "label": label, "returncode": process.returncode})
    if process.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {process.returncode}")


def ensure_cache(row: dict[str, Any], args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "scripts/llm_10x_prepare_cache.py",
        "--matrix",
        str(args.matrix),
        "--train-text",
        str(args.train_text),
        "--val-text",
        str(args.val_text),
        "--cache-dir",
        str(args.cache_dir),
        "--vocab-size",
        str(row["vocab_size"]),
        "--min-frequency",
        str(row.get("tokenizer_min_frequency", 2)),
    ]
    run_command(command, f"cache_{row['condition_id']}", args.queue_log, dry_run=args.dry_run)


def run_experiment(row: dict[str, Any], args: argparse.Namespace) -> None:
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
        str(row["run_number"]),
        "--device",
        args.device,
        "--eval-batches",
        str(args.eval_batches),
        "--queue-status-json",
        str(args.status_json),
    ]
    if args.max_steps_per_run is not None:
        command.extend(["--max-steps", str(args.max_steps_per_run)])
    run_command(command, f"run_{int(row['run_number']):04d}_{row['condition_id']}", args.queue_log, dry_run=args.dry_run)


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
        "--all-results-jsonl",
        str(args.docs_dir / "all_run_results.jsonl"),
    ]
    run_command(command, "aggregate", args.queue_log, dry_run=args.dry_run)


def main() -> None:
    args = parse_args()
    with step_lock(LOCK_PATH):
        completed = 0
        while completed < args.max_runs:
            plan = apply_rules(matrix_path=args.matrix, runs_dir=args.runs_dir, docs_dir=args.docs_dir, write=True)
            next_row = plan.get("next_run")
            write_status(args.status_json, {"status": "planning", "plan": plan, "decision_report": str(args.docs_dir / DECISION_REPORT_PATH.name)})
            if not next_row:
                append_event(args.queue_log, {"event": "no_pending_run", "plan": plan})
                break
            append_event(args.queue_log, {"event": "selected", "run_number": next_row["run_number"], "condition_id": next_row["condition_id"], "phase": next_row["phase"]})
            if args.dry_run:
                print(json.dumps({"dry_run": True, "next_plan_path": str(args.docs_dir / NEXT_PLAN_PATH.name), "next_run": next_row}, ensure_ascii=False, indent=2, sort_keys=True))
                break
            ensure_cache(next_row, args)
            run_experiment(next_row, args)
            aggregate(args)
            apply_rules(matrix_path=args.matrix, runs_dir=args.runs_dir, docs_dir=args.docs_dir, write=True)
            completed += 1
        write_status(args.status_json, {"status": "idle", "completed_this_step": completed, "decision_report": str(args.docs_dir / DECISION_REPORT_PATH.name)})


if __name__ == "__main__":
    main()
