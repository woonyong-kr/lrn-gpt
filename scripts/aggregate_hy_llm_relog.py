# -*- coding: utf-8 -*-
"""Aggregate HY LLM relog run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hy_llm_relog_common import RELOG_DIR, aggregate_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=str(RELOG_DIR / "runs"))
    parser.add_argument("--out", default=str(RELOG_DIR))
    args = parser.parse_args()

    runs_dir = Path(args.runs)
    metrics = []
    history = []
    for run_dir in sorted(runs_dir.glob("E*")):
        metrics_path = run_dir / "metrics.json"
        history_path = run_dir / "history.jsonl"
        if not metrics_path.exists() or not history_path.exists():
            continue
        metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                history.append(json.loads(line))
    aggregate_outputs(metrics, history)
    print(f"aggregated {len(metrics)} runs and {len(history)} history rows into {args.out}")


if __name__ == "__main__":
    main()
