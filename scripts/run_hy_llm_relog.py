# -*- coding: utf-8 -*-
"""Relog existing HY markdown experiment histories into LLM run artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from hy_llm_relog_common import (
    RELOG_DIR,
    TARGET_IDS,
    aggregate_outputs,
    build_manifest,
    load_records,
    read_csv,
    relog_records,
    write_csv,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(RELOG_DIR / "experiment_manifest.csv"))
    parser.add_argument("--source", default="docs/HY/testresult")
    parser.add_argument("--out", default=str(RELOG_DIR / "runs"))
    args = parser.parse_args()

    records = load_records(Path(args.source))
    missing = [experiment_id for experiment_id in TARGET_IDS if experiment_id not in records]
    if missing:
        raise SystemExit(f"missing HY source logs: {', '.join(missing)}")

    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest_rows = read_csv(manifest_path)
    else:
        manifest_rows = build_manifest(records)
        write_manifest(manifest_rows, manifest_path)

    needed = [row for row in manifest_rows if str(row.get("rerun_required")) == "True"]
    write_csv(needed, RELOG_DIR / "rerun_required.csv")
    if needed:
        print(f"warning: {len(needed)} manifest rows require rerun; relogging available source logs first")

    metrics, history = relog_records(records, manifest_rows)
    aggregate_outputs(metrics, history)
    print(f"wrote {len(metrics)} run artifacts under {args.out}")
    print(f"wrote {len(history)} step rows under {RELOG_DIR}")


if __name__ == "__main__":
    main()
