# -*- coding: utf-8 -*-
"""Build a REPORT.md HY experiment manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from hy_llm_relog_common import RELOG_DIR, build_manifest, load_records, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="docs/HY/testresult")
    parser.add_argument("--out", default=str(RELOG_DIR / "experiment_manifest.csv"))
    args = parser.parse_args()
    records = load_records(Path(args.source))
    rows = build_manifest(records)
    write_manifest(rows, Path(args.out))
    rerun_count = sum(1 for row in rows if str(row["rerun_required"]) == "True")
    print(f"wrote {args.out} ({len(rows)} rows, rerun_required={rerun_count})")


if __name__ == "__main__":
    main()
