# -*- coding: utf-8 -*-
"""HY LLM relog metric and artifact regression tests."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def test_hy_manifest_covers_report_experiments():
    from hy_llm_relog_common import TARGET_IDS, build_manifest, load_records

    records = load_records(ROOT / "docs" / "HY" / "testresult")
    rows = build_manifest(records)
    ids = {row["experiment_id"] for row in rows}

    assert set(TARGET_IDS).issubset(ids)
    assert all(str(row["rerun_required"]) == "False" for row in rows)


def test_bits_per_char_formula_from_hy_history():
    from hy_llm_relog_common import LOG2_E, build_history, load_records, number

    record = load_records(ROOT / "docs" / "HY" / "testresult")["E06"]
    history = build_history(record)
    final = history[-1]
    expected = number(final, "val_loss") * number(record.metrics, "val_token_count") / number(record.metrics, "val_char_count") / LOG2_E

    assert math.isclose(number(final, "val_bits_per_char"), expected, rel_tol=1e-9)


def test_vocab_token_loss_and_bits_per_char_rank_differ():
    from hy_llm_relog_common import build_history, build_run_metrics, load_records

    records = load_records(ROOT / "docs" / "HY" / "testresult")
    metrics = {experiment_id: build_run_metrics(records[experiment_id], build_history(records[experiment_id])) for experiment_id in ["E04", "E00", "E05", "E06"]}
    best_by_token_loss = min(metrics.values(), key=lambda row: row["final_val_loss"])["experiment_id"]
    best_by_bits = min(metrics.values(), key=lambda row: row["final_val_bits_per_char"])["experiment_id"]

    assert best_by_token_loss == "E04"
    assert best_by_bits == "E06"


def test_relog_aggregate_outputs_exist_after_generation():
    metrics = ROOT / "docs" / "HY" / "llm_relog" / "report_llm_metrics.csv"
    history = ROOT / "docs" / "HY" / "llm_relog" / "report_llm_metrics_by_step.csv"
    figures = ROOT / "docs" / "HY" / "figures_llm_seq"

    assert metrics.exists()
    assert history.exists()
    assert (figures / "02_vocab_bits_per_char_learning_curve.png").stat().st_size > 10_000
    assert (figures / "14_loss_vs_compute_all_paths.png").stat().st_size > 10_000
