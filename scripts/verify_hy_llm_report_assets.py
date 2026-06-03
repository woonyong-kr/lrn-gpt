# -*- coding: utf-8 -*-
"""Verify REPORT_temp.md HY LLM relog outputs and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg

from hy_llm_relog_common import FIGURE_DIR, RELOG_DIR, ROOT, TARGET_IDS, read_csv

REQUIRED_FIGURES = [
    "01_context_length_learning_curve.png",
    "02_vocab_bits_per_char_learning_curve.png",
    "03_emb_dim_capacity_path.png",
    "04_n_heads_learning_curve.png",
    "05_n_layers_depth_curve.png",
    "06_norm_depth_stability_curve.png",
    "07_ffn_multiplier_capacity_path.png",
    "08_activation_seed_mean_std_curve.png",
    "09_dropout_overfit_curve.png",
    "10_qkv_bias_curve.png",
    "11_weight_tying_long_curve.png",
    "12_stride_overlap_curve.png",
    "13_epoch_overfit_rebound_audit.png",
    "14_loss_vs_compute_all_paths.png",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="REPORT_temp.md")
    parser.add_argument("--figures", default=str(FIGURE_DIR))
    args = parser.parse_args()
    report_path = Path(args.report)
    figures_dir = Path(args.figures)
    failures: list[str] = []

    text = report_path.read_text(encoding="utf-8")
    if "docs/llm_10x" in text:
        failures.append("REPORT_temp.md references docs/llm_10x")
    for required in ["bits/char", "final_minus_best_val_loss", "train_val_gap", "tokens_seen"]:
        if required not in text:
            failures.append(f"REPORT_temp.md missing {required}")

    for figure in REQUIRED_FIGURES:
        path = figures_dir / figure
        if not path.exists():
            failures.append(f"missing figure {path}")
            continue
        if path.stat().st_size < 10_000:
            failures.append(f"figure too small {path}")
            continue
        try:
            image = mpimg.imread(path)
        except Exception as exc:  # pragma: no cover - defensive verification path
            failures.append(f"cannot read figure {path}: {exc}")
            continue
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            failures.append(f"empty dimensions {path}")
        if float(image[..., :3].std()) < 0.005:
            failures.append(f"figure appears blank {path}")

    metrics_path = RELOG_DIR / "report_llm_metrics.csv"
    history_path = RELOG_DIR / "report_llm_metrics_by_step.csv"
    if not metrics_path.exists() or not history_path.exists():
        failures.append("missing relog aggregate csv files")
    else:
        metric_ids = {str(row["experiment_id"]) for row in read_csv(metrics_path)}
        history_rows = read_csv(history_path)
        history_ids = {str(row["experiment_id"]) for row in history_rows}
        missing_metrics = sorted(set(TARGET_IDS) - metric_ids, key=lambda value: int(value[1:]))
        missing_history = sorted(set(TARGET_IDS) - history_ids, key=lambda value: int(value[1:]))
        if missing_metrics:
            failures.append("missing metrics for " + ", ".join(missing_metrics))
        if missing_history:
            failures.append("missing history for " + ", ".join(missing_history))
        for experiment_id in TARGET_IDS:
            count = sum(1 for row in history_rows if str(row["experiment_id"]) == experiment_id)
            if count < 2:
                failures.append(f"not enough evaluation points for {experiment_id}: {count}")

    if (ROOT / "REPORT.md").read_text(encoding="utf-8").startswith("# mini GPT 구현") is False:
        failures.append("REPORT.md does not look like the preserved original report")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("HY LLM report assets verified")


if __name__ == "__main__":
    main()
