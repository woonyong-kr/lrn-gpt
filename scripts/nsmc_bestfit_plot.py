#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate cumulative linear graphs for NSMC best-fit runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from nsmc_bestfit_rules import DOCS_DIR, ROOT, RUNS_DIR, selection_score


LEDGER_PATH = DOCS_DIR / "all_run_results.jsonl"
FIGURES_DIR = DOCS_DIR / "linear_graphs"
INDEX_PATH = FIGURES_DIR / "figure_index.md"
MANIFEST_PATH = FIGURES_DIR / "linear_graph_manifest.json"

PHASE_COLORS = {
    "phase1_tokenizer": "#2563eb",
    "phase2_learning_rate": "#16a34a",
    "phase3_capacity": "#c2410c",
    "phase4_regularization": "#9333ea",
    "phase5_confirm": "#0f766e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    return parser.parse_args()


def configure_matplotlib() -> None:
    for font_path in [Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"), Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"), Path("/Library/Fonts/AppleGothic.ttf")]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def completed_results(ledger_path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in read_jsonl(ledger_path):
        result = record.get("result", record)
        if result.get("status") != "completed":
            continue
        result = dict(result)
        result["_result_path"] = record.get("result_path") or result.get("_result_path")
        result["_run_dir"] = record.get("run_dir") or result.get("_run_dir")
        manifest = result.get("cache_vocab_manifest") or {}
        for key in ("train_tokens_per_char", "train_chars_per_token", "val_tokens_per_char", "val_chars_per_token", "actual_vocab_size", "bpe_merge_count"):
            if key in manifest:
                result[key] = manifest[key]
        results.append(result)
    results.sort(key=lambda row: number(row, "run_number"))
    return results


def histories(results: list[dict[str, Any]], runs_dir: Path) -> dict[int, list[dict[str, Any]]]:
    by_run: dict[int, list[dict[str, Any]]] = {}
    for result in results:
        run_number = int(number(result, "run_number"))
        run_dir = Path(str(result.get("_run_dir") or runs_dir / f"run_{run_number:04d}"))
        history_path = run_dir / "history.jsonl"
        rows = read_jsonl(history_path)
        rows.sort(key=lambda row: (number(row, "tokens_seen"), number(row, "step")))
        if rows:
            by_run[run_number] = rows
    return by_run


def number(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_points(results: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    x: list[float] = []
    y: list[float] = []
    for result in results:
        value = number(result, key)
        if value != value:
            continue
        x.append(number(result, "run_number"))
        y.append(value)
    return x, y


def phase_color(result: dict[str, Any]) -> str:
    return PHASE_COLORS.get(str(result.get("phase")), "#64748b")


def annotate_runs(ax: Any, results: list[dict[str, Any]], y_key: str) -> None:
    for result in results:
        y = number(result, y_key)
        if y != y:
            continue
        ax.text(number(result, "run_number") + 0.03, y, str(result.get("condition_id", "")), fontsize=7, color="#334155", alpha=0.85)


def line(ax: Any, results: list[dict[str, Any]], key: str, label: str, color: str, marker: str = "o") -> None:
    x, y = finite_points(results, key)
    if not x:
        return
    ax.plot(x, y, marker=marker, linewidth=1.9, markersize=4, label=label, color=color)
    for result in results:
        value = number(result, key)
        if value == value:
            ax.scatter([number(result, "run_number")], [value], s=40, color=phase_color(result), edgecolor="white", linewidth=0.7, zorder=3)


def cumulative_min(values: Iterable[float]) -> list[float]:
    best = float("inf")
    output: list[float] = []
    for value in values:
        if value == value and value < best:
            best = value
        output.append(best if best < float("inf") else float("nan"))
    return output


def plot_big_dashboard(path: Path, results: list[dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(8, 1, figsize=(18, 26), sharex=True)
    x = [number(row, "run_number") for row in results]
    bits = [number(row, "best_val_bits_per_char") for row in results]
    axes[0].plot(x, bits, marker="o", color="#2563eb", label="best val bits/char")
    axes[0].plot(x, cumulative_min(bits), marker="s", color="#dc2626", label="cumulative best")
    line(axes[1], results, "final_val_bits_per_char", "final val bits/char", "#0f766e")
    line(axes[1], results, "final_train_bits_per_char", "final train bits/char", "#64748b", marker=".")
    line(axes[2], results, "val_tokens_per_char", "val tokens/char", "#9333ea")
    line(axes[2], results, "val_chars_per_token", "val chars/token", "#c2410c", marker="s")
    line(axes[3], results, "final_generalization_gap", "train-val gap", "#be123c")
    line(axes[3], results, "final_minus_best_val_loss", "final-best rebound", "#ea580c", marker="s")
    line(axes[4], results, "parameter_count", "parameter count", "#334155")
    line(axes[4], results, "tokens_seen", "tokens seen", "#0284c7", marker="s")
    line(axes[5], results, "estimated_train_flops", "estimated FLOPs", "#7c3aed")
    line(axes[5], results, "compute_proxy", "parameter_count * tokens_seen", "#0f766e", marker="s")
    line(axes[6], results, "tokens_per_sec_after_warmup", "warm tokens/sec", "#16a34a")
    line(axes[6], results, "elapsed_sec", "elapsed sec", "#ca8a04", marker="s")
    score_results = [dict(result, selection_score=selection_score(result)) for result in results]
    line(axes[7], score_results, "selection_score", "rule selection score", "#111827")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(fontsize=9, loc="best")
        axis.set_ylabel("value")
    axes[0].set_title("NSMC best-fit cumulative linear dashboard")
    axes[-1].set_xlabel("run_number")
    annotate_runs(axes[0], results, "best_val_bits_per_char")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_quality(path: Path, results: list[dict[str, Any]]) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    line(ax, results, "best_val_bits_per_char", "best val bits/char", "#2563eb")
    line(ax, results, "final_val_bits_per_char", "final val bits/char", "#0f766e", marker="s")
    line(ax, results, "final_train_bits_per_char", "final train bits/char", "#64748b", marker=".")
    annotate_runs(ax, results, "best_val_bits_per_char")
    ax.set_title("Quality timeline by physical run")
    ax.set_xlabel("run_number")
    ax.set_ylabel("bits/char")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_tokenizer(path: Path, results: list[dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    line(axes[0], results, "vocab_size", "requested vocab_size", "#2563eb")
    line(axes[0], results, "actual_vocab_size", "actual vocab_size", "#0f766e", marker="s")
    line(axes[1], results, "tokenizer_min_frequency", "min_frequency", "#9333ea")
    line(axes[1], results, "bpe_merge_count", "BPE merge count", "#c2410c", marker="s")
    line(axes[2], results, "val_tokens_per_char", "val tokens/char", "#be123c")
    line(axes[2], results, "val_chars_per_token", "val chars/token", "#16a34a", marker="s")
    axes[0].set_title("Tokenizer efficiency timeline")
    axes[-1].set_xlabel("run_number")
    for axis in axes:
        axis.set_ylabel("value")
        axis.grid(alpha=0.22)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_compute(path: Path, results: list[dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    line(axes[0], results, "parameter_count", "parameter_count", "#334155")
    line(axes[0], results, "tokens_seen", "tokens_seen", "#0284c7", marker="s")
    line(axes[1], results, "compute_proxy", "parameter_count * tokens_seen", "#0f766e")
    line(axes[1], results, "estimated_train_flops", "estimated train FLOPs", "#7c3aed", marker="s")
    line(axes[2], results, "tokens_per_sec_after_warmup", "warm tokens/sec", "#16a34a")
    line(axes[2], results, "elapsed_sec", "elapsed sec", "#ca8a04", marker="s")
    axes[0].set_title("Compute and system efficiency timeline")
    axes[-1].set_xlabel("run_number")
    for axis in axes:
        axis.set_ylabel("value")
        axis.grid(alpha=0.22)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_generalization(path: Path, results: list[dict[str, Any]]) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    line(ax, results, "final_generalization_gap", "final train-val gap", "#be123c")
    line(ax, results, "generalization_gap_delta", "gap delta", "#9333ea", marker="s")
    line(ax, results, "final_minus_best_val_loss", "final-best rebound", "#ea580c", marker=".")
    annotate_runs(ax, results, "final_generalization_gap")
    ax.axhline(0, color="#94a3b8", linewidth=1, linestyle="--")
    ax.set_title("Overfit and rebound timeline")
    ax.set_xlabel("run_number")
    ax.set_ylabel("loss delta")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_history_curves(path: Path, results: list[dict[str, Any]], history_by_run: dict[int, list[dict[str, Any]]]) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)
    for result in results:
        run_number = int(number(result, "run_number"))
        rows = history_by_run.get(run_number, [])
        if not rows:
            continue
        label = f"run {run_number} {result.get('condition_id')}"
        x = [number(row, "tokens_seen") for row in rows]
        axes[0].plot(x, [number(row, "val_bits_per_char") for row in rows], marker="o", linewidth=1.5, markersize=3, label=label, color=phase_color(result))
        axes[1].plot(x, [number(row, "train_bits_per_char") for row in rows], marker=".", linewidth=1.3, label=label, color=phase_color(result), alpha=0.8)
        axes[2].plot(x, [number(row, "tokens_per_sec_after_warmup") for row in rows], marker=".", linewidth=1.3, label=label, color=phase_color(result), alpha=0.8)
    axes[0].set_title("All training histories accumulated by run")
    axes[0].set_ylabel("val bits/char")
    axes[1].set_ylabel("train bits/char")
    axes[2].set_ylabel("tokens/sec")
    axes[2].set_xlabel("tokens_seen within each run")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_hyperparams(path: Path, results: list[dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    line(axes[0], results, "learning_rate", "learning_rate", "#2563eb")
    line(axes[0], results, "weight_decay", "weight_decay", "#0f766e", marker="s")
    line(axes[1], results, "drop_rate", "drop_rate", "#be123c")
    line(axes[1], results, "grad_clip", "grad_clip", "#64748b", marker="s")
    line(axes[2], results, "emb_dim", "emb_dim", "#9333ea")
    line(axes[2], results, "n_layers", "n_layers", "#c2410c", marker="s")
    line(axes[2], results, "n_heads", "n_heads", "#0284c7", marker=".")
    line(axes[3], results, "ffn_mult", "ffn_mult", "#111827")
    line(axes[3], results, "context_length", "context_length", "#16a34a", marker="s")
    axes[0].set_title("Hyperparameter path chosen by the rule engine")
    axes[-1].set_xlabel("run_number")
    for axis in axes:
        axis.set_ylabel("value")
        axis.grid(alpha=0.22)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def write_index(path: Path, figures: list[Path], results: list[dict[str, Any]]) -> None:
    lines = [
        "# NSMC Best-Fit Linear Graph Index",
        "",
        "이 폴더는 `docs/nsmc_bestfit/all_run_results.jsonl`에 누적된 모든 완료 실행을 `run_number` 순서의 선형 그래프로 다시 그린다.",
        "",
        f"- completed runs: `{len(results)}`",
        f"- latest run: `{int(number(results[-1], 'run_number')) if results else ''}`",
        f"- primary view: `00_all_metrics_linear_dashboard.png`",
        "",
    ]
    for figure in figures:
        rel = figure.name
        lines.extend([f"## {figure.stem}", "", f"![{figure.stem}]({rel})", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, figures: list[Path], results: list[dict[str, Any]], ledger_path: Path) -> None:
    payload = {
        "completed_runs": len(results),
        "latest_run_number": int(number(results[-1], "run_number")) if results else None,
        "figures": [str(figure) for figure in figures],
        "source_ledger": str(ledger_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = completed_results(args.ledger)
    history_by_run = histories(results, args.runs_dir)
    figures: list[Path] = []
    if results:
        figures.extend(
            [
                plot_big_dashboard(args.out_dir / "00_all_metrics_linear_dashboard.png", results),
                plot_quality(args.out_dir / "01_quality_timeline.png", results),
                plot_tokenizer(args.out_dir / "02_tokenizer_timeline.png", results),
                plot_compute(args.out_dir / "03_compute_timeline.png", results),
                plot_generalization(args.out_dir / "04_generalization_timeline.png", results),
                plot_history_curves(args.out_dir / "05_training_histories.png", results, history_by_run),
                plot_hyperparams(args.out_dir / "06_hyperparameter_path.png", results),
            ]
        )
    write_index(args.index, figures, results)
    write_manifest(args.manifest, figures, results, args.ledger)
    print(json.dumps({"completed_runs": len(results), "figures": [str(path) for path in figures], "index": str(args.index), "manifest": str(args.manifest)}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
