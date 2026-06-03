# -*- coding: utf-8 -*-
"""Generate sequential line figures from HY LLM relog metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from hy_llm_relog_common import FIGURE_DIR, RELOG_DIR, ROOT, number, read_csv


def configure_matplotlib() -> None:
    for font_path in [
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
        Path("/Library/Fonts/AppleGothic.ttf"),
    ]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=str(RELOG_DIR / "report_llm_metrics.csv"))
    parser.add_argument("--history", default=str(RELOG_DIR / "report_llm_metrics_by_step.csv"))
    parser.add_argument("--out", default=str(FIGURE_DIR))
    args = parser.parse_args()

    configure_matplotlib()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_csv(Path(args.metrics))
    history = read_csv(Path(args.history))
    by_id = {str(row["experiment_id"]): row for row in metrics}
    history_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in history:
        history_by_id[str(row["experiment_id"])].append(row)
    for rows in history_by_id.values():
        rows.sort(key=lambda row: (number(row, "tokens_seen"), number(row, "step")))

    figures = [
        plot_learning_curve(out_dir / "01_context_length_learning_curve.png", ["E01", "E00", "E02", "E03"], by_id, history_by_id, "context_length", "Context length learning curve", "tokens_seen", "val_loss"),
        plot_vocab(out_dir / "02_vocab_bits_per_char_learning_curve.png", ["E04", "E00", "E05", "E06"], by_id, history_by_id),
        plot_capacity(out_dir / "03_emb_dim_capacity_path.png", ["E07", "E00", "E08"], by_id, history_by_id, "emb_dim", "Embedding dimension capacity path"),
        plot_learning_curve(out_dir / "04_n_heads_learning_curve.png", ["E31", "E32", "E28", "E33", "E34"], by_id, history_by_id, "n_heads", "Attention heads learning curve", "tokens_seen", "val_loss"),
        plot_learning_curve(out_dir / "05_n_layers_depth_curve.png", ["E11", "E00", "E12", "E13", "E37", "E45", "E47"], by_id, history_by_id, "n_layers", "Depth stability curve", "tokens_seen", "val_loss"),
        plot_learning_curve(out_dir / "06_norm_depth_stability_curve.png", ["E28", "E27", "E37", "E38", "E45", "E46", "E47", "E48", "E60", "E61", "E62", "E63"], by_id, history_by_id, norm_depth_label, "Norm/depth stability curve", "tokens_seen", "val_loss"),
        plot_capacity(out_dir / "07_ffn_multiplier_capacity_path.png", ["E35", "E28", "E36"], by_id, history_by_id, "ffn_mult", "FFN multiplier compute path"),
        plot_activation_seed(out_dir / "08_activation_seed_mean_std_curve.png", ["E49", "E50", "E51", "E52", "E53", "E54", "E55", "E56", "E57"], by_id, history_by_id),
        plot_learning_curve(out_dir / "09_dropout_overfit_curve.png", ["E16", "E17", "E00", "E18", "E21", "E22", "E23", "E24"], by_id, history_by_id, dropout_label, "Dropout overfit curve", "tokens_seen", "val_loss"),
        plot_learning_curve(out_dir / "10_qkv_bias_curve.png", ["E28", "E39", "E37", "E40"], by_id, history_by_id, qkv_label, "QKV bias curve", "tokens_seen", "val_loss"),
        plot_learning_curve(out_dir / "11_weight_tying_long_curve.png", ["E28", "E41", "E42", "E43", "E44"], by_id, history_by_id, weight_tying_label, "Weight tying long-run curve", "tokens_seen", "val_loss"),
        plot_stride(out_dir / "12_stride_overlap_curve.png", ["E28", "E29"], by_id, history_by_id),
        plot_learning_curve(out_dir / "13_epoch_overfit_rebound_audit.png", ["E21", "E22", "E23", "E24", "E43", "E44"], by_id, history_by_id, epoch_audit_label, "Epoch overfit rebound audit", "tokens_seen", "val_loss"),
        plot_all_compute(out_dir / "14_loss_vs_compute_all_paths.png", metrics, history_by_id),
    ]
    figure_index = out_dir / "figure_index.md"
    figure_index.write_text(render_index([path for path in figures if path]), encoding="utf-8")
    print(f"wrote {len([path for path in figures if path])} figures to {out_dir}")
    print(f"wrote {figure_index}")


def label_from(metric: dict[str, Any], label_key: str | Callable[[dict[str, Any]], str]) -> str:
    if callable(label_key):
        return label_key(metric)
    return f"{label_key}={metric.get(label_key, '')}"


def norm_depth_label(metric: dict[str, Any]) -> str:
    norm = "pre" if str(metric.get("norm_first")) == "True" else "post"
    return f"L{metric.get('n_layers')}-{norm}-d{metric.get('drop_rate')}-ep{metric.get('num_epochs')}"


def dropout_label(metric: dict[str, Any]) -> str:
    return f"drop={metric.get('drop_rate')},ep={metric.get('num_epochs')}"


def qkv_label(metric: dict[str, Any]) -> str:
    return f"L{metric.get('n_layers')},qkv={metric.get('qkv_bias')}"


def weight_tying_label(metric: dict[str, Any]) -> str:
    return f"tie={metric.get('weight_tying')},ep={metric.get('num_epochs')}"


def epoch_audit_label(metric: dict[str, Any]) -> str:
    if str(metric.get("weight_tying")) in {"True", "False"} and number(metric, "num_epochs") == 50:
        return weight_tying_label(metric)
    return dropout_label(metric)


def plot_learning_curve(
    path: Path,
    ids: list[str],
    by_id: dict[str, dict[str, Any]],
    history_by_id: dict[str, list[dict[str, Any]]],
    label_key: str | Callable[[dict[str, Any]], str],
    title: str,
    x_key: str,
    y_key: str,
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10), sharex=True)
    for experiment_id in ids:
        metric = by_id.get(experiment_id)
        rows = history_by_id.get(experiment_id, [])
        if not metric or not rows:
            continue
        label = f"{experiment_id} {label_from(metric, label_key)}"
        x = [number(row, x_key) for row in rows]
        axes[0].plot(x, [number(row, y_key) for row in rows], marker="o", linewidth=1.7, markersize=3, label=label)
        axes[1].plot(x, [number(row, "train_val_gap") for row in rows], marker=".", linewidth=1.4, label=label)
        axes[2].plot(x, [number(row, "val_loss_minus_best_so_far") for row in rows], marker=".", linewidth=1.4, label=label)
        best_row = min(rows, key=lambda row: number(row, "val_loss", float("inf")))
        axes[0].scatter([number(best_row, x_key)], [number(best_row, y_key)], s=70, edgecolor="black", zorder=5)
    axes[0].set_title(title)
    axes[0].set_ylabel(y_key)
    axes[1].set_ylabel("train-val gap")
    axes[2].set_ylabel("rebound from best")
    axes[2].set_xlabel(x_key)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_vocab(path: Path, ids: list[str], by_id: dict[str, dict[str, Any]], history_by_id: dict[str, list[dict[str, Any]]]) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10), sharex=False)
    for experiment_id in ids:
        metric = by_id[experiment_id]
        rows = history_by_id[experiment_id]
        label = f"{experiment_id} vocab={metric.get('vocab_size')}"
        x = [number(row, "estimated_chars_seen") for row in rows]
        axes[0].plot(x, [number(row, "val_bits_per_char") for row in rows], marker="o", linewidth=1.8, markersize=3, label=label)
        axes[1].plot(x, [number(row, "val_loss") for row in rows], marker=".", linewidth=1.4, label=label)
    vocab_x = [number(by_id[experiment_id], "vocab_size") for experiment_id in ids]
    axes[2].plot(vocab_x, [number(by_id[experiment_id], "val_tokens_per_char") for experiment_id in ids], marker="o", label="val tokens/char")
    axes[2].plot(vocab_x, [number(by_id[experiment_id], "final_val_bits_per_char") for experiment_id in ids], marker="s", label="final bits/char")
    axes[0].set_title("Vocab size fair comparison: bits/char over training")
    axes[0].set_ylabel("val bits/char")
    axes[1].set_ylabel("token-level val loss")
    axes[1].set_xlabel("estimated chars seen")
    axes[2].set_xlabel("vocab size")
    axes[2].set_ylabel("final scale")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_capacity(path: Path, ids: list[str], by_id: dict[str, dict[str, Any]], history_by_id: dict[str, list[dict[str, Any]]], label_key: str, title: str) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10), sharex=True)
    for experiment_id in ids:
        metric = by_id[experiment_id]
        rows = history_by_id[experiment_id]
        label = f"{experiment_id} {label_key}={metric.get(label_key)}"
        x = [number(row, "compute_proxy_param_tokens") for row in rows]
        axes[0].plot(x, [number(row, "val_bits_per_char") for row in rows], marker="o", linewidth=1.8, markersize=3, label=label)
        axes[1].plot(x, [number(row, "train_val_gap") for row in rows], marker=".", linewidth=1.4, label=label)
        axes[2].plot(x, [number(row, "tokens_per_sec_after_warmup") for row in rows], marker=".", linewidth=1.4, label=label)
    axes[0].set_title(title)
    axes[0].set_ylabel("val bits/char")
    axes[1].set_ylabel("train-val gap")
    axes[2].set_ylabel("tokens/sec")
    axes[2].set_xlabel("parameter_count * tokens_seen")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_activation_seed(path: Path, ids: list[str], by_id: dict[str, dict[str, Any]], history_by_id: dict[str, list[dict[str, Any]]]) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8), sharex=True)
    groups: dict[str, list[str]] = defaultdict(list)
    for experiment_id in ids:
        groups[str(by_id[experiment_id].get("activation"))].append(experiment_id)
    for activation, activation_ids in sorted(groups.items()):
        aligned: dict[float, list[float]] = defaultdict(list)
        for experiment_id in activation_ids:
            rows = history_by_id[experiment_id]
            x = [number(row, "tokens_seen") for row in rows]
            y = [number(row, "val_bits_per_char") for row in rows]
            axes[0].plot(x, y, alpha=0.28, linewidth=1.0, label=f"{experiment_id} {activation}")
            for row in rows:
                aligned[number(row, "tokens_seen")].append(number(row, "val_bits_per_char"))
        xs = sorted(aligned)
        means = [sum(aligned[x]) / len(aligned[x]) for x in xs]
        stds = [(sum((v - means[idx]) ** 2 for v in aligned[x]) / len(aligned[x])) ** 0.5 for idx, x in enumerate(xs)]
        axes[0].plot(xs, means, linewidth=2.4, marker="o", label=f"{activation} mean")
        axes[0].fill_between(xs, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=0.12)
        final_values = [number(by_id[experiment_id], "final_val_bits_per_char") for experiment_id in activation_ids]
        axes[1].bar([activation], [sum(final_values) / len(final_values)], yerr=[(sum((v - sum(final_values) / len(final_values)) ** 2 for v in final_values) / len(final_values)) ** 0.5], capsize=4)
    axes[0].set_title("Activation seed mean/std curve")
    axes[0].set_ylabel("val bits/char")
    axes[1].set_ylabel("final mean bits/char")
    axes[0].set_xlabel("tokens_seen")
    axes[0].grid(alpha=0.25)
    axes[1].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_stride(path: Path, ids: list[str], by_id: dict[str, dict[str, Any]], history_by_id: dict[str, list[dict[str, Any]]]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for experiment_id in ids:
        metric = by_id[experiment_id]
        rows = history_by_id[experiment_id]
        label = f"{experiment_id} stride={metric.get('stride')}"
        axes[0][0].plot([number(row, "tokens_seen") for row in rows], [number(row, "val_loss") for row in rows], marker="o", label=label)
        axes[0][1].plot([number(row, "estimated_chars_seen") for row in rows], [number(row, "val_loss") for row in rows], marker="o", label=label)
        axes[1][0].plot([number(row, "tokens_seen") for row in rows], [number(row, "train_val_gap") for row in rows], marker=".", label=label)
        axes[1][1].plot([number(row, "tokens_seen") for row in rows], [number(row, "tokens_per_sec_after_warmup") for row in rows], marker=".", label=label)
    titles = ["val loss vs tokens_seen", "val loss vs estimated_chars_seen", "gap vs tokens_seen", "throughput vs tokens_seen"]
    ylabels = ["val loss", "val loss", "gap", "tokens/sec"]
    for axis, title, ylabel in zip(axes.flatten(), titles, ylabels):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Stride overlap audit", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_all_compute(path: Path, metrics: list[dict[str, Any]], history_by_id: dict[str, list[dict[str, Any]]]) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 7))
    for metric in metrics:
        experiment_id = str(metric["experiment_id"])
        rows = history_by_id.get(experiment_id, [])
        if not rows:
            continue
        ax.plot(
            [number(row, "compute_proxy_param_tokens") for row in rows],
            [number(row, "val_bits_per_char") for row in rows],
            linewidth=1.0,
            alpha=0.55,
        )
        ax.scatter([number(metric, "compute_proxy_param_tokens")], [number(metric, "final_val_bits_per_char")], s=18, alpha=0.75)
        if experiment_id in {"E00", "E06", "E24", "E44", "E46", "E48", "E62"}:
            ax.annotate(experiment_id, (number(metric, "compute_proxy_param_tokens"), number(metric, "final_val_bits_per_char")), fontsize=8)
    ax.set_title("All HY experiments: validation bits/char vs compute path")
    ax.set_xlabel("parameter_count * tokens_seen")
    ax.set_ylabel("val bits/char")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def render_index(figures: list[Path]) -> str:
    lines = ["# HY LLM Sequential Figure Index", ""]
    for figure in figures:
        lines.append(f"- `{figure.name}`")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
