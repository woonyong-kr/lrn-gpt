from __future__ import annotations

from pathlib import Path
import math
import re

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "docs" / "HY" / "testresult"
FIG_DIR = ROOT / "docs" / "HY" / "figures"


CANONICAL = {
    "E00": "E00_baseline_result.md",
    "E01": "E01_context_length_64_result.md",
    "E02": "E02_context_length_192_result.md",
    "E03": "E03_context_length_256_result.md",
    "E04": "E04_vocab_size_2000_result.md",
    "E05": "E05_vocab_size_4000_result.md",
    "E06": "E06_vocab_size_5000_result.md",
    "E07": "E07_emb_dim_128_result.md",
    "E08": "E08_emb_dim_256_result.md",
    "E09": "E09_n_heads_3_result.md",
    "E10": "E10_n_heads_6_result.md",
    "E11": "E11_n_layers_2_result.md",
    "E12": "E12_n_layers_6_result.md",
    "E13": "E13_n_layers_8_result.md",
    "E14": "E14_ffn_multiplier_2_result.md",
    "E15": "E15_ffn_multiplier_6_result.md",
    "E16": "E16_drop_rate_0p0_result.md",
    "E17": "E17_drop_rate_0p05_result.md",
    "E18": "E18_drop_rate_0p2_result.md",
    "E19": "E19_qkv_bias_True_result.md",
    "E20": "E20_weight_tying_True_result.md",
    "E21": "E21_long_epoch_dropout_0p0_result.md",
    "E22": "E22_long_epoch_dropout_0p05_result.md",
    "E23": "E23_long_epoch_dropout_0p1_result.md",
    "E24": "E24_long_epoch_dropout_0p2_result.md",
    "E25": "E25_activation_ReLU_result.md",
    "E26": "E26_activation_SiLU_result.md",
    "E27": "E27_preLN_result.md",
    "E28": "E28_postLN_baseline_10ep_result.md",
    "E29": "E29_stride_64_result.md",
    "E31": "E31_n_heads_1_10ep_result.md",
    "E32": "E32_n_heads_2_10ep_result.md",
    "E33": "E33_n_heads_8_10ep_result.md",
    "E34": "E34_n_heads_12_10ep_result.md",
    "E35": "E35_ffn_multiplier_2_10ep_result.md",
    "E36": "E36_ffn_multiplier_6_10ep_result.md",
    "E37": "E37_postLN_layers8_10ep_result.md",
    "E38": "E38_preLN_layers8_10ep_result.md",
    "E39": "E39_qkv_bias_true_10ep_result.md",
    "E40": "E40_qkv_bias_true_layers8_10ep_result.md",
    "E41": "E41_weight_tying_true_10ep_result.md",
    "E42": "E42_weight_tying_true_20ep_result.md",
    "E43": "E43_weight_tying_false_50ep_result.md",
    "E44": "E44_weight_tying_true_50ep_result.md",
    "E45": "E45_postLN_layers8_20ep_result.md",
    "E46": "E46_preLN_layers8_20ep_result.md",
    "E47": "E47_postLN_layers12_20ep_result.md",
    "E48": "E48_preLN_layers12_20ep_result.md",
    "E49": "E49_activation_GELU_depth8_preLN_seed42_20ep_result.md",
    "E50": "E50_activation_ReLU_depth8_preLN_seed42_20ep_result.md",
    "E51": "E51_activation_SiLU_depth8_preLN_seed42_20ep_result.md",
    "E52": "E52_activation_GELU_depth8_preLN_seed123_20ep_result.md",
    "E53": "E53_activation_ReLU_depth8_preLN_seed123_20ep_result.md",
    "E54": "E54_activation_SiLU_depth8_preLN_seed123_20ep_result.md",
    "E55": "E55_activation_GELU_depth8_preLN_seed2026_20ep_result.md",
    "E56": "E56_activation_ReLU_depth8_preLN_seed2026_20ep_result.md",
    "E57": "E57_activation_SiLU_depth8_preLN_seed2026_20ep_result.md",
    "E58": "E58_FinalA_vocab2000_result.md",
    "E59": "E59_FinalA_vocab3000_result.md",
    "E60": "E60_postLN_layers8_drop0p2_lr0p0002_result.md",
    "E61": "E61_preLN_layers8_drop0p2_lr0p0002_result.md",
    "E62": "E62_postLN_layers12_drop0p2_lr0p0002_result.md",
    "E63": "E63_preLN_layers12_drop0p2_lr0p0002_result.md",
}


LEGEND_LABELS = {
    "E00": "baseline: n_layers=4, 5ep",
    "E23": "baseline: 4L post-LN, 20ep",
    "E25": "activation=ReLU",
    "E26": "activation=SiLU",
    "E27": "pre-LN, 4L, 10ep",
    "E28": "baseline: 4L post-LN, 10ep",
    "E31": "n_heads=1",
    "E32": "n_heads=2",
    "E33": "n_heads=8",
    "E34": "n_heads=12",
    "E35": "ffn_multiplier=2",
    "E36": "ffn_multiplier=6",
    "E37": "post-LN, 8L, 10ep",
    "E38": "pre-LN, 8L, 10ep",
    "E39": "qkv_bias=True, 4L",
    "E40": "qkv_bias=True, 8L",
    "E41": "weight_tying=True, 10ep",
    "E42": "weight_tying=True, 20ep",
    "E43": "weight_tying=False, 50ep",
    "E44": "weight_tying=True, 50ep",
    "E45": "post-LN, 8L, 20ep",
    "E46": "pre-LN, 8L, 20ep",
    "E47": "post-LN, 12L, 20ep",
    "E48": "pre-LN, 12L, 20ep",
    "E49": "GELU, seed=42",
    "E50": "ReLU, seed=42",
    "E51": "SiLU, seed=42",
    "E52": "GELU, seed=123",
    "E53": "ReLU, seed=123",
    "E54": "SiLU, seed=123",
    "E55": "GELU, seed=2026",
    "E56": "ReLU, seed=2026",
    "E57": "SiLU, seed=2026",
    "E58": "Final-A, vocab=2000",
    "E59": "Final-A, vocab=3000",
    "E60": "post-LN, 8L, drop=0.2, lr=0.0002",
    "E61": "pre-LN, 8L, drop=0.2, lr=0.0002",
    "E62": "post-LN, 12L, drop=0.2, lr=0.0002",
    "E63": "pre-LN, 12L, drop=0.2, lr=0.0002",
}


def legend_label(exp_id: str) -> str:
    return LEGEND_LABELS.get(exp_id, exp_id)


def parse_value(value: str):
    value = value.strip().replace(",", "")
    if value in {"True", "False"}:
        return value == "True"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_result(exp_id: str, filename: str) -> dict:
    path = RESULT_DIR / filename
    text = path.read_text(encoding="utf-8")
    row = {"id": exp_id, "file": str(path), "title": text.splitlines()[0].lstrip("# ").strip()}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        key, value = cells[0], cells[1]
        if key in {"---", "항목", "지표", "epoch"}:
            continue
        row[key] = parse_value(value)
    if "final_test_loss" not in row and "final_val_loss" in row:
        row["final_test_loss"] = row["final_val_loss"]
    if "final_loss_gap" not in row and "final_test_loss" in row and "final_train_loss" in row:
        row["final_loss_gap"] = row["final_test_loss"] - row["final_train_loss"]
    return row


def load_results() -> dict[str, dict]:
    return {exp_id: read_result(exp_id, filename) for exp_id, filename in CANONICAL.items()}


def style_ax(ax):
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")


def save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def bar_by_setting(results, ids, x_key, name, title, xlabel, baseline_id):
    rows = sorted([results[i] for i in ids], key=lambda row: row[x_key])
    best = min(rows, key=lambda row: row["final_val_loss"])
    labels = []
    colors = []
    edgecolors = []
    for row in rows:
        label = str(row[x_key])
        if row["id"] == baseline_id:
            label += "\nbaseline"
        labels.append(label)
        colors.append("#f59e0b" if row["id"] == best["id"] else "#cbd5e1")
        edgecolors.append("#111827" if row["id"] == baseline_id else "#64748b")
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bars = ax.bar(labels, [row["final_val_loss"] for row in rows], color=colors, edgecolor=edgecolors, linewidth=1.6)
    for bar, row in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006, f"{row['final_val_loss']:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("final validation loss")
    ax.text(0.99, 0.02, "orange = best, outlined = baseline", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#475569")
    style_ax(ax)
    save(fig, name)


def grouped_bars(groups, name, title, ylabel="final validation loss"):
    labels = [group[0] for group in groups]
    rows = [group[1] for group in groups]
    best = min(rows, key=lambda row: row["final_val_loss"])
    colors = ["#f59e0b" if row["id"] == best["id"] else "#cbd5e1" for row in rows]
    edges = ["#111827" if "baseline" in label.lower() else "#64748b" for label in labels]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bars = ax.bar(labels, [row["final_val_loss"] for row in rows], color=colors, edgecolor=edges, linewidth=1.5)
    for bar, row in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006, f"{row['final_val_loss']:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.text(0.99, 0.02, "orange = best, outlined = baseline", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#475569")
    style_ax(ax)
    save(fig, name)


def curve_rows(results, exp_id: str) -> list[dict[str, float]]:
    path = Path(results[exp_id]["file"])
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```csv\n(.*?)\n```", text, re.S)
    if not match:
        return []
    lines = match.group(1).strip().splitlines()
    header = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.split(",")]
        raw = dict(zip(header, cells))
        row = {}
        for key, value in raw.items():
            if value == "":
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
        rows.append(row)
    return rows


def line_curves(results, ids, name, title):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for exp_id in ids:
        rows = curve_rows(results, exp_id)
        xs = [row["epoch"] for row in rows]
        ys = [row["val_loss"] for row in rows]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.4, label=legend_label(exp_id))
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss")
    ax.legend(fontsize=7, loc="upper right")
    style_ax(ax)
    save(fig, name)


def weight_tying_train_val_gap(results):
    labels = {
        "E43": "weight_tying=False",
        "E44": "weight_tying=True",
    }
    colors = {
        "E43": "#2563eb",
        "E44": "#f59e0b",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for exp_id, label in labels.items():
        rows = curve_rows(results, exp_id)
        xs = [row["epoch"] for row in rows]
        train = [row["train_loss"] for row in rows]
        val = [row["val_loss"] for row in rows]
        gap = [row["loss_gap"] for row in rows]
        axes[0].plot(xs, train, linestyle="--", linewidth=1.4, color=colors[exp_id], label=f"{label}: train")
        axes[0].plot(xs, val, linestyle="-", linewidth=1.8, color=colors[exp_id], label=f"{label}: val")
        axes[1].plot(xs, gap, marker="o", markersize=3, linewidth=1.6, color=colors[exp_id], label=f"{label}: gap")
    axes[0].set_title("train vs validation loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=7, loc="upper right")
    style_ax(axes[0])
    axes[1].set_title("validation - train loss gap")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss gap")
    axes[1].legend(fontsize=7, loc="upper right")
    style_ax(axes[1])
    fig.suptitle("weight_tying=False baseline vs True at 50 epochs")
    save(fig, "50_weight_tying_final_curves.png")


def activation_seed_summary(results):
    groups = {
        "GELU": ["E49", "E52", "E55"],
        "ReLU": ["E50", "E53", "E56"],
        "SiLU": ["E51", "E54", "E57"],
    }
    colors = {"GELU": "#f59e0b", "ReLU": "#2563eb", "SiLU": "#64748b"}
    metrics = [
        ("final_val_loss", "final validation loss"),
        ("final_loss_gap", "validation - train gap"),
        ("final_train_loss", "final train loss"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    labels = list(groups)
    for ax, (key, title) in zip(axes, metrics):
        means = []
        stds = []
        for label in labels:
            values = [results[exp_id][key] for exp_id in groups[label]]
            means.append(sum(values) / len(values))
            mean = means[-1]
            stds.append((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)
        bars = ax.bar(labels, means, yerr=stds, capsize=4, color=[colors[label] for label in labels], edgecolor="#334155", linewidth=1.2)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means) * 0.006, f"{mean:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("activation")
        style_ax(ax)
    fig.suptitle("activation seed repeat summary: n_layers=8, pre-LN, 20 epochs")
    save(fig, "51_activation_seed_summary.png")


def activation_seed_curves(results):
    groups = {
        "GELU": ["E49", "E52", "E55"],
        "ReLU": ["E50", "E53", "E56"],
        "SiLU": ["E51", "E54", "E57"],
    }
    colors = {"GELU": "#f59e0b", "ReLU": "#2563eb", "SiLU": "#64748b"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for label, ids in groups.items():
        series = [curve_rows(results, exp_id) for exp_id in ids]
        epochs = [row["epoch"] for row in series[0]]
        for metric, ax in [("val_loss", axes[0]), ("loss_gap", axes[1])]:
            means = []
            for idx in range(len(epochs)):
                values = [rows[idx][metric] for rows in series]
                means.append(sum(values) / len(values))
            ax.plot(epochs, means, marker="o", markersize=3, linewidth=1.8, color=colors[label], label=f"{label}: seed mean")
    axes[0].set_title("validation loss")
    axes[0].set_ylabel("loss")
    axes[1].set_title("validation - train gap")
    axes[1].set_ylabel("loss gap")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7, loc="upper right")
        style_ax(ax)
    fig.suptitle("activation mean curves across seeds")
    save(fig, "52_activation_seed_curves.png")


def final_candidate_summary(results):
    ids = ["E58", "E59"]
    labels = ["vocab=2000", "vocab=3000"]
    metrics = [
        ("final_val_loss", "final validation loss"),
        ("best_val_loss", "best validation loss"),
        ("final_loss_gap", "validation - train gap"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, (key, title) in zip(axes, metrics):
        values = [results[exp_id][key] for exp_id in ids]
        best_idx = values.index(min(values))
        colors = ["#f59e0b" if idx == best_idx else "#cbd5e1" for idx in range(len(ids))]
        bars = ax.bar(labels, values, color=colors, edgecolor=["#111827", "#64748b"], linewidth=1.4)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Final-A tokenizer")
        style_ax(ax)
    fig.suptitle("Final-A candidate comparison")
    save(fig, "53_final_candidate_summary.png")


def final_candidate_curves(results):
    colors = {"E58": "#f59e0b", "E59": "#2563eb"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for exp_id in ["E58", "E59"]:
        rows = curve_rows(results, exp_id)
        xs = [row["epoch"] for row in rows]
        train = [row["train_loss"] for row in rows]
        val = [row["val_loss"] for row in rows]
        gap = [row["loss_gap"] for row in rows]
        label = legend_label(exp_id)
        axes[0].plot(xs, train, linestyle="--", linewidth=1.4, color=colors[exp_id], label=f"{label}: train")
        axes[0].plot(xs, val, linestyle="-", linewidth=1.8, color=colors[exp_id], label=f"{label}: val")
        axes[1].plot(xs, gap, marker="o", markersize=3, linewidth=1.6, color=colors[exp_id], label=f"{label}: gap")
    axes[0].set_title("train vs validation loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[1].set_title("validation - train loss gap")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss gap")
    for ax in axes:
        ax.legend(fontsize=7, loc="upper right")
        style_ax(ax)
    fig.suptitle("Final-A train/validation behavior")
    save(fig, "54_final_candidate_curves.png")


def stride_metrics(results):
    rows = [results["E28"], results["E29"]]
    labels = ["128\nbaseline", "64"]
    metrics = [
        ("final_val_loss", "final validation loss"),
        ("elapsed_sec", "elapsed seconds"),
        ("final_loss_gap", "loss gap"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    for ax, (key, ylabel) in zip(axes, metrics):
        values = [row[key] for row in rows]
        best_idx = values.index(min(values))
        colors = ["#f59e0b" if idx == best_idx else "#cbd5e1" for idx in range(len(rows))]
        bars = ax.bar(labels, values, color=colors, edgecolor=["#111827", "#64748b"], linewidth=1.5)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(ylabel)
        ax.set_xlabel("stride")
        style_ax(ax)
    fig.suptitle("stride overlap trade-off")
    save(fig, "45_stride_metrics.png")


def norm_depth_regularized(results):
    ids = ["E60", "E61", "E62", "E63"]
    labels = ["8L\npost-LN", "8L\npre-LN", "12L\npost-LN", "12L\npre-LN"]
    rows = [results[exp_id] for exp_id in ids]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    colors = ["#94a3b8", "#f59e0b", "#94a3b8", "#f59e0b"]

    axes[0].bar(labels, [row["final_val_loss"] for row in rows], color=colors, edgecolor="#475569", linewidth=1.2)
    axes[0].set_title("regularized depth/norm final validation loss")
    axes[0].set_ylabel("final validation loss")
    for idx, row in enumerate(rows):
        axes[0].text(idx, row["final_val_loss"] + 0.005, f"{row['final_val_loss']:.4f}", ha="center", va="bottom", fontsize=8)
    style_ax(axes[0])

    axes[1].bar(labels, [row["final_loss_gap"] for row in rows], color=colors, edgecolor="#475569", linewidth=1.2)
    axes[1].axhline(0.5, color="#ef4444", linestyle="--", linewidth=1.1, label="gap=0.5")
    axes[1].set_title("regularized depth/norm train-validation gap")
    axes[1].set_ylabel("final val - train loss")
    axes[1].legend(fontsize=8)
    for idx, row in enumerate(rows):
        axes[1].text(idx, row["final_loss_gap"] + 0.01, f"{row['final_loss_gap']:.3f}", ha="center", va="bottom", fontsize=8)
    style_ax(axes[1])

    save(fig, "62_regularized_norm_depth_metrics.png")


def overview(results):
    ids = ["E00", "E01", "E02", "E03", "E07", "E08", "E11", "E12", "E13", "E16", "E17", "E18", "E19", "E20"]
    rows = [results[i] for i in ids]
    best = min(rows, key=lambda row: row["final_val_loss"])
    colors = ["#f59e0b" if row["id"] == best["id"] else "#cbd5e1" for row in rows]
    edges = ["#111827" if row["id"] == "E00" else "#64748b" for row in rows]
    labels = [f"{row['id']}\n{'baseline' if row['id']=='E00' else ''}" for row in rows]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.bar(labels, [row["final_val_loss"] for row in rows], color=colors, edgecolor=edges, linewidth=1.3)
    ax.axhline(results["E00"]["final_val_loss"], color="#475569", linestyle="--", linewidth=1, label="baseline: E00 context=128, 5ep")
    ax.set_title("5-epoch comparable experiments")
    ax.set_ylabel("final validation loss")
    ax.legend(fontsize=7, loc="upper right")
    style_ax(ax)
    save(fig, "20_overview_final_val_loss.png")


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    results = load_results()
    overview(results)
    bar_by_setting(results, ["E01", "E00", "E02", "E03"], "context_length", "22_context_length_metrics.png", "context_length sweep", "context_length", "E00")
    bar_by_setting(results, ["E04", "E00", "E05", "E06"], "vocab_size", "24_vocab_size_metrics.png", "vocab_size sweep", "vocab_size", "E00")
    bar_by_setting(results, ["E07", "E00", "E08"], "emb_dim", "25_emb_dim_metrics.png", "emb_dim sweep", "emb_dim", "E00")
    bar_by_setting(results, ["E31", "E32", "E28", "E33", "E34"], "n_heads", "26_n_heads_metrics.png", "n_heads sweep at 10 epochs", "n_heads", "E28")
    bar_by_setting(results, ["E11", "E00", "E12", "E13"], "n_layers", "27_n_layers_metrics.png", "n_layers sweep at 5 epochs", "n_layers", "E00")
    bar_by_setting(results, ["E35", "E28", "E36"], "ffn_mult", "28_ffn_multiplier_metrics.png", "ffn_multiplier sweep at 10 epochs", "ffn_multiplier", "E28")
    bar_by_setting(results, ["E16", "E17", "E00", "E18"], "drop_rate", "29_drop_rate_short_metrics.png", "drop_rate sweep at 5 epochs", "drop_rate", "E00")
    grouped_bars(
        [
            ("4L False\nbaseline", results["E28"]),
            ("4L True", results["E39"]),
            ("8L False\nbaseline", results["E37"]),
            ("8L True", results["E40"]),
        ],
        "30_qkv_bias_metrics.png",
        "qkv_bias under 4-layer and 8-layer settings",
    )
    grouped_bars(
        [
            ("10ep False\nbaseline", results["E28"]),
            ("10ep True", results["E41"]),
            ("20ep False\nbaseline", results["E23"]),
            ("20ep True", results["E42"]),
        ],
        "31_weight_tying_metrics.png",
        "weight_tying under more stable training",
    )
    grouped_bars(
        [
            ("GELU\nbaseline", results["E28"]),
            ("ReLU", results["E25"]),
            ("SiLU", results["E26"]),
        ],
        "32_activation_metrics.png",
        "FFN activation at 10 epochs",
    )
    line_curves(results, ["E28", "E25", "E26"], "33_activation_curves.png", "activation validation curves")
    activation_seed_summary(results)
    activation_seed_curves(results)
    grouped_bars(
        [
            ("4L post\nbaseline", results["E28"]),
            ("4L pre", results["E27"]),
            ("8L post\nbaseline", results["E37"]),
            ("8L pre", results["E38"]),
        ],
        "39_norm_position_metrics.png",
        "LayerNorm position by depth",
    )
    line_curves(results, ["E31", "E32", "E28", "E33", "E34"], "40_n_heads_curves.png", "n_heads validation curves at 10 epochs")
    line_curves(results, ["E35", "E28", "E36"], "41_ffn_curves.png", "ffn_multiplier validation curves at 10 epochs")
    line_curves(results, ["E37", "E38"], "42_norm_depth8_curves.png", "8-layer norm position validation curves")
    line_curves(results, ["E28", "E39", "E37", "E40"], "43_qkv_bias_curves.png", "qkv_bias validation curves")
    line_curves(results, ["E28", "E41", "E23", "E42"], "44_weight_tying_curves.png", "weight_tying validation curves")
    grouped_bars(
        [
            ("20ep False\nbaseline", results["E23"]),
            ("20ep True", results["E42"]),
            ("50ep False\nbaseline", results["E43"]),
            ("50ep True", results["E44"]),
        ],
        "45_weight_tying_50ep_metrics.png",
        "weight_tying long-run generalization",
    )
    line_curves(results, ["E23", "E42", "E43", "E44"], "46_weight_tying_50ep_curves.png", "weight_tying validation curves through 50 epochs")
    grouped_bars(
        [
            ("False\n50ep baseline", results["E43"]),
            ("True\n50ep", results["E44"]),
        ],
        "49_weight_tying_final_metrics.png",
        "weight_tying final comparison at 50 epochs",
    )
    weight_tying_train_val_gap(results)
    grouped_bars(
        [
            ("4L post\n20ep baseline", results["E23"]),
            ("8L post\n20ep", results["E45"]),
            ("8L pre\n20ep", results["E46"]),
            ("12L post\n20ep", results["E47"]),
            ("12L pre\n20ep", results["E48"]),
        ],
        "47_n_layers_norm_20ep_metrics.png",
        "n_layers and norm position at 20 epochs",
    )
    line_curves(results, ["E23", "E45", "E46", "E47", "E48"], "48_n_layers_norm_20ep_curves.png", "n_layers and norm validation curves at 20 epochs")
    norm_depth_regularized(results)
    line_curves(results, ["E60", "E61", "E62", "E63"], "63_regularized_norm_depth_curves.png", "regularized n_layers and norm validation curves")
    final_candidate_summary(results)
    final_candidate_curves(results)
    stride_metrics(results)
    print(FIG_DIR)


if __name__ == "__main__":
    main()
