#!/usr/bin/env python3
"""Rewrite the isolated LLM 10x interim report with LLM developer metrics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PLOT_AVAILABLE = True
except ModuleNotFoundError:
    plt = None
    np = None
    PLOT_AVAILABLE = False

from llm_10x_screen_ready_report import (
    PHASE_COLORS,
    PHASE_LABELS,
    available_tokenizer_vocab_sizes,
    fmt_float,
    fmt_int,
    fmt_pct,
    meaningful_phases,
    percentile,
    read_json,
    read_ledger,
    read_optional_json,
    read_summary,
    ready_rows,
    relative_to_report,
    text_corpus_stats,
    to_float,
    to_int,
    tokenizer_diagnostic,
)


LN2 = math.log(2.0)
TRAINING_FLOPS_FACTOR = 6.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=Path("docs/llm_10x/aggregate_summary.csv"))
    parser.add_argument("--meta-json", type=Path, default=Path("docs/llm_10x/aggregate_meta.json"))
    parser.add_argument("--all-results-jsonl", type=Path, default=Path("docs/llm_10x/all_run_results.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("docs/llm_10x/screen_ready_report.md"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/llm_10x/llm_developer_figures"))
    parser.add_argument("--cache-dir", type=Path, default=Path("local/llm_10x_isolated/cache"))
    parser.add_argument("--runs-dir", type=Path, default=Path("local/llm_10x_isolated/runs"))
    parser.add_argument("--queue-status-json", type=Path, default=Path("local/llm_10x_isolated/queue_status.json"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path("data/obsidian_llm_10x_manifest.json"))
    parser.add_argument("--dataset-comparison", type=Path, default=Path("data/obsidian_llm_10x_comparison.json"))
    parser.add_argument("--train-text", type=Path, default=Path("data/obsidian_llm_10x_lm_train.txt"))
    parser.add_argument("--val-text", type=Path, default=Path("data/obsidian_llm_10x_lm_val.txt"))
    return parser.parse_args()


def finite(values: list[float]) -> list[float]:
    return [value for value in values if value == value and math.isfinite(value)]


def median(values: list[float]) -> float:
    cleaned = finite(values)
    return statistics.median(cleaned) if cleaned else math.nan


def iqr(values: list[float]) -> float:
    cleaned = finite(values)
    if not cleaned:
        return math.nan
    return percentile(cleaned, 0.75) - percentile(cleaned, 0.25)


def metric_list(results: list[dict[str, Any]], key: str) -> list[float]:
    return finite([to_float(result.get(key)) for result in results])


def raw_results_by_condition(ledger_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in ledger_rows:
        result = row.get("result", {})
        if result.get("status") != "completed":
            continue
        condition_id = str(result.get("condition_id") or row.get("condition_id") or "")
        if not condition_id:
            continue
        grouped.setdefault(condition_id, []).append(result)
    return grouped


def completed_raw_results(ledger_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in ledger_rows:
        result = row.get("result", {})
        if result.get("status") == "completed":
            rows.append(result)
    return rows


def checkpoint_status(runs_dir: Path, completed_results: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoint_files = list(runs_dir.glob("run_*/checkpoints/*.pt"))
    completed_run_numbers = {to_int(row.get("run_number")) for row in completed_results}
    checkpoint_runs: set[int] = set()
    for path in checkpoint_files:
        parent = path.parent.parent.name
        if parent.startswith("run_"):
            try:
                checkpoint_runs.add(int(parent.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return {
        "checkpoint_file_count": len(checkpoint_files),
        "checkpoint_run_count": len(checkpoint_runs & completed_run_numbers),
        "sampling_quality_available": bool(checkpoint_runs & completed_run_numbers),
    }


def enrich_conditions(ready: list[dict[str, Any]], raw_by_condition: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in ready:
        condition_id = str(row.get("condition_id"))
        raw = raw_by_condition.get(condition_id, [])
        val_loss = to_float(row.get("final_val_loss_median"))
        val_bpc = to_float(row.get("final_val_bits_per_char_median"))
        elapsed_sec = to_float(row.get("elapsed_sec_median"))
        tokens_per_sec = to_float(row.get("tokens_per_sec_median"))
        params = median(metric_list(raw, "parameter_count"))
        tokens_seen = median(metric_list(raw, "tokens_seen"))
        train_bpc = median(metric_list(raw, "final_train_bits_per_char"))
        best_val_loss = to_float(row.get("best_val_loss_median"))
        flops = TRAINING_FLOPS_FACTOR * params * tokens_seen if params == params and tokens_seen == tokens_seen else math.nan
        tokens_per_param = tokens_seen / params if params and params == params else math.nan
        records = dict(row)
        records.update(
            {
                "raw_completed_runs": len(raw),
                "val_ppl_median": math.exp(val_loss) if val_loss == val_loss and val_loss < 700 else math.inf,
                "val_bits_per_token_median": val_loss / LN2 if val_loss == val_loss else math.nan,
                "best_val_ppl_median": math.exp(best_val_loss) if best_val_loss == best_val_loss and best_val_loss < 700 else math.inf,
                "train_bits_per_char_median": train_bpc,
                "parameter_count_median": params,
                "parameter_m_median": params / 1_000_000 if params == params else math.nan,
                "tokens_seen_median": tokens_seen,
                "tokens_seen_m_median": tokens_seen / 1_000_000 if tokens_seen == tokens_seen else math.nan,
                "train_flops_proxy_median": flops,
                "train_flops_proxy_tflop_median": flops / 1_000_000_000_000 if flops == flops else math.nan,
                "tokens_per_param_median": tokens_per_param,
                "quality_runtime_product": val_bpc * elapsed_sec if val_bpc == val_bpc and elapsed_sec == elapsed_sec else math.nan,
                "quality_compute_product": val_bpc * flops if val_bpc == val_bpc and flops == flops else math.nan,
                "tokens_per_sec_median_float": tokens_per_sec,
            }
        )
        enriched.append(records)
    return enriched


def mark_pareto_frontier(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["compute_quality_frontier"] = False
    for row in rows:
        flops = to_float(row.get("train_flops_proxy_median"))
        bpc = to_float(row.get("final_val_bits_per_char_median"))
        if not (flops == flops and bpc == bpc):
            continue
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_flops = to_float(other.get("train_flops_proxy_median"))
            other_bpc = to_float(other.get("final_val_bits_per_char_median"))
            if not (other_flops == other_flops and other_bpc == other_bpc):
                continue
            if other_flops <= flops and other_bpc <= bpc and (other_flops < flops or other_bpc < bpc):
                dominated = True
                break
        row["compute_quality_frontier"] = not dominated


def write_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_pretraining_quality(rows: list[dict[str, Any]], output_path: Path) -> None:
    ranked = sorted(rows, key=lambda row: to_float(row.get("final_val_bits_per_char_median")))[:20]
    labels = [str(row.get("condition_id")) for row in ranked]
    bpc = [to_float(row.get("final_val_bits_per_char_median")) for row in ranked]
    ppl = [to_float(row.get("val_ppl_median")) for row in ranked]
    colors = [PHASE_COLORS.get(str(row.get("phase")), "#64748b") for row in ranked]
    y = np.arange(len(ranked))
    plt.figure(figsize=(10.0, max(5.5, 0.34 * len(ranked) + 1.8)))
    plt.barh(y, bpc, color=colors, alpha=0.88)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Median validation bits/char (lower is better)")
    plt.title("Pretraining quality: tokenizer-fair ranking")
    plt.grid(axis="x", alpha=0.22)
    for index, row in enumerate(ranked):
        plt.text(bpc[index] + 0.012, index, f"PPL {ppl[index]:.0f}", va="center", fontsize=8, color="#334155")
    write_fig(output_path)


def plot_loss_vs_compute(rows: list[dict[str, Any]], output_path: Path) -> None:
    plt.figure(figsize=(9.2, 5.6))
    for phase in sorted({str(row.get("phase")) for row in rows}):
        phase_rows = [row for row in rows if str(row.get("phase")) == phase]
        x = [to_float(row.get("train_flops_proxy_tflop_median")) for row in phase_rows]
        y = [to_float(row.get("final_val_bits_per_char_median")) for row in phase_rows]
        sizes = [max(38, min(140, to_float(row.get("parameter_m_median")) * 2.4)) for row in phase_rows]
        plt.scatter(x, y, s=sizes, color=PHASE_COLORS.get(phase, "#64748b"), alpha=0.82, label=PHASE_LABELS.get(phase, phase))
    frontier = sorted(
        [row for row in rows if row.get("compute_quality_frontier")],
        key=lambda row: to_float(row.get("train_flops_proxy_tflop_median")),
    )
    if len(frontier) >= 2:
        plt.plot(
            [to_float(row.get("train_flops_proxy_tflop_median")) for row in frontier],
            [to_float(row.get("final_val_bits_per_char_median")) for row in frontier],
            color="#0f172a",
            linewidth=1.8,
            linestyle="--",
            label="compute-quality frontier",
        )
    for row in sorted(rows, key=lambda item: to_float(item.get("final_val_bits_per_char_median")))[:10]:
        plt.text(
            to_float(row.get("train_flops_proxy_tflop_median")) + 0.8,
            to_float(row.get("final_val_bits_per_char_median")),
            str(row.get("condition_id")),
            fontsize=8,
            color="#334155",
        )
    plt.xlabel("Training compute proxy: 6 * parameters * tokens_seen (TFLOPs)")
    plt.ylabel("Median validation bits/char (lower is better)")
    plt.title("Quality vs compute proxy")
    plt.grid(alpha=0.22)
    plt.legend(fontsize=8)
    write_fig(output_path)


def plot_loss_vs_tokens(raw_results: list[dict[str, Any]], output_path: Path) -> None:
    plt.figure(figsize=(9.2, 5.6))
    for phase in sorted({str(row.get("phase")) for row in raw_results}):
        phase_rows = [row for row in raw_results if str(row.get("phase")) == phase]
        x = [to_float(row.get("tokens_seen")) / 1_000_000 for row in phase_rows]
        y = [to_float(row.get("final_val_bits_per_char")) for row in phase_rows]
        plt.scatter(x, y, s=34, color=PHASE_COLORS.get(phase, "#64748b"), alpha=0.68, label=PHASE_LABELS.get(phase, phase))
    plt.xlabel("Training tokens seen per physical run (M tokens)")
    plt.ylabel("Final validation bits/char (lower is better)")
    plt.title("Loss vs tokens: raw completed runs")
    plt.grid(alpha=0.22)
    plt.legend(fontsize=8)
    write_fig(output_path)


def plot_throughput_quality(rows: list[dict[str, Any]], output_path: Path) -> None:
    plt.figure(figsize=(9.2, 5.6))
    for phase in sorted({str(row.get("phase")) for row in rows}):
        phase_rows = [row for row in rows if str(row.get("phase")) == phase]
        x = [to_float(row.get("tokens_per_sec_median")) for row in phase_rows]
        y = [to_float(row.get("final_val_bits_per_char_median")) for row in phase_rows]
        sizes = [max(40, min(150, to_float(row.get("parameter_m_median")) * 2.4)) for row in phase_rows]
        plt.scatter(x, y, s=sizes, color=PHASE_COLORS.get(phase, "#64748b"), alpha=0.82, label=PHASE_LABELS.get(phase, phase))
    for row in sorted(rows, key=lambda item: to_float(item.get("final_val_bits_per_char_median")))[:10]:
        plt.text(
            to_float(row.get("tokens_per_sec_median")) + 14,
            to_float(row.get("final_val_bits_per_char_median")),
            str(row.get("condition_id")),
            fontsize=8,
            color="#334155",
        )
    plt.xlabel("Median training throughput (tokens/sec)")
    plt.ylabel("Median validation bits/char (lower is better)")
    plt.title("System efficiency vs quality")
    plt.grid(alpha=0.22)
    plt.legend(fontsize=8)
    write_fig(output_path)


def plot_tokenizer_fairness(tokenizer_diags: list[dict[str, Any]], output_path: Path) -> None:
    selected = sorted(tokenizer_diags, key=lambda row: to_int(row.get("vocab_size")))
    if not selected:
        return
    x = np.arange(len(selected))
    labels = [f"{to_int(row.get('vocab_size')) // 1000}k" for row in selected]
    chars_per_token = [to_float(row.get("val_chars_per_token")) for row in selected]
    tail_types = [to_float(row.get("vocab_types_freq20")) for row in selected]

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.5), sharex=True)
    axes[0].plot(x, chars_per_token, marker="o", color="#2563eb", linewidth=2)
    axes[0].set_ylabel("Validation chars/token")
    axes[0].set_title("Tokenizer comparability: compression vs tail cost")
    axes[0].grid(alpha=0.22)
    axes[1].bar(x, tail_types, color="#ea580c", alpha=0.84)
    axes[1].set_ylabel("Train vocab types <=20 hits")
    axes[1].set_xlabel("Vocabulary size")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    write_fig(output_path)


def create_figures(
    figures_dir: Path,
    condition_rows: list[dict[str, Any]],
    raw_results: list[dict[str, Any]],
    tokenizer_diags: list[dict[str, Any]],
) -> dict[str, Path]:
    figures: dict[str, Path] = {
        "pretraining_quality": figures_dir / "pretraining_quality_rank.png",
        "loss_vs_compute": figures_dir / "loss_vs_compute.png",
        "loss_vs_tokens": figures_dir / "loss_vs_tokens.png",
        "throughput_quality": figures_dir / "throughput_quality.png",
        "tokenizer_fairness": figures_dir / "tokenizer_fairness.png",
    }
    if not PLOT_AVAILABLE:
        return {key: path for key, path in figures.items() if path.exists()}
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_pretraining_quality(condition_rows, figures["pretraining_quality"])
    plot_loss_vs_compute(condition_rows, figures["loss_vs_compute"])
    plot_loss_vs_tokens(raw_results, figures["loss_vs_tokens"])
    plot_throughput_quality(condition_rows, figures["throughput_quality"])
    plot_tokenizer_fairness(tokenizer_diags, figures["tokenizer_fairness"])
    return figures


def phase_summary(condition_rows: list[dict[str, Any]], phase: str) -> dict[str, Any] | None:
    rows = [row for row in condition_rows if str(row.get("phase")) == phase]
    if not rows:
        return None
    best_quality = min(rows, key=lambda row: to_float(row.get("final_val_bits_per_char_median")))
    best_compute = min(rows, key=lambda row: to_float(row.get("quality_compute_product")))
    fastest = max(rows, key=lambda row: to_float(row.get("tokens_per_sec_median")))
    return {"best_quality": best_quality, "best_compute": best_compute, "fastest": fastest, "rows": rows}


def load_queue_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_report(
    report_path: Path,
    figures: dict[str, Path],
    condition_rows: list[dict[str, Any]],
    raw_results: list[dict[str, Any]],
    phases: list[str],
    meta: dict[str, Any],
    tokenizer_diags: list[dict[str, Any]],
    dataset_manifest: dict[str, Any],
    dataset_comparison: dict[str, Any],
    train_stats: dict[str, Any],
    val_stats: dict[str, Any],
    checkpoint_info: dict[str, Any],
    queue_status: dict[str, Any],
) -> None:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ranked = sorted(condition_rows, key=lambda row: to_float(row.get("final_val_bits_per_char_median")))
    compute_ranked = sorted(condition_rows, key=lambda row: to_float(row.get("quality_compute_product")))
    frontier = [row for row in condition_rows if row.get("compute_quality_frontier")]
    best = ranked[0] if ranked else {}
    best_compute = compute_ranked[0] if compute_ranked else {}
    dataset_name = dataset_manifest.get("dataset_name", "obsidian_llm_10x")
    counts = dataset_manifest.get("counts", {})
    llm_vs_nsmc = dataset_comparison.get("multipliers", {}).get("llm_10x_lm_file_vs_nsmc_chars")

    lines: list[str] = []
    lines.append("# LLM 10x 중간 보고서 - LLM 개발자 지표 기준")
    lines.append("")
    lines.append(f"- generated_at_utc: `{generated_at}`")
    lines.append(f"- completed_physical_runs: `{meta.get('completed_physical_runs')}` / `{meta.get('planned_physical_runs')}`")
    lines.append(f"- pending_physical_runs: `{meta.get('pending_physical_runs')}`")
    lines.append(f"- screen_ready_conditions: `{meta.get('screen_ready_conditions')}`")
    lines.append(f"- claim_ready_conditions: `{meta.get('claim_ready_conditions')}`")
    lines.append(f"- result_ledger_status: `{meta.get('result_ledger_status')}`")
    lines.append(f"- all_results_jsonl: `{meta.get('all_results_jsonl')}`")
    if queue_status:
        lines.append(
            f"- queue_now: `run_{to_int(queue_status.get('current_run_number')):04d}` / "
            f"`{queue_status.get('current_condition_id')}` / `{queue_status.get('stage')}` / "
            f"`{to_int(queue_status.get('current_step'))}/{to_int(queue_status.get('total_steps'))}` updates"
        )
    lines.append("")
    lines.append("## 요약")
    lines.append("")
    lines.append(
        "이번 개정판은 단순 학습 로그 시각화가 아니라 LLM 실험에서 보통 분리해서 보는 "
        "`pretraining quality`, `tokenizer comparability`, `scaling/compute efficiency`, "
        "`system efficiency`, `unmeasured benchmark/safety gaps` 순서로 결과를 다시 정리했습니다."
    )
    if best:
        lines.append(
            f"- 현재 screen-ready 품질 1위는 `{best.get('condition_id')}`이며 validation bits/char 중앙값은 "
            f"`{fmt_float(best.get('final_val_bits_per_char_median'), 5)}`, validation perplexity 중앙값은 "
            f"`{fmt_float(best.get('val_ppl_median'), 1)}`입니다."
        )
    if best_compute:
        lines.append(
            f"- compute proxy까지 같이 보면 현재 비용 효율 후보는 `{best_compute.get('condition_id')}`입니다. "
            f"품질-계산량 곱 기준으로 가장 낮고, 추정 학습 compute는 "
            f"`{fmt_float(best_compute.get('train_flops_proxy_tflop_median'), 1)}` TFLOPs입니다."
        )
    lines.append(
        "- `claim-ready=0`이므로 이 문서는 최종 주장용이 아니라, 지금까지의 결과로 다음 반복/보강 순서를 정하는 중간 의사결정 문서입니다."
    )
    lines.append("")
    lines.append("## 지표 체계")
    lines.append("")
    lines.append("| 영역 | 현재 보고서에서 보는 지표 | 현재 상태 | 해석 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| Pretraining 품질 | validation loss, perplexity, bits/token, bits/char | 사용 가능 | loss가 낮을수록 좋지만 tokenizer가 다르면 bits/char를 우선 비교 |")
    lines.append("| Tokenizer 공정성 | chars/token, bits/char, tail vocab type, top-token mass | 사용 가능 | token loss만 비교하면 vocab size가 다른 조건에서 불공정할 수 있음 |")
    lines.append("| Scaling/compute | parameter_count, tokens_seen, FLOPs proxy = 6*N*T, loss-vs-compute | 근사 가능 | 실제 MFU는 없지만 조건 간 계산량 proxy 비교는 가능 |")
    lines.append("| 시스템 비용 | tokens/sec, elapsed time, seconds/epoch | 사용 가능 | 같은 품질이면 더 빠른 조건이 우선 후보 |")
    lines.append("| Downstream benchmark | MMLU, BIG-bench, GSM8K, HumanEval | 없음 | 현재는 사전학습 소형 LM 탐색이라 태스크 성능 결론 불가 |")
    lines.append("| Chat/instruction | pairwise win rate, LLM judge, MT-Bench류 | 없음 | instruction tuning과 평가셋이 없어 사용자 선호 결론 불가 |")
    lines.append("| Safety/holistic | toxicity, robustness, calibration, fairness | 없음 | 별도 HELM류 harness가 필요 |")
    if checkpoint_info.get("sampling_quality_available"):
        lines.append("| Sample quality | checkpoint 기반 next-token/sample inspection | 일부 가능 | checkpoint가 있는 완료 run만 대상으로 제한 |")
    else:
        lines.append("| Sample quality | next-token/sample inspection | 현재 불가 | 큐가 `--no-checkpoints`로 실행되어 완료 run의 모델 가중치가 없음 |")
    lines.append("")
    lines.append("## 데이터셋과 비교 가능성")
    lines.append("")
    lines.append(
        f"`{dataset_name}`는 `{fmt_int(counts.get('documents'))}`개 문서와 `{fmt_int(counts.get('chunks'))}`개 chunk에서 만든 LM 데이터셋입니다. "
        f"train은 `{fmt_int(train_stats.get('chars'))}` 문자, validation은 `{fmt_int(val_stats.get('chars'))}` 문자입니다."
    )
    if llm_vs_nsmc is not None:
        lines.append(f"- LM 파일 문자 수는 기존 NSMC 파일 대비 약 `{fmt_float(llm_vs_nsmc, 2)}배`입니다.")
    lines.append(
        f"- train 문자 구성은 한글 `{fmt_pct(train_stats.get('hangul_non_ws_ratio'))}`, "
        f"ASCII `{fmt_pct(train_stats.get('ascii_alpha_non_ws_ratio'))}`입니다. "
        "따라서 현재 결론은 한글 전용 tokenizer 결론이 아니라 영어/코드/LLM 자료가 많은 혼합 corpus 결론입니다."
    )
    lines.append("")
    if figures:
        figure_specs = [
            ("pretraining_quality", "Figure 1. Pretraining 품질 순위", "bits/char 기준으로 tokenizer 차이를 보정한 품질 순위입니다. 막대 옆 PPL은 같은 tokenization 조건 안에서 직관을 돕는 보조 지표입니다."),
            ("loss_vs_compute", "Figure 2. Loss vs Compute", "6*N*T FLOPs proxy와 validation bits/char를 같이 봅니다. 점선은 현재 screen-ready 조건의 compute-quality frontier입니다."),
            ("loss_vs_tokens", "Figure 3. Loss vs Tokens", "개별 physical run의 tokens_seen과 validation bits/char를 봅니다. 대부분 0.1 epoch라 x축이 좁지만, 300-step/324-step 차이와 축별 품질 분산을 확인할 수 있습니다."),
            ("throughput_quality", "Figure 4. Throughput vs Quality", "tokens/sec와 validation bits/char를 같이 봅니다. 같은 품질이면 오른쪽 아래 조건이 더 좋습니다."),
            ("tokenizer_fairness", "Figure 5. Tokenizer 공정성", "vocab size별 chars/token 압축 이득과 low-frequency tail type 비용을 동시에 보여줍니다."),
        ]
        lines.append("## 핵심 Figures")
        lines.append("")
        for key, title, explanation in figure_specs:
            path = figures.get(key)
            if not path:
                continue
            lines.append(f"### {title}")
            lines.append("")
            lines.append(explanation)
            lines.append("")
            lines.append(f"![{title}]({relative_to_report(path, report_path)})")
            lines.append("")
    lines.append("## Pretraining 품질 순위")
    lines.append("")
    lines.append(
        "여기서는 `validation bits/char`를 1차 순위 기준으로 둡니다. "
        "`validation loss`와 `perplexity`는 token 단위 지표라 같은 tokenizer 안에서는 유용하지만, vocab size 비교에서는 bits/char가 더 공정합니다."
    )
    lines.append("")
    lines.append("| 순위 | 조건 | 축 | 값 | n | val loss nats/token | val PPL | val bits/token | val bits/char | params(M) | tokens seen(M) | FLOPs proxy(T) | tok/s |")
    lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for index, row in enumerate(ranked[:20], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    str(row.get("condition_id")),
                    PHASE_LABELS.get(str(row.get("phase")), str(row.get("phase"))),
                    str(row.get("axis_value")),
                    str(row.get("completed_runs")),
                    fmt_float(row.get("final_val_loss_median"), 4),
                    fmt_float(row.get("val_ppl_median"), 1),
                    fmt_float(row.get("val_bits_per_token_median"), 4),
                    fmt_float(row.get("final_val_bits_per_char_median"), 5),
                    fmt_float(row.get("parameter_m_median"), 1),
                    fmt_float(row.get("tokens_seen_m_median"), 3),
                    fmt_float(row.get("train_flops_proxy_tflop_median"), 1),
                    fmt_float(row.get("tokens_per_sec_median"), 0),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Scaling / Compute 해석")
    lines.append("")
    lines.append(
        "현재 모든 run은 대체로 `epoch_cap_per_run=0.1`이라 학습 토큰 수가 비슷합니다. "
        "따라서 scaling law를 본격 피팅하기에는 아직 부족하고, 지금은 `동일 예산 근처에서 어떤 조건이 더 좋은가`를 보는 단계입니다."
    )
    if frontier:
        lines.append("")
        lines.append("현재 compute-quality frontier 조건:")
        for row in sorted(frontier, key=lambda item: to_float(item.get("train_flops_proxy_tflop_median"))):
            lines.append(
                f"- `{row.get('condition_id')}`: `{fmt_float(row.get('final_val_bits_per_char_median'), 5)}` bits/char, "
                f"`{fmt_float(row.get('train_flops_proxy_tflop_median'), 1)}` TFLOPs proxy, "
                f"`{fmt_float(row.get('parameter_m_median'), 1)}`M params"
            )
    lines.append("")
    lines.append("## 축별 결론")
    lines.append("")
    for phase in phases:
        summary = phase_summary(condition_rows, phase)
        if not summary:
            continue
        label = PHASE_LABELS.get(phase, phase)
        best_quality = summary["best_quality"]
        best_compute = summary["best_compute"]
        fastest = summary["fastest"]
        rows = sorted(summary["rows"], key=lambda item: to_float(item.get("axis_value")))
        lines.append(f"### {label}")
        lines.append("")
        lines.append(
            f"- 품질 1위: `{best_quality.get('condition_id')}` "
            f"({best_quality.get('sweep_axis')}={best_quality.get('axis_value')}), "
            f"`{fmt_float(best_quality.get('final_val_bits_per_char_median'), 5)}` bits/char."
        )
        lines.append(
            f"- 비용 효율 1위: `{best_compute.get('condition_id')}`, "
            f"`{fmt_float(best_compute.get('train_flops_proxy_tflop_median'), 1)}` TFLOPs proxy."
        )
        lines.append(
            f"- 처리량 1위: `{fastest.get('condition_id')}`, "
            f"`{fmt_float(fastest.get('tokens_per_sec_median'), 0)}` tok/s."
        )
        lines.append(
            f"- 현재 screen-ready 범위: `{rows[0].get('axis_value')}`부터 `{rows[-1].get('axis_value')}`까지, `{len(rows)}` 조건."
        )
        if phase == "phase1_lr":
            lines.append("- 해석: 지금까지는 `2e-4`-`3e-4`가 좋은 구간입니다. `3e-4`는 품질이 좋지만 LR3000 계열은 최근 run에서 PPL/bit가 흔들려, 상위 후보 반복 보강이 필요합니다.")
        elif phase == "phase3_vocab":
            lines.append("- 해석: V10000은 bits/char가 좋아 유망하지만, vocab 확대가 low-frequency tail type을 같이 늘립니다. token loss보다 bits/char와 tail 진단을 같이 봐야 합니다.")
        elif phase == "phase6_context":
            lines.append("- 해석: 0.1 epoch 예산에서는 짧은 context가 더 좋습니다. 긴 context가 나쁜 것이 아니라 제한된 update 수에서 short context가 더 데이터 효율적으로 보이는 신호입니다.")
        elif phase == "phase11_ffn_mult":
            lines.append("- 해석: FFN2/3은 parameter_count가 낮아 비용 효율 후보입니다. 기본 FFN4와 큰 FFN 조건이 screen-ready가 되기 전까지는 최종 모델 크기 결론을 보류해야 합니다.")
        lines.append("")
    if tokenizer_diags:
        lines.append("## Tokenizer 공정성 진단")
        lines.append("")
        lines.append("| vocab | train token | val token | val chars/token | train 20회 이하 type | merge 20회 이하 type | top1 | top10 | top100 | 한글 token 비중 |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for diag in sorted(tokenizer_diags, key=lambda row: to_int(row.get("vocab_size"))):
            lines.append(
                "| "
                + " | ".join(
                    [
                        fmt_int(diag.get("vocab_size")),
                        fmt_int(diag.get("train_tokens")),
                        fmt_int(diag.get("val_tokens")),
                        fmt_float(diag.get("val_chars_per_token"), 3),
                        fmt_int(diag.get("vocab_types_freq20")),
                        fmt_int(diag.get("merge_types_freq20")),
                        fmt_pct(diag.get("top1_mass")),
                        fmt_pct(diag.get("top10_mass")),
                        fmt_pct(diag.get("top100_mass")),
                        fmt_pct(diag.get("hangul_train_mass")),
                    ]
                )
                + " |"
            )
        diag_by_vocab = {to_int(diag.get("vocab_size")): diag for diag in tokenizer_diags}
        if 6000 in diag_by_vocab and 10000 in diag_by_vocab:
            low = diag_by_vocab[6000]
            high = diag_by_vocab[10000]
            token_reduction = 1.0 - to_float(high.get("train_tokens")) / to_float(low.get("train_tokens"))
            lines.append("")
            lines.append(
                f"`V6000 -> V10000`에서 train token은 약 `{fmt_pct(token_reduction)}` 줄지만, "
                f"train 20회 이하 type은 `{fmt_int(low.get('vocab_types_freq20'))}`개에서 "
                f"`{fmt_int(high.get('vocab_types_freq20'))}`개로 늘어납니다. "
                "즉 압축 이득과 tail 학습 부족 위험이 동시에 있습니다."
            )
        lines.append("")
    lines.append("## 아직 이 보고서가 말하지 못하는 것")
    lines.append("")
    lines.append("- MMLU/BIG-bench/GSM8K/HumanEval류 downstream 정확도: 평가 harness와 task prompt가 아직 없습니다.")
    lines.append("- Chat/instruction following: instruction tuning이 없고 pairwise judge 또는 사람 평가가 없습니다.")
    lines.append("- Safety/robustness/fairness/toxicity: 별도 평가셋과 rubric이 없습니다.")
    if not checkpoint_info.get("sampling_quality_available"):
        lines.append("- Next-token sample quality: 현재 완료 run들이 `--no-checkpoints`라 모델 가중치를 보존하지 않아 생성 샘플을 만들 수 없습니다.")
    lines.append("- MFU와 실제 비용: GPU/MPS 저수준 FLOPs utilization과 전력/금액 로그가 없어 `6*N*T` proxy만 사용했습니다.")
    lines.append("")
    lines.append("## 다음 실행 제안")
    lines.append("")
    lines.append("1. 현재 상위 후보를 `n=10` 반복으로 보강해 claim-ready 조건을 먼저 만듭니다.")
    lines.append("2. 상위 후보 2-3개는 `--with-checkpoints`로 별도 짧은 검증 run을 돌려 next-token sample quality를 붙입니다.")
    lines.append("3. tokenizer 비교는 token loss가 아니라 bits/char, chars/token, tail type, top-token mass를 함께 유지합니다.")
    lines.append("4. downstream 평가는 지금 queue와 분리해서 작은 eval harness로 시작합니다. 이 프로젝트 규모에서는 MMLU 전체보다 domain-relevant cloze/QA와 짧은 next-token qualitative set이 먼저 현실적입니다.")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_rows = read_summary(args.summary_csv)
    meta = read_json(args.meta_json)
    ledger_rows = read_ledger(args.all_results_jsonl)
    ready = ready_rows(summary_rows)
    phases = meaningful_phases(ready)
    scoped_ready = [row for row in ready if row.get("phase") in phases]
    raw_by_condition = raw_results_by_condition(ledger_rows)
    raw_results = completed_raw_results(ledger_rows)
    condition_rows = enrich_conditions(scoped_ready, raw_by_condition)
    mark_pareto_frontier(condition_rows)
    tokenizer_diags = [
        diag
        for vocab_size in available_tokenizer_vocab_sizes(args.cache_dir)
        if (diag := tokenizer_diagnostic(args.cache_dir, vocab_size)) is not None
    ]
    figures = create_figures(args.figures_dir, condition_rows, raw_results, tokenizer_diags)
    dataset_manifest = read_optional_json(args.dataset_manifest)
    dataset_comparison = read_optional_json(args.dataset_comparison)
    train_stats = text_corpus_stats(args.train_text)
    val_stats = text_corpus_stats(args.val_text)
    checkpoint_info = checkpoint_status(args.runs_dir, raw_results)
    queue_status = load_queue_status(args.queue_status_json)
    write_report(
        args.report,
        figures,
        condition_rows,
        raw_results,
        phases,
        meta,
        tokenizer_diags,
        dataset_manifest,
        dataset_comparison,
        train_stats,
        val_stats,
        checkpoint_info,
        queue_status,
    )
    print(
        {
            "report": str(args.report),
            "figures_dir": str(args.figures_dir),
            "figure_count": len(figures),
            "screen_ready_conditions": len(condition_rows),
            "raw_completed_runs": len(raw_results),
            "sampling_quality_available": checkpoint_info.get("sampling_quality_available"),
        }
    )


if __name__ == "__main__":
    main()
