# -*- coding: utf-8 -*-
"""HY 실험 로그를 LLM 개발 지표로 재해석해 REPORT_temp.md와 보조 그래프를 생성합니다."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent.parent
HY_RESULT_DIR = ROOT / "docs" / "HY" / "testresult"
HY_FIGURE_DIR = ROOT / "docs" / "HY" / "figures_temp"
HY_METRIC_CSV = ROOT / "docs" / "HY" / "report_llm_metrics.csv"
TRAIN_LEADERBOARD = ROOT / "docs" / "train" / "leaderboard.csv"
REPORT_TEMP = ROOT / "REPORT_temp.md"
LOG2_E = math.log(2)


def configure_matplotlib_font() -> None:
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


@dataclass
class ExperimentRecord:
    experiment_id: str
    title: str
    source_path: Path
    metrics: dict[str, Any]
    series: list[dict[str, Any]] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metrics.get(key, default)

    def number(self, key: str, default: float = 0.0) -> float:
        return to_float(self.metrics.get(key), default)


def main() -> None:
    configure_matplotlib_font()
    HY_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    records = load_hy_records()
    write_metric_csv(records, HY_METRIC_CSV)
    figures = make_figures(records)
    leaderboard_rows = load_train_leaderboard()
    REPORT_TEMP.write_text(build_report(records, figures, leaderboard_rows), encoding="utf-8")
    print(f"wrote {HY_METRIC_CSV.relative_to(ROOT)}")
    print(f"wrote {REPORT_TEMP.relative_to(ROOT)}")
    for figure in figures:
        print(f"wrote {figure.relative_to(ROOT)}")


def load_hy_records() -> list[ExperimentRecord]:
    candidates: dict[str, list[ExperimentRecord]] = {}
    for path in sorted(HY_RESULT_DIR.glob("E*_result.md")):
        record = parse_hy_markdown(path)
        candidates.setdefault(record.experiment_id, []).append(record)
    chosen = [max(records, key=record_priority) for records in candidates.values()]
    return sorted(chosen, key=lambda record: int(record.experiment_id[1:]))


def record_priority(record: ExperimentRecord) -> tuple[int, int]:
    simple_name = f"{record.experiment_id}_result.md"
    return (0 if record.source_path.name == simple_name else 1, len(record.source_path.name))


def parse_hy_markdown(path: Path) -> ExperimentRecord:
    text = path.read_text(encoding="utf-8")
    experiment_id = re.search(r"(E\d+)", path.name)
    if not experiment_id:
        raise ValueError(f"cannot find experiment id in {path}")
    title = first_heading(text) or path.stem
    metrics = parse_two_column_tables(text)
    series = parse_csv_blocks(text)
    metrics["experiment_id"] = experiment_id.group(1)
    metrics["title"] = title
    metrics["source"] = str(path.relative_to(ROOT))
    normalize_record_metrics(metrics, series)
    return ExperimentRecord(experiment_id.group(1), title, path, metrics, series)


def first_heading(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def parse_two_column_tables(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in text.splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 2:
            continue
        key, value = cells
        if key in {"항목", "지표"} or value == "값":
            continue
        normalized_key = normalize_key(key)
        data[normalized_key] = parse_value(value)
    return data


def parse_csv_blocks(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in re.finditer(r"```csv\n(.*?)```", text, flags=re.DOTALL):
        block = match.group(1).strip()
        if not block:
            continue
        reader = csv.DictReader(block.splitlines())
        for row in reader:
            rows.append({key: parse_value(value) for key, value in row.items()})
    return rows


def normalize_record_metrics(metrics: dict[str, Any], series: list[dict[str, Any]]) -> None:
    alias_pairs = {
        "num_epochs": "epochs",
        "lr": "learning_rate",
        "weight_tying": "tie_embeddings",
        "train_tokens": "train_token_count",
        "val_tokens": "val_token_count",
        "train_chars": "train_char_count",
        "val_chars": "val_char_count",
    }
    for old_key, new_key in alias_pairs.items():
        if old_key in metrics and new_key not in metrics:
            metrics[new_key] = metrics[old_key]

    final_train_loss = to_float(metrics.get("final_train_loss"))
    final_val_loss = to_float(metrics.get("final_val_loss"))
    train_chars = to_float(metrics.get("train_char_count"))
    val_chars = to_float(metrics.get("val_char_count"))
    train_tokens = to_float(metrics.get("train_token_count"))
    val_tokens = to_float(metrics.get("val_token_count"))
    parameter_count = to_float(metrics.get("parameter_count"))
    tokens_seen = max_float(series, "tokens_seen")
    if tokens_seen == 0:
        tokens_seen = train_tokens * to_float(metrics.get("epochs"))

    best_val_loss = to_float(metrics.get("best_val_loss"))
    if best_val_loss == 0:
        best_val_loss = min_positive([to_float(row.get("val_loss")) for row in series], final_val_loss)
    best_step = to_float(metrics.get("best_step"))
    if best_step == 0 and series:
        best_row = min(series, key=lambda row: to_float(row.get("val_loss"), float("inf")))
        best_step = to_float(best_row.get("step"))

    train_tokens_per_char = safe_ratio(train_tokens, train_chars)
    val_tokens_per_char = safe_ratio(val_tokens, val_chars)
    train_chars_per_token = safe_ratio(train_chars, train_tokens)
    val_chars_per_token = safe_ratio(val_chars, val_tokens)

    metrics["train_tokens_per_char"] = train_tokens_per_char
    metrics["val_tokens_per_char"] = val_tokens_per_char
    metrics["train_chars_per_token"] = train_chars_per_token
    metrics["val_chars_per_token"] = val_chars_per_token
    metrics["final_train_nats_per_char"] = final_train_loss * train_tokens_per_char
    metrics["final_val_nats_per_char"] = final_val_loss * val_tokens_per_char
    metrics["final_train_bits_per_char"] = safe_ratio(metrics["final_train_nats_per_char"], LOG2_E)
    metrics["final_val_bits_per_char"] = safe_ratio(metrics["final_val_nats_per_char"], LOG2_E)
    metrics["best_val_loss"] = best_val_loss
    metrics["best_step"] = best_step
    metrics["tokens_seen"] = tokens_seen
    metrics["final_minus_best_val_loss"] = final_val_loss - best_val_loss
    metrics["final_generalization_gap"] = final_val_loss - final_train_loss
    metrics["compute_proxy_param_tokens"] = parameter_count * tokens_seen
    metrics["estimated_train_flops"] = 6 * parameter_count * tokens_seen
    metrics["tokens_per_parameter"] = safe_ratio(tokens_seen, parameter_count)
    metrics["final_val_perplexity"] = safe_exp(final_val_loss)
    metrics["estimated_chars_seen"] = tokens_seen * train_chars_per_token


def write_metric_csv(records: list[ExperimentRecord], path: Path) -> None:
    fields = [
        "experiment_id",
        "title",
        "source",
        "vocab_size",
        "context_length",
        "emb_dim",
        "n_heads",
        "n_layers",
        "ffn_mult",
        "drop_rate",
        "norm_first",
        "activation",
        "activation_name",
        "tie_embeddings",
        "qkv_bias",
        "stride",
        "epochs",
        "parameter_count",
        "train_char_count",
        "val_char_count",
        "train_token_count",
        "val_token_count",
        "val_tokens_per_char",
        "val_chars_per_token",
        "final_train_loss",
        "final_val_loss",
        "best_val_loss",
        "final_minus_best_val_loss",
        "final_generalization_gap",
        "final_val_nats_per_char",
        "final_val_bits_per_char",
        "tokens_seen",
        "tokens_per_sec",
        "compute_proxy_param_tokens",
        "estimated_train_flops",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def make_figures(records: list[ExperimentRecord]) -> list[Path]:
    figures = [
        plot_vocab(records),
        plot_capacity(records),
        plot_compute(records),
        plot_rebound(records),
        plot_activation_seed(records),
    ]
    return [figure for figure in figures if figure is not None]


def plot_vocab(records: list[ExperimentRecord]) -> Path | None:
    vocab_records = select(records, ["E04", "E00", "E05", "E06"])
    if not vocab_records:
        return None
    path = HY_FIGURE_DIR / "vocab_loss_vs_bits_per_char.png"
    x = [record.number("vocab_size") for record in vocab_records]
    token_loss = [record.number("final_val_loss") for record in vocab_records]
    bits = [record.number("final_val_bits_per_char") for record in vocab_records]
    tokens_per_char = [record.number("val_tokens_per_char") for record in vocab_records]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(x, token_loss, marker="o", label="token-level val loss")
    ax1.set_xlabel("vocab_size")
    ax1.set_ylabel("final_val_loss")
    ax2 = ax1.twinx()
    ax2.plot(x, bits, marker="s", color="#c2410c", label="val bits/char")
    ax2.plot(x, tokens_per_char, marker="^", color="#15803d", label="val tokens/char")
    ax2.set_ylabel("bits/char, tokens/char")
    fig.suptitle("vocab_size: token loss와 문자당 지표는 다른 결론을 낸다")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_capacity(records: list[ExperimentRecord]) -> Path | None:
    selected = select(records, ["E07", "E00", "E08", "E11", "E12", "E13", "E35", "E28", "E36"])
    if not selected:
        return None
    path = HY_FIGURE_DIR / "parameter_count_vs_bits_per_char.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for record in selected:
        x = record.number("parameter_count")
        y = record.number("final_val_bits_per_char")
        ax.scatter(x, y, s=70)
        ax.text(x, y, f" {record.experiment_id}", fontsize=8)
    ax.set_xlabel("parameter_count")
    ax.set_ylabel("final val bits/char")
    ax.set_title("모델 크기 실험은 loss가 아니라 parameter 대비 bits/char로 본다")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_compute(records: list[ExperimentRecord]) -> Path | None:
    selected = [record for record in records if record.number("compute_proxy_param_tokens") > 0 and record.number("final_val_bits_per_char") > 0]
    if not selected:
        return None
    path = HY_FIGURE_DIR / "compute_proxy_vs_bits_per_char.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for record in selected:
        ax.scatter(record.number("compute_proxy_param_tokens"), record.number("final_val_bits_per_char"), s=28, alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("parameter_count * tokens_seen")
    ax.set_ylabel("final val bits/char")
    ax.set_title("loss-vs-compute proxy")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_rebound(records: list[ExperimentRecord]) -> Path | None:
    selected = [record for record in records if record.number("final_minus_best_val_loss", -999.0) > 0]
    if not selected:
        return None
    selected = sorted(selected, key=lambda record: record.number("final_minus_best_val_loss"), reverse=True)[:16]
    path = HY_FIGURE_DIR / "final_minus_best_val_loss.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [record.experiment_id for record in selected]
    values = [record.number("final_minus_best_val_loss") for record in selected]
    ax.bar(labels, values, color="#d97706")
    ax.set_ylabel("final_val_loss - best_val_loss")
    ax.set_title("후반 validation rebound가 큰 실험")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_activation_seed(records: list[ExperimentRecord]) -> Path | None:
    selected = select(records, [f"E{number:02d}" for number in range(49, 58)])
    groups: dict[str, list[float]] = {}
    for record in selected:
        activation = str(record.get("activation") or record.get("activation_name") or "unknown")
        groups.setdefault(activation, []).append(record.number("final_val_bits_per_char"))
    groups = {name: values for name, values in groups.items() if len(values) >= 2}
    if not groups:
        return None
    path = HY_FIGURE_DIR / "activation_seed_bits_per_char.png"
    names = list(groups)
    means = [mean(groups[name]) for name in names]
    errors = [stdev(groups[name]) if len(groups[name]) > 1 else 0.0 for name in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(names, means, yerr=errors, capsize=5, color="#2563eb")
    ax.set_ylabel("mean final val bits/char")
    ax.set_title("activation seed 반복: 평균과 표준편차")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build_report(records: list[ExperimentRecord], figures: list[Path], leaderboard_rows: list[dict[str, Any]]) -> str:
    vocab = select(records, ["E04", "E00", "E05", "E06"])
    context = select(records, ["E01", "E00", "E02", "E03"])
    emb = select(records, ["E07", "E00", "E08"])
    layers = select(records, ["E11", "E00", "E12", "E13"])
    ffn = select(records, ["E35", "E28", "E36"])
    weight_tying = select(records, ["E43", "E44"])
    activation = select(records, [f"E{number:02d}" for number in range(49, 58)])
    figures_md = "\n".join(f"- `{figure.relative_to(ROOT)}`" for figure in figures)
    top_leaderboard = sorted(leaderboard_rows, key=lambda row: to_float(row.get("final_val_bits_per_char"), float("inf")))[:8]

    return f"""# REPORT_temp: LLM 개발 지표로 다시 해석한 실험 보고서

이 문서는 루트 `REPORT.md`를 수정하지 않고, 기존 HY 실험 로그와 최근 `docs/train/leaderboard.csv`를 LLM 개발자가 보는 방식으로 다시 읽기 위해 만든 임시 보고서입니다.

## 결론 요약

기존 보고서의 큰 흐름은 유효하지만, `vocab_size` 해석은 반드시 수정해야 합니다. tokenizer가 달라지면 한 token이 담당하는 문자 수가 달라지므로 `final_val_loss`만 비교하면 공정하지 않습니다.

- token-level loss 기준: E04, vocab 2000이 좋아 보입니다.
- 문자당 정보량 기준: E06, vocab 5000이 가장 낮은 `final_val_bits_per_char`를 보입니다.
- 따라서 vocab 실험의 결론은 "vocab 2000이 best"가 아니라 "token loss는 2000이 낮지만, 문자 기준 압축 효율은 5000이 좋다. 다만 parameter_count와 처리량 비용을 함께 봐야 한다"로 바뀌어야 합니다.

## 추가로 구현한 로그 항목

`src/experiments.py`의 실험 반환값에 아래 지표를 추가했습니다. 기존 함수 시그니처는 바꾸지 않았습니다.

| 지표 | 의미 |
| --- | --- |
| `final_train_perplexity`, `final_val_perplexity` | token loss를 perplexity로 변환한 값 |
| `observed_best_val_loss` | smoke 실험에서 관측 가능한 initial/final 중 낮은 val loss |
| `final_minus_observed_best_val_loss` | final이 관측 best보다 얼마나 나빠졌는지 |
| `estimated_chars_seen` | tokenizer 차이를 감안한 raw text 노출량 근사 |
| `compute_proxy_param_tokens` | `parameter_count * tokens_seen` |
| `estimated_train_flops` | `6 * parameter_count * tokens_seen` 근사 |
| `tokens_per_parameter` | 파라미터 하나당 본 token 수 |
| `warmup_excluded_tokens_per_sec` | 첫 step warmup을 제외한 train throughput |

HY 기존 로그에는 이미 `best_val_loss`, `tokens_seen`, `tokens_per_sec`가 있으므로 재실행 없이도 아래 재해석을 만들 수 있었습니다. 새 로그 항목은 앞으로 돌릴 실험부터 더 정확히 남습니다.

## 생성 산출물

- `docs/HY/report_llm_metrics.csv`
{figures_md}

## Vocab Size: 결론이 바뀌는 핵심 실험

{markdown_table(vocab, ["experiment_id", "vocab_size", "parameter_count", "val_tokens_per_char", "val_chars_per_token", "final_val_loss", "final_val_bits_per_char", "tokens_per_sec"])}

해석:

- `vocab_size`가 커질수록 `val_tokens_per_char`는 내려갑니다. 같은 문장을 더 적은 token으로 표현한다는 뜻입니다.
- `final_val_loss`는 token 하나를 맞히는 난이도라 vocab이 커질수록 불리해질 수 있습니다.
- 그래서 E04의 `final_val_loss=4.8114`만 보고 best라고 하면 tokenizer 효율을 놓칩니다.
- `final_val_bits_per_char`는 E06이 가장 낮습니다. 문자 하나를 설명하는 정보량 기준으로는 vocab 5000이 더 낫습니다.
- 단, E06은 `parameter_count`도 더 크고 처리량도 낮으므로 최종 선택은 품질/비용 trade-off로 써야 합니다.

## Context Length

{markdown_table(context, ["experiment_id", "context_length", "stride", "final_val_loss", "final_val_bits_per_char", "tokens_per_sec", "compute_proxy_param_tokens"])}

해석:

- context 실험은 token loss와 bits/char 해석이 크게 충돌하지 않습니다. E01, context 64가 가장 좋습니다.
- 긴 context가 항상 좋은 것이 아니라, 이 데이터와 5 epoch 조건에서는 학습 sample 수 감소와 attention 비용 증가가 더 크게 작용한 것으로 보입니다.
- `tokens_per_sec`는 일반 기대와 다르게 흔들릴 수 있으므로 앞으로는 `warmup_excluded_tokens_per_sec`를 우선 보겠습니다.

## Embedding Dimension

{markdown_table(emb, ["experiment_id", "emb_dim", "parameter_count", "final_val_loss", "final_val_bits_per_char", "tokens_per_sec", "compute_proxy_param_tokens"])}

해석:

- E08, emb_dim 256은 품질 지표가 가장 좋지만 parameter_count와 compute proxy가 크게 증가합니다.
- 따라서 "큰 embedding이 좋다"가 아니라 "품질은 좋아졌지만 비용 대비 이득을 따로 판단해야 한다"로 써야 합니다.

## Layer Count

{markdown_table(layers, ["experiment_id", "n_layers", "parameter_count", "final_val_loss", "final_val_bits_per_char", "tokens_per_sec", "compute_proxy_param_tokens"])}

해석:

- 5 epoch 조건에서는 4층 baseline이 2/6/8층보다 약간 낫습니다.
- 깊이를 늘리면 표현력은 늘지만 작은 데이터와 짧은 학습에서는 최적화가 어려워질 수 있습니다.
- 층 수 실험은 `n_layers -> loss`만 보지 말고 `parameter_count`, `tokens_per_sec`, `compute_proxy`를 함께 봐야 합니다.

## FFN Multiplier

{markdown_table(ffn, ["experiment_id", "ffn_mult", "parameter_count", "final_val_loss", "final_val_bits_per_char", "final_generalization_gap", "compute_proxy_param_tokens"])}

해석:

- FFN 폭을 키우면 token별 비선형 변환 능력은 늘지만 parameter_count와 과적합 위험도 같이 봐야 합니다.
- 이 실험군은 `ffn_mult` 하나만으로 결론을 내리기보다 depth, norm_first, dropout과의 조합으로 보는 것이 맞습니다.

## Weight Tying

{markdown_table(weight_tying, ["experiment_id", "tie_embeddings", "parameter_count", "final_train_loss", "final_val_loss", "final_generalization_gap", "final_val_bits_per_char"])}

해석:

- weight tying은 parameter_count를 줄이면서 validation 기준을 개선했습니다.
- 작은 corpus에서는 출력 embedding 공유가 regularization처럼 작동했을 가능성이 있습니다.
- 이 결론은 LLM식 지표로 봐도 유지됩니다.

## Activation Seed 반복

{activation_summary_table(activation)}

해석:

- activation 실험은 단일 seed보다 평균과 표준편차가 중요합니다.
- 평균이 낮아도 표준편차가 크면 "우연히 한 seed에서 잘 됐다"일 수 있습니다.
- 이 실험군은 LLM 보고서식으로 이미 좋은 방향입니다. `mean ± std`를 본문 결론에 넣으면 더 설득력이 올라갑니다.

## 최근 docs/train 자동화 로그에서 참고할 점

`docs/train/leaderboard.csv`는 HY와 corpus/규모가 달라 직접 수치 비교하면 안 됩니다. 대신 어떤 지표를 계속 남겨야 하는지 보여주는 보조 evidence로 쓸 수 있습니다.

{markdown_rows(top_leaderboard, ["run_id", "vocab_size", "context_length", "activation_name", "ffn_mult", "final_val_loss", "final_val_bits_per_char", "final_generalization_gap", "overfit_score", "parameter_count", "tokens_per_sec"])}

이 자동화 로그는 이미 `final_val_bits_per_char`, `parameter_count`, `tokens_per_sec`, `overfit_score`를 남기고 있습니다. 앞으로 HY 보고서도 같은 표준으로 맞추면 됩니다.

## REPORT.md에 반영해야 할 수정 방향

1. vocab 실험 결론을 token loss 중심에서 bits/char 중심으로 바꿉니다.
2. tokenizer가 바뀌는 비교는 `final_val_loss` 단독 결론을 금지합니다.
3. 모델 크기가 바뀌는 비교는 `parameter_count`, `tokens_seen`, `tokens_per_sec`, `compute_proxy`를 같이 표시합니다.
4. 오래 학습한 실험은 `best_val_loss`, `final_val_loss`, `final_minus_best_val_loss`, `final_generalization_gap`을 같이 봅니다.
5. seed 반복이 있는 실험은 단일 표가 아니라 평균과 표준편차로 결론을 씁니다.

## 최종 재해석

현재 실험은 "하이퍼파라미터별 final validation loss 비교"로는 이미 충분한 출발점입니다. 다만 LLM 개발자 관점에서는 결론의 중심축을 바꿔야 합니다.

- tokenizer/vocab 실험: `bits/char`가 주 지표입니다.
- capacity 실험: `bits/char`와 `parameter_count`, `compute proxy`를 같이 봅니다.
- training stability 실험: `best-final rebound`, `train-val gap`, `overfit_score`를 봅니다.
- runtime 실험: 전체 elapsed 기준 `tokens_per_sec`보다 warmup 제외 처리량을 봅니다.

따라서 가장 먼저 고칠 문장은 "vocab_size=2000이 가장 좋다"입니다. 더 정확히는 "token-level loss는 vocab_size=2000이 가장 낮았지만, 문자당 정보량 기준인 bits/char는 vocab_size=5000이 가장 낮다. 이 결과는 tokenizer 효율과 모델 크기 증가의 trade-off로 해석해야 한다"입니다.
"""


def activation_summary_table(records: list[ExperimentRecord]) -> str:
    groups: dict[str, list[ExperimentRecord]] = {}
    for record in records:
        activation = str(record.get("activation") or record.get("activation_name") or "unknown")
        groups.setdefault(activation, []).append(record)
    rows = []
    for activation, group in sorted(groups.items()):
        values = [record.number("final_val_bits_per_char") for record in group]
        losses = [record.number("final_val_loss") for record in group]
        rows.append(
            {
                "activation": activation,
                "runs": len(group),
                "mean_val_loss": mean(losses),
                "std_val_loss": stdev(losses) if len(losses) > 1 else 0.0,
                "mean_bits_per_char": mean(values),
                "std_bits_per_char": stdev(values) if len(values) > 1 else 0.0,
            }
        )
    return markdown_rows(rows, ["activation", "runs", "mean_val_loss", "std_val_loss", "mean_bits_per_char", "std_bits_per_char"])


def markdown_table(records: list[ExperimentRecord], fields: list[str]) -> str:
    return markdown_rows([record.metrics for record in records], fields)


def markdown_rows(rows: list[dict[str, Any]], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join(["---"] * len(fields)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(format_cell(row.get(field, "")) for field in fields) + " |")
    return "\n".join([header, divider, *body])


def select(records: list[ExperimentRecord], experiment_ids: list[str]) -> list[ExperimentRecord]:
    by_id = {record.experiment_id: record for record in records}
    return [by_id[experiment_id] for experiment_id in experiment_ids if experiment_id in by_id]


def load_train_leaderboard() -> list[dict[str, Any]]:
    if not TRAIN_LEADERBOARD.exists():
        return []
    with TRAIN_LEADERBOARD.open("r", encoding="utf-8", newline="") as file:
        return [{key: parse_value(value) for key, value in row.items()} for row in csv.DictReader(file)]


def normalize_key(key: str) -> str:
    key = key.strip().replace(" ", "_")
    replacements = {
        "activation": "activation",
        "final_perplexity": "final_perplexity",
    }
    return replacements.get(key, key)


def parse_value(value: str | None) -> Any:
    if value is None:
        return ""
    value = value.strip()
    if value in {"", " "}:
        return ""
    if value in {"True", "False"}:
        return value == "True"
    numeric = value.replace(",", "")
    try:
        if re.fullmatch(r"[-+]?\d+", numeric):
            return int(numeric)
        if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+)([eE][-+]?\d+)?", numeric) or re.fullmatch(r"[-+]?\d+[eE][-+]?\d+", numeric):
            return float(numeric)
    except ValueError:
        return value
    return value


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf")


def max_float(rows: list[dict[str, Any]], key: str) -> float:
    return max([to_float(row.get(key)) for row in rows], default=0.0)


def min_positive(values: list[float], fallback: float) -> float:
    positives = [value for value in values if value > 0]
    return min(positives) if positives else fallback


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        if abs(value) >= 1_000_000:
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
