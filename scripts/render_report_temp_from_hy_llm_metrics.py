# -*- coding: utf-8 -*-
"""Render REPORT_temp.md from HY LLM relog metrics and sequential figures."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from hy_llm_relog_common import FIGURE_DIR, RELOG_DIR, ROOT, format_float, markdown_table, number, read_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=str(RELOG_DIR / "report_llm_metrics.csv"))
    parser.add_argument("--history", default=str(RELOG_DIR / "report_llm_metrics_by_step.csv"))
    parser.add_argument("--figures", default=str(FIGURE_DIR))
    parser.add_argument("--out", default="REPORT_temp.md")
    args = parser.parse_args()

    metrics = read_csv(Path(args.metrics))
    history = read_csv(Path(args.history))
    by_id = {str(row["experiment_id"]): row for row in metrics}
    history_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in history:
        history_by_id[str(row["experiment_id"])].append(row)
    for rows in history_by_id.values():
        rows.sort(key=lambda row: (number(row, "tokens_seen"), number(row, "step")))

    report = build_report(by_id, history_by_id, Path(args.figures))
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")


def build_report(by_id: dict[str, dict[str, Any]], history_by_id: dict[str, list[dict[str, Any]]], figures: Path) -> str:
    rel = lambda name: str((figures / name).relative_to(ROOT))
    vocab = rows(by_id, ["E04", "E00", "E05", "E06"])
    dropout_long = rows(by_id, ["E21", "E22", "E23", "E24"])
    activation_rows = rows(by_id, ["E49", "E50", "E51", "E52", "E53", "E54", "E55", "E56", "E57"])
    activation_summary = summarize_activation(activation_rows)
    rebound_top = sorted(by_id.values(), key=lambda row: number(row, "final_minus_best_val_loss"), reverse=True)[:8]
    compute_best = sorted(by_id.values(), key=lambda row: number(row, "final_val_bits_per_char"))[:8]

    return f"""# mini GPT HY 실험 LLM 지표 재분석 보고서

이 문서는 원본 `REPORT.md`를 수정하지 않고, `REPORT.md`에 들어간 `docs/HY/testresult/*.md` 실험 로그를 LLM 개발자 기준으로 재로깅해 다시 작성한 수정본입니다.

이번 갱신에서는 기존 final loss 표를 그대로 반복하지 않았습니다. 각 실험의 step별 CSV를 `docs/HY/llm_relog/runs/*/history.jsonl`로 변환하고, `tokens_seen`, `bits/char`, `best/final rebound`, `train-val gap`, `compute proxy` 기준의 순차 선형 그래프를 새로 만들었습니다.

## 1. 재로깅 기준

| 항목 | 값 |
| --- | --- |
| 원본 보고서 | `REPORT.md` |
| 기준 실험 | `docs/HY/testresult/*.md`의 E00~E63 HY 실험 |
| 재로깅 산출물 | `docs/HY/llm_relog/` |
| 순차 그래프 | `docs/HY/figures_llm_seq/` |
| primary x축 | `tokens_seen` |
| vocab 비교 primary metric | `val_bits_per_char` |
| 과적합 primary metric | `train_val_gap`, `final_minus_best_val_loss` |

핵심 해석 규칙은 다음과 같습니다.

- tokenizer가 같은 실험끼리는 `val_loss`를 볼 수 있지만, tokenizer가 다른 vocab 실험은 `bits/char`를 우선합니다.
- final checkpoint만 보지 않고 `best checkpoint`, `final-best rebound`, `gap slope`를 같이 봅니다.
- 모델 크기나 학습량이 바뀐 실험은 `parameter_count * tokens_seen`을 compute proxy로 같이 봅니다.
- seed 반복이 있는 activation 실험은 평균과 표준편차로만 결론을 냅니다.

## 2. 전체 요약

![all compute paths]({rel("14_loss_vs_compute_all_paths.png")})

전체 HY 실험을 `compute_proxy_param_tokens -> val_bits_per_char` 경로로 다시 그리면, 단순 final loss 순위보다 더 조심스러운 결론이 나옵니다. 작은 설정에서 좋아 보이는 실험도 compute를 늘리면 gap이나 rebound가 커질 수 있고, 반대로 초반이 느린 설정도 장기 학습에서 best checkpoint는 더 좋아질 수 있습니다.

상위 `final_val_bits_per_char` 실험은 다음과 같습니다.

{markdown_table(compute_best, ["experiment_id", "changed_variable", "changed_value", "parameter_count", "best_val_bits_per_char", "final_val_bits_per_char", "final_minus_best_val_loss", "final_generalization_gap", "compute_proxy_param_tokens"], 8)}

rebound가 큰 실험은 다음과 같습니다. 이 실험들은 final checkpoint 기준 해석이 특히 위험합니다.

{markdown_table(rebound_top, ["experiment_id", "changed_variable", "changed_value", "best_val_loss", "final_val_loss", "final_minus_best_val_loss", "final_generalization_gap"], 8)}

## 3. Vocab Size: 결론이 실제로 바뀌는 축

질문: vocab이 커질수록 LM 품질이 좋아지는가, 아니면 token loss scale만 바뀌는가?

![vocab bits per char]({rel("02_vocab_bits_per_char_learning_curve.png")})

vocab 실험은 token-level `final_val_loss`로 직접 비교하면 안 됩니다. vocab이 커지면 같은 문장을 더 적은 token으로 표현하므로 cross entropy의 단위가 바뀝니다. 따라서 primary metric은 `final_val_bits_per_char`입니다.

{markdown_table(vocab, ["experiment_id", "vocab_size", "parameter_count", "val_tokens_per_char", "val_chars_per_token", "final_val_loss", "final_val_bits_per_char", "tokens_per_sec_after_warmup"], 10)}

해석:

- token loss만 보면 E04, vocab 2000이 좋아 보입니다.
- 문자당 정보량인 `bits/char`로 보면 E06, vocab 5000이 가장 낮습니다.
- 즉 원본식으로 "vocab 2000이 best"라고 말하면 LLM 지표 기준에서는 틀립니다.
- 다만 vocab 5000은 parameter_count와 LM head 비용이 커지므로 결론은 `confirmed`가 아니라 `trade-off`입니다.

결론 라벨: `trade-off`

## 4. Context Length

질문: NSMC 리뷰 corpus에서 긴 문맥이 실제로 도움이 되는가?

![context length]({rel("01_context_length_learning_curve.png")})

{markdown_table(rows(by_id, ["E01", "E00", "E02", "E03"]), ["experiment_id", "context_length", "best_val_loss", "final_val_loss", "final_val_bits_per_char", "final_generalization_gap", "tokens_per_sec_after_warmup"], 10)}

해석:

- 이 데이터에서는 `context_length=64`가 가장 낮은 validation curve를 보입니다.
- 긴 context는 더 많은 위치/attention 비용을 쓰지만, 짧은 영화 리뷰 corpus에서는 그만큼의 장거리 문맥 이득이 작습니다.
- `context_length=192/256`은 더 빠른 throughput처럼 보이는 구간도 있지만, 품질 곡선은 baseline보다 좋지 않습니다.

결론 라벨: `confirmed`, 단 현재 corpus 길이 분포에 한정

## 5. Capacity: Embedding, Layer, FFN

### 5.1 Embedding Dimension

![emb dim]({rel("03_emb_dim_capacity_path.png")})

{markdown_table(rows(by_id, ["E07", "E00", "E08"]), ["experiment_id", "emb_dim", "parameter_count", "best_val_bits_per_char", "final_val_bits_per_char", "final_generalization_gap", "compute_proxy_param_tokens"], 10)}

해석: `emb_dim=256`은 bits/char를 개선하지만 parameter와 compute가 크게 늘어납니다. 작은 corpus에서 무조건 폭을 키우는 결론이 아니라, 비용 대비 개선폭을 함께 보는 `trade-off`입니다.

### 5.2 Depth와 Norm 안정성

![depth]({rel("05_n_layers_depth_curve.png")})

![norm depth]({rel("06_norm_depth_stability_curve.png")})

{markdown_table(rows(by_id, ["E28", "E27", "E37", "E38", "E45", "E46", "E47", "E48", "E60", "E61", "E62", "E63"]), ["experiment_id", "n_layers", "norm_first", "drop_rate", "num_epochs", "best_val_loss", "final_val_loss", "final_minus_best_val_loss", "final_generalization_gap"], 20)}

해석:

- 질문은 "Pre-LN이 좋은가?"가 아니라 "깊이가 늘어날 때 Pre-LN이 안정성을 주는가?"입니다.
- 12-layer post-LN E47은 loss 7대에 머물러 최적화 실패에 가깝습니다.
- 12-layer pre-LN E48은 정상적으로 수렴해 깊은 모델에서 Pre-LN의 안정성 근거가 됩니다.
- 하지만 E60~E63처럼 `drop_rate=0.2`, `lr=0.0002`로 안정화하면 post-LN도 회복됩니다.
- 따라서 결론은 "Pre-LN 절대 우위"가 아니라 "깊이 증가와 불안정 lr/dropout 조건에서 Pre-LN이 안정성 guardrail을 제공한다"입니다.

결론 라벨: `trade-off`

### 5.3 FFN Multiplier

![ffn]({rel("07_ffn_multiplier_capacity_path.png")})

{markdown_table(rows(by_id, ["E35", "E28", "E36"]), ["experiment_id", "ffn_mult", "parameter_count", "best_val_bits_per_char", "final_val_bits_per_char", "final_generalization_gap", "compute_proxy_param_tokens"], 10)}

해석: FFN을 키우면 최종 bits/char가 소폭 좋아지지만 차이는 작고 compute가 늘어납니다. seed 반복 없이는 강한 결론보다 `inconclusive/trade-off`가 맞습니다.

## 6. Attention Heads와 QKV Bias

![n heads]({rel("04_n_heads_learning_curve.png")})

{markdown_table(rows(by_id, ["E31", "E32", "E28", "E33", "E34"]), ["experiment_id", "n_heads", "best_val_loss", "final_val_loss", "final_generalization_gap", "tokens_per_sec_after_warmup"], 10)}

해석: 같은 `emb_dim=192`에서 head 수를 늘리면 head_dim은 줄어듭니다. E32의 2-head가 가장 낮지만 차이는 작습니다. 이 축은 "많은 head가 항상 좋다"가 아니라 "head 수와 head_dim의 trade-off"입니다.

![qkv bias]({rel("10_qkv_bias_curve.png")})

{markdown_table(rows(by_id, ["E28", "E39", "E37", "E40"]), ["experiment_id", "n_layers", "qkv_bias", "best_val_loss", "final_val_loss", "final_generalization_gap"], 10)}

해석: qkv bias는 4-layer에서는 미세하게 좋아 보이나 8-layer에서는 악화됩니다. 효과 크기가 작고 방향도 조건에 따라 달라져 `inconclusive`입니다.

## 7. Dropout, Epoch, Overfitting

질문: 오래 학습하면 validation도 계속 좋아지는가, 아니면 어느 시점부터 train만 좋아지는가?

![dropout]({rel("09_dropout_overfit_curve.png")})

![epoch rebound]({rel("13_epoch_overfit_rebound_audit.png")})

{markdown_table(dropout_long, ["experiment_id", "drop_rate", "num_epochs", "best_val_loss", "final_val_loss", "final_minus_best_val_loss", "final_generalization_gap", "gap_slope_last_quarter"], 10)}

해석:

- train loss는 epoch가 늘수록 계속 내려가는 것이 정상입니다.
- 과적합 판단은 train loss 하락이 아니라 `val_loss`가 best 이후 되오르는지, 그리고 `train_val_gap`이 커지는지로 봐야 합니다.
- E21 dropout 0.0은 20 epoch에서 final gap과 rebound가 큽니다.
- E24 dropout 0.2는 final이 best와 같고 gap도 가장 작아, 이번 corpus에서는 장기 학습 과적합 억제에 가장 유리합니다.
- "옆 팀이 1500 epoch까지 과적합이 없었다"는 주장은 epoch 숫자만으로 비교할 수 없습니다. `tokens_seen`, `gap`, `rebound`, `generation repetition`이 같이 있어야 합니다.

결론 라벨: `confirmed`

## 8. Weight Tying

![weight tying]({rel("11_weight_tying_long_curve.png")})

{markdown_table(rows(by_id, ["E28", "E41", "E42", "E43", "E44"]), ["experiment_id", "weight_tying", "num_epochs", "parameter_count", "best_val_loss", "final_val_loss", "final_minus_best_val_loss", "final_generalization_gap"], 10)}

해석:

- weight tying은 5/10 epoch에서는 느리게 보일 수 있습니다.
- 장기 50 epoch에서는 E44가 E43보다 final loss, rebound, gap이 모두 낫습니다.
- 즉 weight tying은 단기 수렴 가속 옵션이 아니라 작은 corpus에서 parameter를 줄이고 장기 일반화를 안정화하는 regularization으로 해석하는 편이 맞습니다.

결론 라벨: `confirmed`

## 9. Activation과 Seed Variance

![activation seed]({rel("08_activation_seed_mean_std_curve.png")})

{markdown_table(activation_summary, ["activation", "seed_count", "mean_best_val_bits_per_char", "std_best_val_bits_per_char", "mean_final_val_bits_per_char", "std_final_val_bits_per_char"], 10)}

해석:

- activation은 seed 반복이 있으므로 평균과 표준편차로만 말해야 합니다.
- 평균 기준 GELU가 가장 낮고 ReLU가 그 다음, SiLU가 가장 높습니다.
- 단일 seed 결과만 보고 activation 순위를 말하면 안 됩니다.

결론 라벨: `confirmed`, 현재 depth8/pre-LN/20epoch 조건에 한정

## 10. Stride

![stride]({rel("12_stride_overlap_curve.png")})

{markdown_table(rows(by_id, ["E28", "E29"]), ["experiment_id", "stride", "best_val_loss", "final_val_loss", "final_generalization_gap", "final_tokens_seen", "compute_proxy_param_tokens"], 10)}

해석:

- stride 64는 validation loss를 낮추지만, overlap으로 학습 window 수가 늘어난 효과가 섞여 있습니다.
- 따라서 "데이터가 늘었다"가 아니라 "같은 raw corpus를 더 촘촘히 반복해서 봤다"로 해석해야 합니다.
- `tokens_seen`와 `estimated_chars_seen` 기준 그래프를 같이 봐야 착시를 줄일 수 있습니다.

결론 라벨: `trade-off`

## 11. 최종 결론

1. vocab 실험은 원본 결론을 바꿔야 합니다. token loss 기준 best는 vocab 2000이지만, LLM식 공정 지표인 bits/char 기준 best는 vocab 5000입니다.
2. 작은 corpus에서 epoch를 늘리면 train loss는 계속 내려가지만, dropout이 낮은 조건은 gap과 rebound가 커집니다.
3. Pre-LN은 "항상 성능이 좋다"가 아니라 깊이가 늘거나 post-LN이 불안정한 조건에서 안정성을 주는 옵션입니다.
4. weight tying은 단기 성능 옵션보다 장기 일반화 regularization으로 해석하는 것이 맞습니다.
5. activation, qkv_bias처럼 차이가 작은 축은 seed variance 또는 조건 interaction을 함께 봐야 합니다.
6. 앞으로 추가 실험은 final loss 표가 아니라 `history.jsonl`과 sequential graph를 먼저 남겨야 합니다.
"""


def rows(by_id: dict[str, dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    return [by_id[experiment_id] for experiment_id in ids if experiment_id in by_id]


def summarize_activation(rows_: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_:
        groups[str(row.get("activation"))].append(row)
    output = []
    for activation, items in sorted(groups.items()):
        best = [number(row, "best_val_bits_per_char") for row in items]
        final = [number(row, "final_val_bits_per_char") for row in items]
        output.append(
            {
                "activation": activation,
                "seed_count": len(items),
                "mean_best_val_bits_per_char": mean(best),
                "std_best_val_bits_per_char": pstdev(best) if len(best) > 1 else 0.0,
                "mean_final_val_bits_per_char": mean(final),
                "std_final_val_bits_per_char": pstdev(final) if len(final) > 1 else 0.0,
            }
        )
    return sorted(output, key=lambda row: number(row, "mean_final_val_bits_per_char"))


if __name__ == "__main__":
    main()
