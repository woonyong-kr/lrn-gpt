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
    figure = lambda name, alt: f"![{alt}]({rel(name)})"
    vocab = rows(by_id, ["E04", "E00", "E05", "E06"])
    context = rows(by_id, ["E01", "E00", "E02", "E03"])
    emb = rows(by_id, ["E07", "E00", "E08"])
    n_heads_short = rows(by_id, ["E09", "E00", "E10"])
    n_heads_long = rows(by_id, ["E31", "E32", "E28", "E33", "E34"])
    layers_short = rows(by_id, ["E11", "E00", "E12", "E13"])
    ffn_short = rows(by_id, ["E14", "E00", "E15"])
    ffn_long = rows(by_id, ["E35", "E28", "E36"])
    dropout_short = rows(by_id, ["E16", "E17", "E00", "E18"])
    dropout_long = rows(by_id, ["E21", "E22", "E23", "E24"])
    qkv_short = rows(by_id, ["E00", "E19"])
    qkv_long = rows(by_id, ["E28", "E39", "E37", "E40"])
    weight_short = rows(by_id, ["E00", "E20"])
    weight_long = rows(by_id, ["E28", "E41", "E42", "E43", "E44"])
    norm_short = rows(by_id, ["E28", "E27", "E37", "E38"])
    norm_long = rows(by_id, ["E45", "E46", "E47", "E48", "E60", "E61", "E62", "E63"])
    stride = rows(by_id, ["E28", "E29"])
    activation_rows = rows(by_id, ["E49", "E50", "E51", "E52", "E53", "E54", "E55", "E56", "E57"])
    activation_summary = summarize_activation(activation_rows)
    rebound_top = sorted(by_id.values(), key=lambda row: number(row, "final_minus_best_val_loss"), reverse=True)[:8]
    compute_best = sorted(by_id.values(), key=lambda row: number(row, "final_val_bits_per_char"))[:8]
    run_count = len(by_id)
    history_count = sum(len(history_by_id.get(experiment_id, [])) for experiment_id in by_id)

    return f"""# mini GPT HY 실험 LLM 지표 재분석 보고서

이 문서는 원본 `REPORT.md`를 수정하지 않고, `REPORT.md`에 들어간 `docs/HY/testresult/*.md` 실험 로그를 LLM 개발자 기준으로 재로깅해 다시 작성한 수정본입니다.

이번 갱신에서는 기존 final loss 표를 그대로 반복하지 않았습니다. 각 실험의 step별 CSV를 `docs/HY/llm_relog/runs/*/history.jsonl`로 변환하고, `tokens_seen`, `bits/char`, `best/final rebound`, `train-val gap`, `compute proxy` 기준의 순차 선형 그래프를 새로 만들었습니다.

중요한 전제부터 명확히 둡니다. 여기서 "실험을 실행했다"는 말은 현재 repo에 남아 있는 `REPORT.md` 기준 HY 실험 로그 전체를 다시 학습했다는 뜻이 아닙니다. 현재 repo에는 원본 HY 학습 runner 파일이 없고, `docs/HY/testresult/*.md`에 이미 기록된 step별 CSV가 있습니다. 이번 작업은 그 기존 로그 {run_count}개 실험, {history_count}개 evaluation point를 LLM 지표 체계로 재로깅하고, 선형 그래프로 다시 해석한 것입니다. 매니페스트 기준으로는 `rerun_required=0`이므로, 누락 로그 때문에 새 학습을 추가로 돌린 실험은 없습니다.

## 1. 실행 범위와 검증 범위

| 항목 | 값 |
| --- | --- |
| 원본 보고서 | `REPORT.md` |
| 기준 실험 | `docs/HY/testresult/*.md`의 E00~E63 HY 실험 |
| 재로깅한 고유 실험 수 | {run_count} |
| 재계산한 evaluation point 수 | {history_count} |
| 재로깅 산출물 | `docs/HY/llm_relog/` |
| 순차 그래프 | `docs/HY/figures_llm_seq/` |
| primary x축 | `tokens_seen` |
| vocab 비교 primary metric | `val_bits_per_char` |
| 과적합 primary metric | `train_val_gap`, `final_minus_best_val_loss` |
| 새 학습 재실행 여부 | 기존 HY 로그에 누락이 없어 재학습 없음 |

실제로 실행한 검증은 아래와 같습니다.

- `python scripts/verify_hy_llm_report_assets.py`
- `python -m pytest tests/test_hy_llm_relog.py -q`
- `python -m pytest tests/ -q`

따라서 "수많은 테스트를 모두 실행했느냐"에 대한 정확한 답은 이렇습니다.

- 전체 HY 실험을 새로 학습 실행한 것은 아닙니다.
- 기존 HY 실험 로그 전체를 재파싱해 LLM 지표로 다시 계산했습니다.
- 새로 만든 relog/graph/report 검증 테스트와 repo 전체 pytest는 실행했습니다.
- `REPORT.md`는 수정하지 않았고, 수정본은 `REPORT_temp.md`에만 작성했습니다.

## 2. LLM 전문가식으로 보는 순서

핵심 해석 규칙은 다음과 같습니다.

- tokenizer가 같은 실험끼리는 `val_loss`를 볼 수 있지만, tokenizer가 다른 vocab 실험은 `bits/char`를 우선합니다.
- final checkpoint만 보지 않고 `best checkpoint`, `final-best rebound`, `gap slope`를 같이 봅니다.
- 모델 크기나 학습량이 바뀐 실험은 `parameter_count * tokens_seen`을 compute proxy로 같이 봅니다.
- seed 반복이 있는 activation 실험은 평균과 표준편차로만 결론을 냅니다.

선형 그래프는 아래 순서로 읽습니다.

1. x축을 먼저 봅니다. 이번 그래프의 기본 x축은 epoch가 아니라 `tokens_seen`입니다.
2. validation curve가 계속 내려가는지, 어느 지점에서 best를 찍고 되오르는지 봅니다.
3. 같은 구간에서 train curve 또는 `train_val_gap`이 커지는지 확인합니다.
4. final point가 best point와 같은지, 아니면 `final_minus_best_val_loss`가 큰지 봅니다.
5. 조건별 성능 차이가 compute 증가, throughput 감소, parameter 증가를 정당화하는지 봅니다.

## 3. 전체 경로: Loss vs Compute

{figure("14_loss_vs_compute_all_paths.png", "all compute paths")}

전체 HY 실험을 `compute_proxy_param_tokens -> val_bits_per_char` 경로로 다시 그리면, 단순 final loss 순위보다 더 조심스러운 결론이 나옵니다. 작은 설정에서 좋아 보이는 실험도 compute를 늘리면 gap이나 rebound가 커질 수 있고, 반대로 초반이 느린 설정도 장기 학습에서 best checkpoint는 더 좋아질 수 있습니다.

상위 `final_val_bits_per_char` 실험은 다음과 같습니다.

{markdown_table(compute_best, ["experiment_id", "changed_variable", "changed_value", "parameter_count", "best_val_bits_per_char", "final_val_bits_per_char", "final_minus_best_val_loss", "final_generalization_gap", "compute_proxy_param_tokens"], 8)}

rebound가 큰 실험은 다음과 같습니다. 이 실험들은 final checkpoint 기준 해석이 특히 위험합니다.

{markdown_table(rebound_top, ["experiment_id", "changed_variable", "changed_value", "best_val_loss", "final_val_loss", "final_minus_best_val_loss", "final_generalization_gap"], 8)}

이 표에서 중요한 점은 "상위 bits/char"와 "작은 rebound"가 항상 같은 실험을 가리키지 않는다는 것입니다. E42와 E49는 문자당 정보량이 좋지만, E43처럼 긴 학습에서 train loss만 크게 내려간 실험은 final checkpoint가 best보다 나빠집니다. 그래서 LLM 보고서에서는 "best", "final", "rebound", "gap"을 함께 적어야 합니다.

## 4. Context Length

질문: NSMC 리뷰 corpus에서 긴 문맥이 실제로 도움이 되는가?

{figure("01_context_length_learning_curve.png", "context length learning curve")}

선형적으로 읽는 순서:

1. E01, E00, E02, E03의 validation curve가 초반부터 어떤 순서로 내려가는지 본다.
2. `context_length=64`가 best까지 가장 낮게 가는지 확인한다.
3. 긴 context에서 gap이 작아지는지, 아니면 loss만 나빠지는지 본다.
4. throughput 숫자는 보조로만 본다. context가 길어져도 품질이 나빠지면 primary 결론은 품질 쪽이다.

{markdown_table(context, ["experiment_id", "context_length", "best_val_loss", "final_val_loss", "final_val_bits_per_char", "final_generalization_gap", "tokens_per_sec_after_warmup"], 10)}

해석: 이 데이터에서는 `context_length=64`가 가장 낮은 validation curve를 보입니다. 긴 context는 더 많은 위치/attention 비용을 쓰지만, 짧은 영화 리뷰 corpus에서는 그만큼의 장거리 문맥 이득이 작습니다. `context_length=192/256`은 더 빠른 throughput처럼 보이는 구간도 있지만, 품질 곡선은 baseline보다 좋지 않습니다.

결론 라벨: `confirmed`, 단 현재 corpus 길이 분포에 한정

## 5. Vocab Size

질문: vocab이 커질수록 LM 품질이 좋아지는가, 아니면 token loss scale만 바뀌는가?

{figure("02_vocab_bits_per_char_learning_curve.png", "vocab bits per char learning curve")}

vocab 실험은 token-level `final_val_loss`로 직접 비교하면 안 됩니다. vocab이 커지면 같은 문장을 더 적은 token으로 표현하므로 cross entropy의 단위가 바뀝니다. 따라서 primary metric은 `final_val_bits_per_char`입니다.

선형적으로 읽는 순서:

1. 위 패널에서 `estimated_chars_seen -> val_bits_per_char`를 먼저 본다.
2. 중간 패널의 token-level `val_loss`는 참고만 한다. vocab이 바뀌면 loss scale이 바뀐다.
3. 아래 패널에서 `tokens/char`가 vocab 증가에 따라 어떻게 줄어드는지 본다.
4. 같은 raw text를 몇 token으로 압축했는지와, 그 압축이 bits/char를 낮췄는지를 함께 본다.

{markdown_table(vocab, ["experiment_id", "vocab_size", "parameter_count", "val_tokens_per_char", "val_chars_per_token", "final_val_loss", "final_val_bits_per_char", "tokens_per_sec_after_warmup"], 10)}

- token loss만 보면 E04, vocab 2000이 좋아 보입니다.
- 문자당 정보량인 `bits/char`로 보면 E06, vocab 5000이 가장 낮습니다.
- 즉 원본식으로 "vocab 2000이 best"라고 말하면 LLM 지표 기준에서는 틀립니다.
- 다만 vocab 5000은 parameter_count와 LM head 비용이 커지므로 결론은 `confirmed`가 아니라 `trade-off`입니다.

결론 라벨: `trade-off`

## 6. Embedding Dimension

질문: embedding 폭을 키우면 표현력이 좋아지는가, 아니면 작은 corpus에서 비용만 늘어나는가?

{figure("03_emb_dim_capacity_path.png", "embedding dimension capacity path")}

선형적으로 읽는 순서:

1. x축을 `compute_proxy_param_tokens`로 둔다.
2. E07, E00, E08의 경로가 compute를 더 쓸수록 bits/char를 얼마나 낮추는지 본다.
3. 같은 그래프에서 gap이 함께 커지는지 본다.
4. 개선량이 parameter 증가를 정당화하는지 판단한다.

{markdown_table(emb, ["experiment_id", "emb_dim", "parameter_count", "best_val_bits_per_char", "final_val_bits_per_char", "final_generalization_gap", "compute_proxy_param_tokens"], 10)}

해석: `emb_dim=256`은 bits/char를 개선하지만 parameter와 compute가 크게 늘어납니다. 작은 corpus에서 무조건 폭을 키우는 결론이 아니라, 비용 대비 개선폭을 함께 보는 `trade-off`입니다.

결론 라벨: `trade-off`

## 7. 초기 5 Epoch 구조 탐색

원본 보고서에는 5 epoch 기준의 빠른 구조 탐색도 들어 있습니다. 이 실험들은 결론을 확정하기보다 후보를 걸러내는 용도로 봐야 합니다.

### 7.1 n_heads 5ep

{markdown_table(n_heads_short, ["experiment_id", "n_heads", "best_val_loss", "final_val_loss", "final_val_bits_per_char", "final_generalization_gap"], 10)}

해석: E09, E00, E10의 5 epoch 결과는 차이가 작습니다. 그래서 원본 보고서가 다시 10 epoch n_heads sweep을 수행한 것은 타당합니다. 5 epoch만으로 head 수 결론을 내리면 안 됩니다.

### 7.2 n_layers 5ep

{markdown_table(layers_short, ["experiment_id", "n_layers", "parameter_count", "best_val_loss", "final_val_loss", "final_val_bits_per_char", "tokens_per_sec_after_warmup"], 10)}

해석: 5 epoch에서는 4-layer baseline이 가장 균형적입니다. 하지만 이 결과만으로 "깊이를 늘리면 나쁘다"라고 말할 수 없습니다. 20 epoch와 norm 위치까지 함께 봐야 합니다.

### 7.3 FFN Multiplier 5ep

{markdown_table(ffn_short, ["experiment_id", "ffn_mult", "parameter_count", "best_val_loss", "final_val_loss", "final_val_bits_per_char", "compute_proxy_param_tokens"], 10)}

해석: 5 epoch FFN 결과도 차이가 작습니다. 이 축은 10 epoch 재실험의 compute path를 더 신뢰해야 합니다.

## 8. n_heads 10 Epoch

질문: head 수를 늘리면 attention 관점이 늘어 성능이 좋아지는가, 아니면 head_dim이 줄어 손해인가?

{figure("04_n_heads_learning_curve.png", "n heads learning curve")}

선형적으로 읽는 순서:

1. E31, E32, E28, E33, E34의 validation line을 같은 x축에서 본다.
2. 같은 `emb_dim=192`에서 `n_heads`가 커지면 `head_dim`은 줄어든다는 점을 같이 적는다.
3. curve 차이가 작으면 seed variance 없이 강한 결론을 피한다.

{markdown_table(n_heads_long, ["experiment_id", "n_heads", "best_val_loss", "final_val_loss", "final_generalization_gap", "tokens_per_sec_after_warmup"], 10)}

해석: E32의 2-head가 가장 낮지만 차이는 작습니다. 이 축은 "많은 head가 항상 좋다"가 아니라 "head 수와 head_dim의 trade-off"입니다.

결론 라벨: `inconclusive`, 후보는 `n_heads=2`

## 9. FFN Multiplier 10 Epoch

질문: FFN 내부 폭을 키우면 token별 비선형 변환 능력이 실제로 좋아지는가?

{figure("07_ffn_multiplier_capacity_path.png", "ffn multiplier compute path")}

선형적으로 읽는 순서:

1. x축을 compute proxy로 둔다.
2. `ffn_mult=2,4,6`이 compute를 더 쓰면서 bits/char를 얼마나 낮추는지 본다.
3. 차이가 작으면 단일 seed 결과로 확정하지 않는다.

{markdown_table(ffn_long, ["experiment_id", "ffn_mult", "parameter_count", "best_val_bits_per_char", "final_val_bits_per_char", "final_generalization_gap", "compute_proxy_param_tokens"], 10)}

해석: FFN을 키우면 최종 bits/char가 소폭 좋아지지만 차이는 작고 compute가 늘어납니다. seed 반복 없이는 강한 결론보다 `inconclusive/trade-off`가 맞습니다.

결론 라벨: `trade-off`

## 10. Depth와 Norm 안정성

질문은 "Pre-LN이 좋은가?"가 아니라 "깊이가 늘어날 때 Pre-LN이 안정성을 주는가?"입니다.

{figure("05_n_layers_depth_curve.png", "depth curve")}

{figure("06_norm_depth_stability_curve.png", "norm depth stability curve")}

선형적으로 읽는 순서:

1. 먼저 4-layer post-LN baseline E28과 pre-LN E27을 비교한다.
2. 다음으로 8-layer post/pre E37/E38, E45/E46을 본다.
3. 마지막으로 12-layer post/pre E47/E48과 regularized E62/E63을 본다.
4. post-LN 12-layer가 실패하는지, pre-LN이 정상 수렴하는지, regularization으로 post-LN이 회복되는지 순서대로 판단한다.

10 epoch norm/depth:

{markdown_table(norm_short, ["experiment_id", "n_layers", "norm_first", "drop_rate", "num_epochs", "best_val_loss", "final_val_loss", "final_generalization_gap"], 10)}

20 epoch 및 regularized 조건:

{markdown_table(norm_long, ["experiment_id", "n_layers", "norm_first", "drop_rate", "num_epochs", "best_val_loss", "final_val_loss", "final_minus_best_val_loss", "final_generalization_gap"], 20)}

- 질문은 "Pre-LN이 좋은가?"가 아니라 "깊이가 늘어날 때 Pre-LN이 안정성을 주는가?"입니다.
- 12-layer post-LN E47은 loss 7대에 머물러 최적화 실패에 가깝습니다.
- 12-layer pre-LN E48은 정상적으로 수렴해 깊은 모델에서 Pre-LN의 안정성 근거가 됩니다.
- 하지만 E60~E63처럼 `drop_rate=0.2`, `lr=0.0002`로 안정화하면 post-LN도 회복됩니다.
- 따라서 결론은 "Pre-LN 절대 우위"가 아니라 "깊이 증가와 불안정 lr/dropout 조건에서 Pre-LN이 안정성 guardrail을 제공한다"입니다.

결론 라벨: `trade-off`

## 11. Dropout과 Epoch

질문: 오래 학습하면 validation도 계속 좋아지는가, 아니면 어느 시점부터 train만 좋아지는가?

{figure("09_dropout_overfit_curve.png", "dropout overfit curve")}

{figure("13_epoch_overfit_rebound_audit.png", "epoch rebound audit")}

선형적으로 읽는 순서:

1. 5 epoch dropout에서는 final loss 차이가 작고, dropout이 학습 속도에 미치는 영향을 본다.
2. 20 epoch dropout에서는 `best_val_loss`, `final_val_loss`, `final_minus_best_val_loss`, `final_generalization_gap`을 본다.
3. train loss가 계속 내려가는 것은 정상으로 간주한다.
4. validation이 best 이후 올라가면 rebound로 표시한다.
5. gap이 커지면 작은 corpus에서 memorization 위험이 커진 것으로 해석한다.

5 epoch dropout:

{markdown_table(dropout_short, ["experiment_id", "drop_rate", "best_val_loss", "final_val_loss", "final_val_bits_per_char", "final_generalization_gap"], 10)}

20 epoch dropout:

{markdown_table(dropout_long, ["experiment_id", "drop_rate", "num_epochs", "best_val_loss", "final_val_loss", "final_minus_best_val_loss", "final_generalization_gap", "gap_slope_last_quarter"], 10)}

- E21 dropout 0.0은 20 epoch에서 final gap과 rebound가 큽니다.
- E24 dropout 0.2는 final이 best와 같고 gap도 가장 작아, 이번 corpus에서는 장기 학습 과적합 억제에 가장 유리합니다.
- "1500 epoch까지 과적합이 없었다"는 주장은 epoch 숫자만으로 비교할 수 없습니다. `tokens_seen`, `gap`, `rebound`, `generation repetition`이 같이 있어야 합니다.

결론 라벨: `confirmed`

## 12. QKV Bias

질문: attention projection에 bias를 넣는 것이 의미 있는 개선인가?

{figure("10_qkv_bias_curve.png", "qkv bias curve")}

5 epoch qkv:

{markdown_table(qkv_short, ["experiment_id", "n_layers", "qkv_bias", "best_val_loss", "final_val_loss", "final_generalization_gap"], 10)}

10 epoch/depth qkv:

{markdown_table(qkv_long, ["experiment_id", "n_layers", "qkv_bias", "best_val_loss", "final_val_loss", "final_generalization_gap"], 10)}

해석: qkv bias는 4-layer에서는 미세하게 좋아 보이나 8-layer에서는 악화됩니다. 효과 크기가 작고 방향도 조건에 따라 달라져 `inconclusive`입니다.

결론 라벨: `inconclusive`

## 13. Weight Tying

질문: weight tying은 단기 수렴을 돕는가, 아니면 장기 일반화 regularization인가?

{figure("11_weight_tying_long_curve.png", "weight tying long curve")}

5 epoch weight tying:

{markdown_table(weight_short, ["experiment_id", "weight_tying", "num_epochs", "parameter_count", "best_val_loss", "final_val_loss", "final_generalization_gap"], 10)}

10/20/50 epoch weight tying:

{markdown_table(weight_long, ["experiment_id", "weight_tying", "num_epochs", "parameter_count", "best_val_loss", "final_val_loss", "final_minus_best_val_loss", "final_generalization_gap"], 10)}

해석:

- weight tying은 5/10 epoch에서는 느리게 보일 수 있습니다.
- 장기 50 epoch에서는 E44가 E43보다 final loss, rebound, gap이 모두 낫습니다.
- 즉 weight tying은 단기 수렴 가속 옵션이 아니라 작은 corpus에서 parameter를 줄이고 장기 일반화를 안정화하는 regularization으로 해석하는 편이 맞습니다.

결론 라벨: `confirmed`

## 14. Activation과 Seed Variance

질문: GELU/ReLU/SiLU 차이는 seed variance를 넘어서는가?

{figure("08_activation_seed_mean_std_curve.png", "activation seed mean std curve")}

선형적으로 읽는 순서:

1. 개별 seed 선을 먼저 본다.
2. activation별 평균 선을 본다.
3. std band가 평균 차이보다 큰지 확인한다.
4. final뿐 아니라 best checkpoint도 함께 비교한다.

{markdown_table(activation_summary, ["activation", "seed_count", "mean_best_val_bits_per_char", "std_best_val_bits_per_char", "mean_final_val_bits_per_char", "std_final_val_bits_per_char"], 10)}

해석:

- activation은 seed 반복이 있으므로 평균과 표준편차로만 말해야 합니다.
- 평균 기준 GELU가 가장 낮고 ReLU가 그 다음, SiLU가 가장 높습니다.
- 단일 seed 결과만 보고 activation 순위를 말하면 안 됩니다.

결론 라벨: `confirmed`, 현재 depth8/pre-LN/20epoch 조건에 한정

## 15. Stride

질문: stride를 줄여 overlap을 늘린 것이 성능 개선인가, 아니면 같은 데이터를 더 자주 본 효과인가?

{figure("12_stride_overlap_curve.png", "stride overlap curve")}

선형적으로 읽는 순서:

1. `tokens_seen -> val_loss`를 본다.
2. `estimated_chars_seen -> val_loss`를 다시 본다.
3. overlap으로 window 수가 늘어난 효과를 데이터 증가로 착각하지 않는다.
4. gap과 compute 증가를 함께 본다.

{markdown_table(stride, ["experiment_id", "stride", "best_val_loss", "final_val_loss", "final_generalization_gap", "final_tokens_seen", "compute_proxy_param_tokens"], 10)}

해석:

- stride 64는 validation loss를 낮추지만, overlap으로 학습 window 수가 늘어난 효과가 섞여 있습니다.
- 따라서 "데이터가 늘었다"가 아니라 "같은 raw corpus를 더 촘촘히 반복해서 봤다"로 해석해야 합니다.
- `tokens_seen`와 `estimated_chars_seen` 기준 그래프를 같이 봐야 착시를 줄일 수 있습니다.

결론 라벨: `trade-off`

## 16. Epoch 단위 실험 기준

이번 로그를 기준으로 epoch 실험은 다음 단위가 적절합니다.

| 목적 | epoch 단위 | 판단 지표 |
| --- | ---: | --- |
| 빠른 후보 제거 | 5 | final/best val loss, throughput |
| 구조 재검증 | 10 | full learning curve, gap |
| 과적합/dropout/norm | 20 | best-final rebound, gap slope |
| 장기 일반화/weight tying | 50 | final-minus-best, generation repetition 필요 |

epoch를 더 길게 늘릴 때는 단순히 train loss가 내려가는지 보지 않습니다. validation best가 언제 찍혔는지, 이후 얼마나 되올랐는지, gap이 얼마나 커졌는지를 먼저 봐야 합니다. 1500 epoch 같은 긴 실험은 `epoch`보다 `tokens_seen`과 `compute_proxy` 기준으로 다시 정렬해야 옆 팀 결과와 공정하게 비교할 수 있습니다.

## 17. 최종 결론

1. vocab 실험은 원본 결론을 바꿔야 합니다. token loss 기준 best는 vocab 2000이지만, LLM식 공정 지표인 bits/char 기준 best는 vocab 5000입니다.
2. 작은 corpus에서 epoch를 늘리면 train loss는 계속 내려가지만, dropout이 낮은 조건은 gap과 rebound가 커집니다.
3. Pre-LN은 "항상 성능이 좋다"가 아니라 깊이가 늘거나 post-LN이 불안정한 조건에서 안정성을 주는 옵션입니다.
4. weight tying은 단기 성능 옵션보다 장기 일반화 regularization으로 해석하는 것이 맞습니다.
5. activation, qkv_bias처럼 차이가 작은 축은 seed variance 또는 조건 interaction을 함께 봐야 합니다.
6. 앞으로 추가 실험은 final loss 표가 아니라 `history.jsonl`과 sequential graph를 먼저 남겨야 합니다.

## 18. 다음 실험으로 넘어가기 전 체크리스트

- vocab 실험은 반드시 tokenizer profile, `tokens/char`, `bits/char`를 같이 기록한다.
- epoch 실험은 final loss 하나가 아니라 best/final/rebound/gap을 기록한다.
- seed 고정은 기본이고, 최종 주장으로 쓸 축은 seed 반복을 추가한다.
- activation, qkv_bias처럼 효과가 작은 축은 단일 seed 결론을 쓰지 않는다.
- 깊이 실험은 norm 위치, lr, dropout을 같이 묶어 봐야 한다.
- 모든 새 실험은 `history.jsonl`과 sequential graph가 생성되지 않으면 REPORT에 쓰지 않는다.
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
