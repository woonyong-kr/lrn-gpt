# REPORT.md HY 실험 재로깅 및 LLM 지표 그래프 갱신 지시서

이 문서는 원본 `REPORT.md`를 수정하지 않고, `REPORT.md`를 LLM 개발자 관점의 지표와 그래프로 다시 작성하기 위한 작업 지시서다.

작업자는 `REPORT.md`에 이미 들어간 HY/NSMC 실험을 기준으로 실험 로그를 다시 찍고, 그 로그에서 순차 선형 그래프를 다시 만들고, 최종 해석을 갱신해야 한다.

## 0. 절대 기준

- `REPORT.md`는 원본으로 보존한다.
- 이번 작업의 기준 실험은 `REPORT.md`와 `docs/HY/testresult/*.md`에 등장하는 E00~E63 HY 실험이다.
- `docs/llm_10x/*`는 이번 REPORT 재작성 기준이 아니다. 10x 결과를 섞어서 결론을 쓰지 않는다.
- final loss 한 점만 모은 막대/산점도는 보조 자료로만 사용한다.
- 본문 결론은 반드시 `epoch`, `step`, `tokens_seen`에 따른 순차 선형 그래프에서 나온다.
- 기존 로그에 필요한 지표가 없으면 해당 REPORT 실험을 같은 config/seed/split로 다시 실행해 로그를 보강한다.
- tokenizer가 다른 실험은 token-level loss로 직접 비교하지 않는다. vocab 실험의 primary metric은 `bits/char`다.
- "과적합이 없다"는 주장은 train loss 하락 여부가 아니라 `train-val gap`, `best-final rebound`, `val_bits_per_char` 곡선으로 판단한다.

## 1. 대상 실험 매니페스트

먼저 `REPORT.md`에서 언급한 모든 실험을 매니페스트로 고정한다.

생성 파일:

```text
docs/HY/llm_relog/experiment_manifest.csv
```

필수 컬럼:

```text
experiment_id,source_md,report_section,comparison_group,baseline_id,
changed_variable,changed_value,seed,vocab_size,context_length,emb_dim,
n_heads,n_layers,ffn_mult,activation,drop_rate,norm_first,qkv_bias,
weight_tying,stride,num_epochs,batch_size,lr,weight_decay,
rerun_required,rerun_reason
```

대상 실험군은 아래를 빠뜨리지 않는다.

| REPORT 축 | 기준/비교 실험 |
| --- | --- |
| 5 epoch baseline | E00 |
| context_length | E01, E00, E02, E03 |
| vocab_size | E04, E00, E05, E06 |
| emb_dim | E07, E00, E08 |
| 초기 n_heads 5ep | E09, E00, E10 |
| n_layers 5ep | E11, E00, E12, E13 |
| ffn_multiplier 5ep | E14, E00, E15 |
| drop_rate 5ep | E16, E17, E00, E18 |
| qkv_bias 5ep | E00, E19 |
| weight_tying 5ep | E00, E20 |
| 10 epoch baseline | E28 |
| activation 10ep | E25, E28, E26 |
| norm_first 10ep | E28, E27 |
| stride 10ep | E28, E29 |
| n_heads 10ep | E31, E32, E28, E33, E34 |
| ffn_multiplier 10ep | E35, E28, E36 |
| depth/norm 10ep | E28, E37, E38 |
| qkv_bias 10ep | E28, E39, E37, E40 |
| weight_tying 10/20/50ep | E28, E41, E42, E43, E44 |
| long dropout 20ep | E21, E22, E23, E24 |
| depth/norm 20ep | E23, E45, E46, E47, E48 |
| activation seed 반복 | E49, E50, E51, E52, E53, E54, E55, E56, E57 |
| regularized depth/norm | E60, E61, E62, E63 |

매니페스트 검증 규칙:

- `REPORT.md`에 언급된 실험 ID가 매니페스트에 없으면 실패다.
- 매니페스트에 있는 실험의 `source_md`가 없으면 실패다.
- `source_md`에 step별 CSV가 없거나 LLM 지표 계산에 필요한 `train_chars`, `val_chars`, `train_tokens`, `val_tokens`, `parameter_count`, `tokens_seen`이 없으면 `rerun_required=True`로 둔다.
- 재실행이 필요한 실험은 기존 REPORT config와 seed를 그대로 복원한다.

## 2. 로그 재기록 산출물

각 실험은 아래 구조로 다시 저장한다.

```text
docs/HY/llm_relog/runs/{experiment_id}/config.json
docs/HY/llm_relog/runs/{experiment_id}/tokenizer_profile.json
docs/HY/llm_relog/runs/{experiment_id}/history.jsonl
docs/HY/llm_relog/runs/{experiment_id}/metrics.json
docs/HY/llm_relog/runs/{experiment_id}/samples.jsonl
docs/HY/llm_relog/runs/{experiment_id}/result.md
```

집계 파일:

```text
docs/HY/llm_relog/report_llm_metrics.csv
docs/HY/llm_relog/report_llm_metrics_by_step.csv
docs/HY/llm_relog/report_llm_summary_by_condition.csv
```

### 2.1 `history.jsonl` 필수 필드

`history.jsonl`은 evaluation 시점마다 한 줄씩 기록한다. epoch final만 기록하면 안 된다.

```json
{
  "experiment_id": "E34",
  "epoch": 10,
  "step": 1960,
  "tokens_seen": 8028160,
  "estimated_chars_seen": 13794860.0,
  "train_loss": 4.633036,
  "val_loss": 5.122543,
  "test_loss": 5.122543,
  "train_nats_per_char": 2.7041,
  "val_nats_per_char": 2.9907,
  "train_bits_per_char": 3.9012,
  "val_bits_per_char": 4.3146,
  "train_val_gap": 0.489507,
  "best_val_loss_so_far": 5.122543,
  "best_val_bits_per_char_so_far": 4.3146,
  "val_loss_minus_best_so_far": 0.0,
  "lr": 0.0004,
  "grad_norm": 1.160649,
  "elapsed_sec": 41.1085,
  "tokens_per_sec": 195291.946,
  "tokens_per_sec_after_warmup": 195291.946,
  "parameter_count": 2954112,
  "compute_proxy_param_tokens": 23716083793920,
  "estimated_train_flops": 142296502763520
}
```

### 2.2 `metrics.json` 필수 필드

`metrics.json`은 run 전체의 요약이다.

```json
{
  "experiment_id": "E34",
  "seed": 123,
  "final_step": 1960,
  "final_epoch": 10,
  "final_tokens_seen": 8028160,
  "final_train_loss": 4.633036,
  "final_val_loss": 5.122543,
  "final_val_bits_per_char": 4.314626,
  "best_step": 1960,
  "best_epoch": 10,
  "best_tokens_seen": 8028160,
  "best_val_loss": 5.122543,
  "best_val_bits_per_char": 4.314626,
  "final_minus_best_val_loss": 0.0,
  "final_minus_best_val_bits_per_char": 0.0,
  "final_generalization_gap": 0.489507,
  "gap_slope_last_quarter": 0.0,
  "overfit_score": 0.0,
  "parameter_count": 2954112,
  "tokens_per_parameter": 2.718,
  "compute_proxy_param_tokens": 23716083793920,
  "estimated_train_flops": 142296502763520,
  "tokens_per_sec_after_warmup": 195291.946
}
```

### 2.3 `tokenizer_profile.json` 필수 필드

vocab 실험은 tokenizer 자체가 바뀌므로 tokenizer profile을 반드시 저장한다.

```json
{
  "requested_vocab_size": 5000,
  "actual_vocab_size": 5000,
  "bpe_merge_count": 4744,
  "special_token_count": 0,
  "train_chars": 1379486,
  "val_chars": 120560,
  "train_tokens": 712949,
  "val_tokens": 62493,
  "train_tokens_per_char": 0.5168,
  "val_tokens_per_char": 0.5184,
  "train_chars_per_token": 1.9347,
  "val_chars_per_token": 1.9292,
  "token_length_histogram": {"1": 112, "2": 1800, "3": 2100},
  "top_50_merge_rules": [["영", "화"], ["재", "미"]],
  "byte_fallback_ratio": 0.0
}
```

## 3. LLM 전문 지표 정의

아래 지표는 코드에 계산 로직을 추가하고 모든 결과 표/그래프에서 같은 이름으로 사용한다.

| 지표 | 계산식 | 의미 | 해석 기준 |
| --- | --- | --- | --- |
| `tokens_per_char` | `token_count / char_count` | tokenizer가 문자를 얼마나 잘게 쪼개는지 | 낮을수록 같은 텍스트를 적은 token으로 표현한다 |
| `chars_per_token` | `char_count / token_count` | token 하나의 평균 raw text 길이 | 커질수록 긴 token/희소 token 위험이 생긴다 |
| `nats_per_char` | `token_loss * tokens_per_char` | 문자당 negative log likelihood | tokenizer가 다른 실험 비교의 기본 단위 |
| `bits_per_char` | `nats_per_char / ln(2)` | 문자당 정보량 | vocab 실험의 primary metric |
| `best_val_loss` | `min(val_loss)` | 가장 좋은 checkpoint | final보다 중요할 수 있다 |
| `best_val_bits_per_char` | `min(val_bits_per_char)` | 문자 기준 best checkpoint | vocab/토큰화 비교에 사용 |
| `final_minus_best_val_loss` | `final_val_loss - best_val_loss` | 후반 rebound | 크면 마지막 checkpoint가 과적합/퇴화했다 |
| `final_generalization_gap` | `final_val_loss - final_train_loss` | train/validation 차이 | 장기 학습과 dropout에서 핵심 |
| `gap_slope_last_quarter` | 마지막 25% step의 gap 기울기 | gap이 후반에 커지는지 | 양수면 과적합 위험 증가 |
| `tokens_seen` | 학습 중 누적 token | 학습량 | epoch보다 공정한 x축 |
| `estimated_chars_seen` | `tokens_seen * train_chars_per_token` | raw text 노출량 근사 | vocab이 다른 경우 보조 x축 |
| `parameter_count` | trainable parameter 수 | 모델 용량 | capacity 실험의 비용 축 |
| `compute_proxy_param_tokens` | `parameter_count * tokens_seen` | 간단한 compute proxy | loss-vs-compute 그래프 x축 |
| `estimated_train_flops` | `6 * parameter_count * tokens_seen` | 대략 학습 FLOPs | 정확 FLOPs가 아닌 비교용 근사 |
| `tokens_per_parameter` | `tokens_seen / parameter_count` | 데이터/모델 비율 | 작으면 데이터 부족/과적합 위험 |
| `tokens_per_sec_after_warmup` | warmup 제외 평균 throughput | 처리량 | context/depth/ffn 비용 비교 |
| `seed_mean`, `seed_std`, `seed_iqr` | seed 반복 통계 | 재현성 | 평균 차이가 std보다 작으면 주장 금지 |
| `paired_delta_to_baseline` | 같은 seed에서 조건 - baseline | baseline 대비 효과 | seed variance를 줄여 비교한다 |

## 4. 코드 수정 요구사항

기존 runner가 없거나 현재 repo 이름과 다르면 새 script를 만들고 아래 기능을 구현한다.

필수 script:

```text
scripts/hy_llm_relog_manifest.py
scripts/run_hy_llm_relog.py
scripts/aggregate_hy_llm_relog.py
scripts/generate_hy_llm_seq_figures.py
scripts/render_report_temp_from_hy_llm_metrics.py
scripts/verify_hy_llm_report_assets.py
```

권장 실행 순서:

```bash
python scripts/hy_llm_relog_manifest.py \
  --report REPORT.md \
  --source docs/HY/testresult \
  --out docs/HY/llm_relog/experiment_manifest.csv

python scripts/run_hy_llm_relog.py \
  --manifest docs/HY/llm_relog/experiment_manifest.csv \
  --out docs/HY/llm_relog/runs \
  --seed-policy fixed

python scripts/aggregate_hy_llm_relog.py \
  --runs docs/HY/llm_relog/runs \
  --out docs/HY/llm_relog

python scripts/generate_hy_llm_seq_figures.py \
  --metrics docs/HY/llm_relog/report_llm_metrics.csv \
  --history docs/HY/llm_relog/report_llm_metrics_by_step.csv \
  --out docs/HY/figures_llm_seq

python scripts/render_report_temp_from_hy_llm_metrics.py \
  --report REPORT.md \
  --metrics docs/HY/llm_relog/report_llm_metrics.csv \
  --history docs/HY/llm_relog/report_llm_metrics_by_step.csv \
  --figures docs/HY/figures_llm_seq \
  --out REPORT_temp.md

python scripts/verify_hy_llm_report_assets.py \
  --report REPORT_temp.md \
  --figures docs/HY/figures_llm_seq
```

runner는 기존 HY 실험 config를 그대로 복원해야 한다. 실험을 새로 설계하지 말고, `REPORT.md`에 있는 실험을 같은 조건으로 재실행한다.

seed 규칙:

- 기존 REPORT 재현 run은 seed를 원래 값으로 고정한다. 기본은 `123`.
- seed 반복이 이미 있는 activation 실험은 `42`, `123`, `2026`을 유지한다.
- 최종 주장으로 바꾸고 싶은 축은 같은 조건으로 `42`, `123`, `2026` 세 seed를 추가한다.
- seed가 다르면 같은 그래프에 옅은 개별 선을 그리고, 평균은 굵은 선, 표준편차/IQR은 band로 표시한다.

## 5. 순차 선형 그래프 요구사항

그래프 저장 경로:

```text
docs/HY/figures_llm_seq/*.png
docs/HY/figures_llm_seq/figure_index.md
```

공통 규칙:

- primary x축은 `tokens_seen`이다.
- 같은 tokenizer/같은 epoch budget 안에서만 `epoch`를 보조 x축으로 쓸 수 있다.
- vocab 실험은 `val_loss` 단독 그래프를 결론 근거로 쓰지 않는다. `val_bits_per_char`가 primary다.
- 모든 그래프에는 baseline, best checkpoint, final checkpoint를 표시한다.
- long run 그래프는 `val_loss_minus_best_so_far` 또는 `final_minus_best` 계열 rebound 곡선을 포함한다.
- train/val 곡선을 분리해서 그리고, 같은 그림 또는 인접 subplot에 `train_val_gap`을 그린다.
- final-only bar chart는 본문 근거가 아니라 appendix/summary로만 둔다.
- 색상은 같은 축에서 같은 값이 항상 같은 색을 갖게 한다.
- PNG는 비어 있으면 안 된다. 파일 크기, 이미지 크기, non-white pixel ratio를 검사한다.

필수 그래프:

| 파일명 | 목적 | x축 | y축/패널 | 포함 실험 |
| --- | --- | --- | --- | --- |
| `01_context_length_learning_curve.png` | 문맥 길이별 수렴 비교 | tokens_seen | val_loss, train_loss, gap, tokens/sec | E01,E00,E02,E03 |
| `02_vocab_bits_per_char_learning_curve.png` | vocab별 공정 비교 | estimated_chars_seen 또는 tokens_seen | val_bits_per_char, val_loss 보조, tokens_per_char | E04,E00,E05,E06 |
| `03_emb_dim_capacity_path.png` | 폭 증가와 compute 효율 | compute_proxy_param_tokens | val_bits_per_char, train_val_gap | E07,E00,E08 |
| `04_n_heads_learning_curve.png` | head 수와 head_dim 비교 | tokens_seen | val_loss, gap, throughput | E31,E32,E28,E33,E34 |
| `05_n_layers_depth_curve.png` | 깊이 증가와 수렴 안정성 | tokens_seen | val_loss, gap, throughput | E11,E00,E12,E13,E37,E45,E47 |
| `06_norm_depth_stability_curve.png` | 깊이가 늘 때 Pre-LN이 안정성을 주는지 | tokens_seen | val_loss, gap, rebound | E28,E27,E37,E38,E45,E46,E47,E48,E60,E61,E62,E63 |
| `07_ffn_multiplier_capacity_path.png` | FFN 폭과 비용 대비 품질 | compute_proxy_param_tokens | val_bits_per_char, tokens/sec | E35,E28,E36 |
| `08_activation_seed_mean_std_curve.png` | activation 차이와 seed variance | tokens_seen | mean val_bits_per_char + std band, individual seeds | E49~E57 |
| `09_dropout_overfit_curve.png` | dropout별 과적합 억제 | tokens_seen | train_loss, val_loss, gap, rebound | E16,E17,E00,E18,E21,E22,E23,E24 |
| `10_qkv_bias_curve.png` | qkv bias 효과가 의미 있는지 | tokens_seen | val_loss delta, gap | E28,E39,E37,E40 |
| `11_weight_tying_long_curve.png` | weight tying 장기 일반화 | tokens_seen | val_loss, gap, rebound, params | E28,E41,E42,E43,E44 |
| `12_stride_overlap_curve.png` | overlap 증가와 학습량 착시 제거 | tokens_seen, estimated_chars_seen | val_loss, gap, throughput | E28,E29 |
| `13_epoch_overfit_rebound_audit.png` | epoch 증가가 언제부터 손해인지 | epoch/tokens_seen | best_so_far, final-best rebound, gap | E21~E24,E43,E44 |
| `14_loss_vs_compute_all_paths.png` | 전체 실험 compute 대비 성능 | compute_proxy_param_tokens | val_bits_per_char path | 모든 HY 실험 |

각 그래프는 선형 그래프가 기본이다. 실험별 final 점만 모은 그래프는 같은 파일명에 `_summary`를 붙여 별도로 둔다.

## 6. 축별 해석 기준

### 6.1 context_length

질문: 긴 문맥이 현재 NSMC corpus에서 실제로 도움이 되는가?

Primary metric:

- 같은 tokenizer이므로 `val_loss`와 `val_bits_per_char` 모두 볼 수 있다.

반드시 볼 것:

- `tokens_seen -> val_loss` 선형 곡선
- `tokens_seen -> train_val_gap`
- `context_length -> tokens_per_sec_after_warmup`

해석:

- 짧은 context가 이기면 "짧은 리뷰 corpus에서 필요한 문맥 길이가 작다"로 해석한다.
- 긴 context가 느리면서 val 개선이 없으면 비용 대비 손해다.
- final loss만 낮거나 높다고 결론 내리지 말고, 수렴 속도와 best 시점도 같이 본다.

### 6.2 vocab_size

질문: vocab이 커질수록 언어 모델 품질이 좋아지는가, 아니면 token loss scale만 바뀌는가?

Primary metric:

- `val_bits_per_char`

Guardrail:

- `val_tokens_per_char`
- `actual_vocab_size`
- `bpe_merge_count`
- `token_length_histogram`
- `parameter_count`
- `tokens_per_sec_after_warmup`

해석:

- `final_val_loss` 기준 best와 `val_bits_per_char` 기준 best가 다르면 `val_bits_per_char`를 우선한다.
- 예: vocab 2000이 token loss에서 좋아 보여도 bits/char에서 vocab 5000이 낮으면 "vocab 2000 best"라고 쓰면 안 된다.
- 큰 vocab이 bits/char를 낮추지만 parameter/throughput 비용이 크면 `trade-off`로 쓴다.
- tokenizer profile 없이 vocab 결론을 내리면 불합격이다.

### 6.3 emb_dim, n_layers, ffn_multiplier

질문: 모델 용량을 늘리면 품질이 좋아지는가, 아니면 데이터/compute 대비 손해인가?

Primary metric:

- `val_bits_per_char` 또는 같은 tokenizer 내 `val_loss`

반드시 볼 것:

- `compute_proxy_param_tokens -> val_bits_per_char` 선형 path
- `parameter_count -> best_val_bits_per_char`
- `tokens_per_parameter`
- `tokens_per_sec_after_warmup`
- `train_val_gap`

해석:

- 더 큰 모델이 val을 낮추지만 gap도 커지면 `trade-off`다.
- 더 큰 모델이 compute를 많이 쓰고 val이 거의 같으면 `rejected` 또는 `inconclusive`다.
- 5 epoch에서 안 좋았던 depth가 20 epoch에서 좋아지면 "짧은 학습에서 결론 불가"라고 쓴다.

### 6.4 n_heads

질문: head 수를 늘리는 것이 도움이 되는가, 아니면 head_dim 축소가 손해인가?

반드시 볼 것:

- `n_heads`, `head_dim`, `val_loss`를 같은 표에 둔다.
- `tokens_seen -> val_loss` 선형 곡선을 그린다.
- `tokens_per_sec_after_warmup`도 같이 본다.

해석:

- 같은 `emb_dim`에서 `n_heads` 증가는 head 수 증가와 head_dim 감소를 동시에 의미한다.
- 차이가 seed variance보다 작으면 `inconclusive`다.

### 6.5 Pre-LN / Post-LN

질문은 "Pre-LN이 좋은가?"가 아니다. 질문은 "깊이가 늘어날 때 Pre-LN이 안정성을 주는가?"다.

Primary metric:

- `tokens_seen -> val_loss`
- `tokens_seen -> train_val_gap`
- `tokens_seen -> val_loss_minus_best_so_far`

반드시 비교할 것:

- 4 layer: E28 vs E27
- 8 layer: E37 vs E38, E45 vs E46, E60 vs E61
- 12 layer: E47 vs E48, E62 vs E63

해석:

- post-LN 12 layer가 loss 7대에서 멈추면 최적화 실패로 표시한다.
- regularized 조건 E60~E63에서 post-LN이 회복되면 "Pre-LN 절대 우위"라고 쓰지 않는다.
- Pre-LN이 val이 더 낮지 않아도 gap이 작으면 안정성 측면의 장점으로 분리해 쓴다.

### 6.6 activation

질문: GELU/ReLU/SiLU 차이가 seed variance를 넘어서는가?

Primary metric:

- seed별 `val_bits_per_char` 곡선의 평균과 표준편차

반드시 볼 것:

- 개별 seed 선
- 평균 선
- std 또는 IQR band
- best checkpoint 평균과 final 평균

해석:

- 평균 차이가 std보다 작으면 결론은 `inconclusive`다.
- 한 seed만 보고 activation 순위를 매기면 불합격이다.

### 6.7 dropout과 epoch

질문: 오래 학습하면 validation이 계속 좋아지는가, 아니면 어느 시점부터 train만 좋아지는가?

Primary metric:

- `best_val_loss`
- `final_minus_best_val_loss`
- `train_val_gap`
- `gap_slope_last_quarter`

반드시 볼 것:

- 5 epoch dropout: E16,E17,E00,E18
- 20 epoch dropout: E21,E22,E23,E24
- 50 epoch weight tying: E43,E44

해석:

- train loss가 계속 내려가는 것은 정상이다. 그것만으로 좋은 학습이라고 말하지 않는다.
- validation이 best 이후 올라가면 rebound로 표시한다.
- dropout이 큰 조건에서 final gap이 작고 rebound가 작으면 과적합 억제 근거다.
- validation이 final까지 계속 내려가면 과적합이 아직 관찰되지 않은 것이지, 데이터가 충분하다는 증거는 아니다.

### 6.8 qkv_bias

질문: qkv bias가 실제 효과인지 노이즈인지 확인한다.

Primary metric:

- paired delta to baseline

해석:

- 차이가 0.001 수준이면 seed 반복 없이는 `inconclusive`다.
- 4 layer와 8 layer에서 방향이 다르면 interaction으로 표시하고 단독 결론을 피한다.

### 6.9 weight_tying

질문: weight tying이 단순히 parameter를 줄이는가, 아니면 장기 일반화에도 도움이 되는가?

반드시 볼 것:

- parameter_count 감소
- `tokens_seen -> val_loss`
- `tokens_seen -> train_val_gap`
- `tokens_seen -> val_loss_minus_best_so_far`
- 10/20/50 epoch 경로

해석:

- 5/10 epoch에서 느리게 학습해도 20/50 epoch에서 rebound가 작으면 장기 regularization으로 해석한다.
- final loss가 아니라 best와 rebound를 함께 써야 한다.

### 6.10 stride

질문: stride를 줄여 overlap을 늘리는 것이 성능 개선인지, 같은 데이터를 더 자주 본 효과인지 분리한다.

Primary metric:

- `tokens_seen -> val_loss`
- `estimated_chars_seen -> val_loss`

해석:

- stride 64가 좋아도 실제 raw corpus 다양성이 늘어난 것은 아니다.
- tokens_seen 증가, 중복 sample 증가, throughput 변화까지 같이 써야 한다.

## 7. epoch 실험 단위

REPORT 재작성에서 epoch는 final 결과 표의 행이 아니라 learning curve의 x축이다.

최소 규칙:

- 모든 실험은 `eval_freq=200` 또는 기존 REPORT의 평가 주기를 유지한다.
- 5 epoch 실험도 epoch 0~5의 모든 평가점을 저장한다.
- 10/20/50 epoch 실험은 final만 보지 말고 best 시점과 rebound를 표시한다.
- long epoch 주장은 5, 10, 20, 50 epoch budget을 분리해서 쓴다.

권장 budget:

| 목적 | epoch budget | 이유 |
| --- | ---: | --- |
| 빠른 구조 탐색 | 5 | 초기 후보 제거 |
| baseline 재검증 | 10 | 구조 차이 수렴 확인 |
| overfit/dropout/norm | 20 | gap과 rebound 관찰 |
| 장기 일반화/weight tying | 50 | best 이후 rebound 관찰 |

1500 epoch처럼 긴 실험을 한다면 다음 조건을 만족해야 한다.

- 모든 evaluation point를 line graph로 남긴다.
- `tokens_seen`, `estimated_chars_seen`, `compute_proxy`를 x축으로 다시 그린다.
- "과적합 없음"은 final loss가 아니라 `best-final rebound`, `gap_slope_last_quarter`, generation repetition 지표가 안정적일 때만 말한다.

## 8. REPORT_temp.md 최종 작성 규칙

실험과 그래프를 다시 만든 뒤 이 파일은 지시서가 아니라 `REPORT.md` 수정본 초안으로 갱신한다.

각 섹션은 아래 구조를 따른다.

```text
### 4.x 실험명

질문:
고정 조건:
변경 변수:
비교 실험:
Primary metric:
Guardrail metric:
순차 그래프:
결과 표:
해석:
결론 라벨: confirmed / trade-off / inconclusive / rejected
```

본문 표는 final scalar만 넣지 않는다. 최소한 아래 컬럼을 포함한다.

```text
experiment_id, changed_value, seed, parameter_count,
best_epoch, best_tokens_seen, best_val_loss, best_val_bits_per_char,
final_val_loss, final_val_bits_per_char,
final_minus_best_val_loss, final_generalization_gap,
tokens_per_sec_after_warmup, compute_proxy_param_tokens
```

vocab 섹션은 추가로 아래 컬럼을 포함한다.

```text
requested_vocab_size, actual_vocab_size, bpe_merge_count,
val_tokens_per_char, val_chars_per_token, token_length_histogram_summary
```

activation 섹션은 추가로 아래 컬럼을 포함한다.

```text
activation, seed_count, mean_best_val_bits_per_char,
std_best_val_bits_per_char, mean_final_val_bits_per_char,
std_final_val_bits_per_char, baseline_win_rate
```

## 9. 그래프 검증 기준

`scripts/verify_hy_llm_report_assets.py`는 아래를 검사한다.

- `REPORT_temp.md`가 `docs/llm_10x`를 본문 결론 근거로 참조하지 않는다.
- `REPORT_temp.md`에 포함된 모든 이미지 경로가 존재한다.
- 모든 PNG가 width/height 0이 아니다.
- 모든 PNG의 non-white pixel ratio가 최소 기준 이상이다.
- 필수 그래프 14개가 모두 존재한다.
- `report_llm_metrics_by_step.csv`에 모든 매니페스트 실험 ID가 있다.
- 각 실험 ID마다 최소 2개 이상의 evaluation point가 있다.
- vocab 실험 섹션의 primary metric이 `final_val_loss`가 아니라 `final_val_bits_per_char`다.
- overfit/dropout/epoch 섹션에 `best_val_loss`, `final_minus_best_val_loss`, `train_val_gap`이 모두 등장한다.

## 10. 테스트 요구사항

새로 추가하거나 수정할 테스트:

```text
tests/test_hy_llm_manifest.py
tests/test_hy_llm_metrics.py
tests/test_hy_llm_seq_figures.py
tests/test_hy_report_render.py
```

테스트 항목:

- `REPORT.md`에서 추출한 실험 ID가 매니페스트와 일치한다.
- `bits_per_char = loss * tokens_per_char / ln(2)` 계산이 맞다.
- tokenizer가 다른 vocab 실험에서 token loss 순위와 bits/char 순위를 따로 계산한다.
- `best_val_loss`, `best_step`, `final_minus_best_val_loss`가 history에서 재계산된다.
- `compute_proxy_param_tokens`와 `estimated_train_flops`가 누락 없이 생성된다.
- seed 반복 실험은 평균, 표준편차, 개별 seed를 모두 산출한다.
- 필수 sequential graph가 모두 생성되고 nonblank다.
- `REPORT_temp.md` 렌더링 결과가 `REPORT.md`를 수정하지 않는다.

최종 검증 명령:

```bash
python -m pytest tests/test_hy_llm_manifest.py tests/test_hy_llm_metrics.py tests/test_hy_llm_seq_figures.py tests/test_hy_report_render.py -q
python scripts/verify_hy_llm_report_assets.py --report REPORT_temp.md --figures docs/HY/figures_llm_seq
git diff -- REPORT.md
```

마지막 명령의 출력은 비어 있어야 한다.

## 11. 결론 작성 금지 문장

아래 식의 결론은 쓰지 않는다.

- "`vocab_size=2000`이 가장 좋다." 단, token loss만 보고 말한 경우.
- "1500 epoch까지 과적합이 없다." 단, gap/rebound/seed/generation 반복 지표 없이 말한 경우.
- "Pre-LN이 좋다." 단, depth 증가에 따른 안정성 분석 없이 말한 경우.
- "dropout 때문에 과적합이 발생한다." 단, dropout별 train/val/gap/rebound 곡선 없이 말한 경우.
- "activation A가 가장 좋다." 단, seed 평균/분산 없이 말한 경우.
- "final validation loss가 낮으므로 채택한다." 단, best/final/compute/gap 검토 없이 말한 경우.

## 12. 최종 산출물 체크리스트

- [ ] `REPORT.md`는 수정하지 않았다.
- [ ] `docs/HY/llm_relog/experiment_manifest.csv`가 REPORT 실험 전체를 포함한다.
- [ ] 누락 지표가 있는 REPORT 실험은 같은 config/seed/split로 재실행했다.
- [ ] 모든 run에 `config.json`, `tokenizer_profile.json`, `history.jsonl`, `metrics.json`, `samples.jsonl`, `result.md`가 있다.
- [ ] `report_llm_metrics.csv`, `report_llm_metrics_by_step.csv`, `report_llm_summary_by_condition.csv`가 생성됐다.
- [ ] `docs/HY/figures_llm_seq`에 필수 순차 선형 그래프 14개가 있다.
- [ ] 그래프는 final-only가 아니라 step/epoch/tokens_seen 흐름을 보여준다.
- [ ] vocab 결론은 `bits/char` 기준으로 다시 썼다.
- [ ] dropout/epoch 결론은 `best`, `final`, `rebound`, `gap` 기준으로 다시 썼다.
- [ ] activation 결론은 seed 평균/표준편차 기준으로 다시 썼다.
- [ ] Pre-LN 결론은 depth 증가 안정성 질문으로 다시 썼다.
- [ ] `REPORT_temp.md`가 실제 새 로그와 새 그래프를 참조하는 수정본 보고서로 갱신됐다.
- [ ] 테스트와 asset verification을 통과했다.

## 13. 실행자에게 줄 최종 프롬프트

아래 요청을 그대로 수행하라.

```text
REPORT.md를 수정하지 말고, REPORT.md에 들어간 docs/HY/testresult 기준 실험 전체를 대상으로 LLM 개발자 지표를 다시 로깅하라.

1. REPORT.md에서 실험 ID와 비교 그룹을 추출해 docs/HY/llm_relog/experiment_manifest.csv를 만든다.
2. 기존 docs/HY/testresult 로그를 파싱하되, 필요한 LLM 지표나 step별 history가 부족한 실험은 원래 config/seed/split로 다시 실행한다.
3. 각 실험마다 config.json, tokenizer_profile.json, history.jsonl, metrics.json, samples.jsonl, result.md를 만든다.
4. history에는 매 eval step마다 train_loss, val_loss, val_bits_per_char, train_val_gap, best_so_far, rebound, tokens_seen, compute_proxy, throughput을 기록한다.
5. vocab 실험은 token loss가 아니라 bits/char를 primary metric으로 둔다.
6. dropout, epoch, weight_tying, norm/depth 실험은 final loss가 아니라 best/final/rebound/gap의 시간 변화로 해석한다.
7. activation은 seed별 개별 선, 평균 선, std/IQR band로 해석한다.
8. docs/HY/figures_llm_seq에 final-only가 아닌 순차 선형 그래프를 다시 만든다.
9. 새 로그와 새 그래프만 근거로 REPORT_temp.md를 REPORT.md 수정본 초안으로 다시 작성한다.
10. pytest와 asset verification을 통과시키고, git diff -- REPORT.md가 비어 있는지 확인한다.
```
