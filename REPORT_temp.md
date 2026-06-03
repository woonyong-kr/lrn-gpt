# REPORT_temp: LLM 개발 지표로 다시 해석한 실험 보고서

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
- `docs\HY\figures_temp\vocab_loss_vs_bits_per_char.png`
- `docs\HY\figures_temp\parameter_count_vs_bits_per_char.png`
- `docs\HY\figures_temp\compute_proxy_vs_bits_per_char.png`
- `docs\HY\figures_temp\final_minus_best_val_loss.png`
- `docs\HY\figures_temp\activation_seed_bits_per_char.png`

## Vocab Size: 결론이 바뀌는 핵심 실험

| experiment_id | vocab_size | parameter_count | val_tokens_per_char | val_chars_per_token | final_val_loss | final_val_bits_per_char | tokens_per_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E04 | 2000 | 2570112 | 0.6527 | 1.5322 | 4.8114 | 4.5304 | 195011.6770 |
| E00 | 3000 | 2954112 | 0.5838 | 1.7128 | 5.3020 | 4.4658 | 195309.4630 |
| E05 | 4000 | 3338112 | 0.5444 | 1.8369 | 5.6384 | 4.4285 | 195553.0710 |
| E06 | 5000 | 3722112 | 0.5184 | 1.9292 | 5.8825 | 4.3991 | 188826.0500 |

해석:

- `vocab_size`가 커질수록 `val_tokens_per_char`는 내려갑니다. 같은 문장을 더 적은 token으로 표현한다는 뜻입니다.
- `final_val_loss`는 token 하나를 맞히는 난이도라 vocab이 커질수록 불리해질 수 있습니다.
- 그래서 E04의 `final_val_loss=4.8114`만 보고 best라고 하면 tokenizer 효율을 놓칩니다.
- `final_val_bits_per_char`는 E06이 가장 낮습니다. 문자 하나를 설명하는 정보량 기준으로는 vocab 5000이 더 낫습니다.
- 단, E06은 `parameter_count`도 더 크고 처리량도 낮으므로 최종 선택은 품질/비용 trade-off로 써야 합니다.

## Context Length

| experiment_id | context_length | stride | final_val_loss | final_val_bits_per_char | tokens_per_sec | compute_proxy_param_tokens |
| --- | --- | --- | --- | --- | --- | --- |
| E01 | 64 | 64 | 5.1461 | 4.3344 | 102490.3850 | 1.184e+13 |
| E00 | 128 | 128 | 5.3020 | 4.4658 | 195309.4630 | 1.186e+13 |
| E02 | 192 | 192 | 5.3886 | 4.5388 | 288413.6300 | 1.194e+13 |
| E03 | 256 | 256 | 5.4470 | 4.5879 | 340329.1080 | 1.196e+13 |

해석:

- context 실험은 token loss와 bits/char 해석이 크게 충돌하지 않습니다. E01, context 64가 가장 좋습니다.
- 긴 context가 항상 좋은 것이 아니라, 이 데이터와 5 epoch 조건에서는 학습 sample 수 감소와 attention 비용 증가가 더 크게 작용한 것으로 보입니다.
- `tokens_per_sec`는 일반 기대와 다르게 흔들릴 수 있으므로 앞으로는 `warmup_excluded_tokens_per_sec`를 우선 보겠습니다.

## Embedding Dimension

| experiment_id | emb_dim | parameter_count | final_val_loss | final_val_bits_per_char | tokens_per_sec | compute_proxy_param_tokens |
| --- | --- | --- | --- | --- | --- | --- |
| E07 | 128 | 1576192 | 5.3921 | 4.5417 | 198883.6610 | 6.327e+12 |
| E00 | 192 | 2954112 | 5.3020 | 4.4658 | 195309.4630 | 1.186e+13 |
| E08 | 256 | 4725248 | 5.2292 | 4.4044 | 196908.6140 | 1.897e+13 |

해석:

- E08, emb_dim 256은 품질 지표가 가장 좋지만 parameter_count와 compute proxy가 크게 증가합니다.
- 따라서 "큰 embedding이 좋다"가 아니라 "품질은 좋아졌지만 비용 대비 이득을 따로 판단해야 한다"로 써야 합니다.

## Layer Count

| experiment_id | n_layers | parameter_count | final_val_loss | final_val_bits_per_char | tokens_per_sec | compute_proxy_param_tokens |
| --- | --- | --- | --- | --- | --- | --- |
| E11 | 2 | 2065536 | 5.3213 | 4.4820 | 324062.8290 | 8.291e+12 |
| E00 | 4 | 2954112 | 5.3020 | 4.4658 | 195309.4630 | 1.186e+13 |
| E12 | 6 | 3842688 | 5.3121 | 4.4743 | 141418.0980 | 1.542e+13 |
| E13 | 8 | 4731264 | 5.3179 | 4.4792 | 109846.3980 | 1.899e+13 |

해석:

- 5 epoch 조건에서는 4층 baseline이 2/6/8층보다 약간 낫습니다.
- 깊이를 늘리면 표현력은 늘지만 작은 데이터와 짧은 학습에서는 최적화가 어려워질 수 있습니다.
- 층 수 실험은 `n_layers -> loss`만 보지 말고 `parameter_count`, `tokens_per_sec`, `compute_proxy`를 함께 봐야 합니다.

## FFN Multiplier

| experiment_id | ffn_mult | parameter_count | final_val_loss | final_val_bits_per_char | final_generalization_gap | compute_proxy_param_tokens |
| --- | --- | --- | --- | --- | --- | --- |
| E35 | 2 | 2362752 | 5.1104 | 4.3044 | 0.4936 | 1.897e+13 |
| E28 | 4 | 2954112 | 5.1027 | 4.2979 | 0.4830 | 2.372e+13 |
| E36 | 6 | 3545472 | 5.0956 | 4.2920 | 0.4715 | 2.846e+13 |

해석:

- FFN 폭을 키우면 token별 비선형 변환 능력은 늘지만 parameter_count와 과적합 위험도 같이 봐야 합니다.
- 이 실험군은 `ffn_mult` 하나만으로 결론을 내리기보다 depth, norm_first, dropout과의 조합으로 보는 것이 맞습니다.

## Weight Tying

| experiment_id | tie_embeddings | parameter_count | final_train_loss | final_val_loss | final_generalization_gap | final_val_bits_per_char |
| --- | --- | --- | --- | --- | --- | --- |
| E43 | False | 2954112 | 3.1171 | 5.3906 | 2.2735 | 4.5404 |
| E44 | True | 2378112 | 3.6723 | 5.0771 | 1.4049 | 4.2764 |

해석:

- weight tying은 parameter_count를 줄이면서 validation 기준을 개선했습니다.
- 작은 corpus에서는 출력 embedding 공유가 regularization처럼 작동했을 가능성이 있습니다.
- 이 결론은 LLM식 지표로 봐도 유지됩니다.

## Activation Seed 반복

| activation | runs | mean_val_loss | std_val_loss | mean_bits_per_char | std_bits_per_char |
| --- | --- | --- | --- | --- | --- |
| gelu | 3 | 5.0146 | 0.0132 | 4.2237 | 0.0111 |
| relu | 3 | 5.0567 | 0.0053 | 4.2592 | 0.0044 |
| silu | 3 | 5.1267 | 0.0140 | 4.3181 | 0.0118 |

해석:

- activation 실험은 단일 seed보다 평균과 표준편차가 중요합니다.
- 평균이 낮아도 표준편차가 크면 "우연히 한 seed에서 잘 됐다"일 수 있습니다.
- 이 실험군은 LLM 보고서식으로 이미 좋은 방향입니다. `mean ± std`를 본문 결론에 넣으면 더 설득력이 올라갑니다.

## 최근 docs/train 자동화 로그에서 참고할 점

`docs/train/leaderboard.csv`는 HY와 corpus/규모가 달라 직접 수치 비교하면 안 됩니다. 대신 어떤 지표를 계속 남겨야 하는지 보여주는 보조 evidence로 쓸 수 있습니다.

| run_id | vocab_size | context_length | activation_name | ffn_mult | final_val_loss | final_val_bits_per_char | final_generalization_gap | overfit_score | parameter_count | tokens_per_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 111 | 600 | 48 | mish | 3 | 5.5253 | 3.2762 | 0.0160 | 0.0622 | 413184 | 34145.4128 |
| 112 | 600 | 48 | mish | 3 | 5.5256 | 3.2764 | 0.0145 | 0.0576 | 413184 | 15033.8660 |
| 103 | 600 | 48 | mish | 3 | 5.5287 | 3.2782 | 0.0087 | 0.0402 | 413184 | 36336.0572 |
| 101 | 600 | 48 | mish | 3 | 5.5304 | 3.2793 | 0.0161 | 0.1010 | 413184 | 33158.9886 |
| 109 | 600 | 48 | mish | 3 | 5.5332 | 3.2809 | 0.0129 | 0.0494 | 413184 | 35343.6183 |
| 104 | 600 | 48 | mish | 3 | 5.5335 | 3.2811 | 0.0505 | 0.2164 | 413184 | 33647.7215 |
| 110 | 600 | 48 | mish | 3 | 5.5340 | 3.2813 | 0.0108 | 0.0452 | 413184 | 33323.3256 |
| 102 | 600 | 48 | mish | 3 | 5.5345 | 3.2817 | -0.0005 | 0.0117 | 413184 | 36049.1331 |

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
