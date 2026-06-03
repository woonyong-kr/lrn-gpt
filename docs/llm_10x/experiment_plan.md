# LLM 10x Broad Sweep Experiment Plan

## 판단

이번 버전은 확증형 10-seed matrix가 아니라, 조건을 넓게 늘리고 조건당 seed 수를 낮춘 탐색형 matrix다. 목적은 최종 결론을 바로 내는 것이 아니라, loss, 과적합, 시간 비용을 보면서 가능성 낮은 조합을 빠르게 버리고 상위 후보만 10-seed 확증 실험으로 넘기는 것이다.

핵심 원칙:

1. 탐색 단계는 condition당 `3 seeds`만 사용한다.
2. 모든 condition은 같은 seed set을 사용한다.
3. 각 phase에서는 하나의 축만 바꾸고 나머지는 baseline에 고정한다.
4. `n=3` 결과는 screen-ready, 즉 후보 선별용이다.
5. 최종 claim은 상위 후보를 다시 `10 seeds`로 돌린 뒤에만 낸다.

## 고정 데이터

- train: `data/obsidian_llm_10x_lm_train.txt`
- val: `data/obsidian_llm_10x_lm_val.txt`
- chunk 본문 기준: `15,001,711` chars
- 기존 NSMC 대비: `10.0008x`
- document duplicate hash: `0`
- chunk duplicate hash: `0`

## Seed 정책

탐색 seed:

```text
123, 321, 777
```

확증 seed:

```text
123, 321, 777, 2026, 3407, 42, 9001, 2718, 31415, 1618
```

탐색 결과에서 좋은 조건을 고르면 `scripts/build_llm_10x_run_matrix.py --seed-mode confirm`으로 같은 조건을 10-seed matrix로 다시 만들 수 있다.

## 전체 실험 수

| Phase | 축 | physical conditions | repeats | physical runs |
| --- | --- | ---: | ---: | ---: |
| Phase 1 | learning rate | 12 | 3 | 36 |
| Phase 2 | epoch horizon | 1 long-run | 3 | 3 |
| Phase 3 | vocab size | 13 | 3 | 39 |
| Phase 4 | tokenizer min frequency | 10 | 3 | 30 |
| Phase 5 | capacity | 10 | 3 | 30 |
| Phase 6 | context length | 9 | 3 | 27 |
| Phase 7 | batch size | 5 | 3 | 15 |
| Phase 8 | dropout | 11 | 3 | 33 |
| Phase 9 | weight decay | 12 | 3 | 36 |
| Phase 10 | grad clip | 6 | 3 | 18 |
| Phase 11 | FFN multiplier | 6 | 3 | 18 |
| Phase 12 | activation | 11 | 3 | 33 |
| Phase 13 | init std | 6 | 3 | 18 |
| Phase 14 | structure | 6 | 3 | 18 |
| 합계 |  | 118 | 3 | 354 |

Epoch phase는 `1500 epoch` 물리 run 3개에서 `25/50/75/100/150/200/300/400/600/800/1000/1200/1500` milestone을 추출한다. 그래서 aggregate 분석 기준으로는 `130 analysis conditions`, `390 analysis rows`가 된다.

## Baseline

```text
vocab_size=12000
tokenizer_min_frequency=2
context_length=512
batch_size=4
emb_dim=512
n_heads=8
n_layers=8
drop_rate=0.10
ffn_mult=4
activation_name=gelu
attention_impl=sdpa
qkv_bias=False
tie_embeddings=True
init_std=0.02
learning_rate=0.0003
weight_decay=0.05
grad_clip=1.0
epochs=50
```

## Sweep 축

| Phase | 값 |
| --- | --- |
| learning rate | `0.00005, 0.00007, 0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003` |
| epoch milestones | `25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1500` |
| vocab size | `4000, 6000, 8000, 10000, 12000, 16000, 20000, 24000, 32000, 40000, 48000, 56000, 64000` |
| tokenizer min frequency | `1, 2, 3, 4, 5, 8, 13, 21, 34, 55` |
| capacity | 31M급부터 240M급까지 10개 shape |
| context length | `256, 384, 512, 640, 768, 1024, 1280, 1536, 2048` |
| batch size | `1, 2, 4, 8, 16` |
| dropout | `0.00, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40` |
| weight decay | `0.00, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50` |
| grad clip | `0.0, 0.25, 0.5, 1.0, 2.0, 5.0` |
| FFN multiplier | `2, 3, 4, 5, 6, 8` |
| activation | `gelu, gelu_exact, quick_gelu, relu, silu, swish, mish, squared_relu, identity, swiglu, geglu` |
| init std | `0.005, 0.01, 0.015, 0.02, 0.03, 0.04` |
| structure | baseline, QKV bias, untied embeddings, post-LN, manual attention, QKV bias plus untied embeddings |

## 판단 지표

최저 loss만 보지 않는다. 세 그룹을 동시에 본다.

1. tokenizer-fair 품질: `final_val_bits_per_char`, `final_val_nats_per_char`, `best_val_bits_per_char`, `best_val_loss`
2. 과적합/rebound: `overfit_score`, `final_generalization_gap`, `generalization_gap_delta`, `train_val_improvement_gap`, `final_minus_best_val_loss`
3. 노출량/compute: `tokens_seen`, `estimated_chars_seen`, `tokens_per_param`, `compute_proxy`, `estimated_train_flops`
4. 시스템 비용: `elapsed_sec`, `seconds_per_epoch`, `tokens_per_sec_after_warmup`

`vocab_size`와 `tokenizer_min_frequency`는 tokenizer가 달라지는 실험이므로 `final_val_loss` 단독 순위를 사용하지 않는다. 이 두 phase는 반드시 `bits/char`, `tokens_per_char`, `chars_per_token`, tokenizer profile을 같이 본다.

Tokenizer profile 필수 필드:

```text
corpus_sha256
train_sha256
val_sha256
requested_vocab_size
actual_vocab_size
tokenizer_min_frequency
bpe_merge_count
train_tokens_per_char
val_tokens_per_char
train_chars_per_token
val_chars_per_token
token_length_histogram
top_50_merge_rules
special_token_ids
```

좋은 탐색 후보의 조건:

1. 같은 phase baseline보다 paired delta가 낮다.
2. 3 seed 중 최소 2 seed에서 baseline보다 낫다.
3. median val bits/char가 낮다.
4. overfit score가 baseline보다 크게 나빠지지 않는다.
5. `final_minus_best_val_loss`가 크지 않아 final checkpoint 사용이 가능하거나, best checkpoint 사용 전략이 명확하다.
6. 시간/compute 비용 증가가 bits/char 개선폭으로 설명된다.

탈락 조건:

1. train loss만 낮고 val loss가 개선되지 않는다.
2. generalization gap이 크게 벌어진다.
3. `final_minus_best_val_loss`가 커서 장기 학습 후반 rebound가 명확하다.
4. `tokens_per_sec_after_warmup`이 급락하는데 val 개선이 작다.
5. `compute_proxy` 또는 `estimated_train_flops`가 크게 늘지만 bits/char 개선이 작다.
6. epoch milestone에서 best epoch가 훨씬 앞이면 이후 epoch는 과적합 또는 낭비 후보로 본다.

## 결론 규칙

탐색 matrix는 넓고 빠르게 보는 용도다. `n=3`에서 나온 결과는 "후보" 또는 "탈락 후보"라고만 부른다. 최종 모델, 최종 hyperparameter, 논문식 claim은 상위 후보를 10-seed 확증 실험으로 다시 돌린 뒤에만 말한다.

## Epoch / Long-run 해석 규칙

1500 epoch 물리 run은 final 한 점으로 해석하지 않는다. `history.jsonl`의 milestone event에서 아래 값을 추출해 분석한다.

```text
val_bits_per_char
train_bits_per_char
final_generalization_gap
best_val_loss_so_far
final_minus_current_best_val_loss
tokens_seen
estimated_chars_seen
tokens_per_sec_after_warmup
```

해석:

- val bits/char 하락 + gap 안정: 장기 학습 confirmed 후보
- val bits/char 하락 + gap 증가: quality improves with memorization risk
- best 이후 final rebound 큼: early stopping 필요
- final val 상승 + gap 증가: 과적합 또는 학습 낭비

## Generation Audit

loss 개선이 실제 생성 품질로 이어지는지는 별도 sanity check가 필요하다. checkpoint가 있는 후보는 아래 고정 prompt set으로 생성 diversity/repetition을 기록한다.

```text
prompt_set = ["이 영화는", "정말", "스토리는", "배우들의 연기는"]
temperature = 0.8
top_k = 50
max_new_tokens = 100
num_samples_per_prompt = 3
```

필수 지표:

```text
distinct_1
distinct_2
repetition_ratio_3gram
average_generated_length
manual_quality_note
```

`val_bits_per_char`가 좋아져도 `repetition_ratio_3gram`이 커지면 채택을 보류한다. 현재 완료 run 중 checkpoint가 없는 경우 generation 품질 결론은 쓰지 않는다.
