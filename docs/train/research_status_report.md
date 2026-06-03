# mini GPT 실험 연구 상황 보고서

작성 시점: 2026-06-03 KST

참조 자료:

- `docs/HY/testresult/*.md`: 수동 하이퍼파라미터 실험 E01-E20
- `docs/train/state.json`
- `docs/train/leaderboard.csv`
- `docs/train/metrics_summary.csv`
- `docs/train/dashboard.md`
- `docs/train/hypotheses.md`
- `docs/train/runs/run_111.md`
- `docs/train/visuals/loss_overfit_trends.svg`
- `docs/train/visuals/latest_run_metrics.svg`

## 1. 결론 요약

현재 자동 실험 루프의 overfit-aware best는 run 102다. 설정은 `mish`, `context_length=48`, `stride=24`, `batch_size=8`, `emb_dim=128`, `n_heads=4`, `n_layers=2`, `ffn_mult=3`, `drop_rate=0.12`, `tie_embeddings=True`, `attention_impl=sdpa`, `epochs=2.564103`이며, `final_val_loss=5.534507`, `final_generalization_gap=-0.000533`, `overfit_score=0.011694`다.

raw validation loss만 보면 최신 run 111이 가장 낮다. run 111은 같은 413,184 parameter mish 계열에서 `epochs=2.692308`으로 늘린 실험이고, `final_val_loss=5.525291`, `gap=0.015986`, `overfit_score=0.062211`로 일반화 상태를 유지했다. 다만 best 선정 점수는 gap과 overfit_score를 함께 벌점으로 반영하므로 run 102가 여전히 더 안전한 후보로 남아 있다.

향후 실험 계획 입력은 `max_steps`가 아니라 `epochs`를 사용한다. `max_steps`는 과거 호환성과 실제 optimizer update 수 확인을 위한 결과 필드로만 남긴다. 현재 schema와 대기 중인 `next_plan.json`, heartbeat 자동화 지시도 epoch-only 계획 방식으로 수정했다.

## 2. 평가 기준

자동 루프는 단순히 validation loss가 낮은 run을 best로 고르지 않는다. 작은 corpus에서는 train loss가 빠르게 내려가면서 validation이 조금만 좋아져도 과적합일 수 있기 때문이다.

best 선정 점수:

```text
score = final_val_loss
      + 0.25 * overfit_score
      + 0.5 * max(0, final_generalization_gap)
      + 0.5 if fit_status == overfit_risk else 0
```

주요 판단 기준:

- `final_val_loss`: 낮을수록 좋다.
- `final_generalization_gap = final_val_loss - final_train_loss`: 양수로 커질수록 과적합 위험이 커진다.
- `overfit_score`: gap 변화량, train/val 개선 불균형, 최종 gap을 합친 과적합 신호다.
- `fit_status`: `generalizing`, `overfit_risk`, `underfit_or_too_short`, `val_regressed` 등으로 다음 가설의 방향을 정한다.
- `tokens_per_sec`와 `parameter_count`: MPS 환경에서 너무 긴 실험이나 과한 모델 확장을 피하기 위한 운영 지표다.

현재 자동 loop 데이터 기준 epoch 환산:

| 조건 | 값 |
| --- | ---: |
| train token count | 7,470 |
| context_length | 48 |
| stride | 24 |
| batch_size | 8 |
| steps_per_epoch | 39 |
| 90 updates | 2.307692 epochs |
| 100 updates | 2.564103 epochs |
| 105 updates | 2.692308 epochs |

## 3. `docs/HY/testresult` 수동 실험 해석

E01-E20은 모두 `num_epochs=5`로 기록된 별도 수동 실험 묶음이다. 자동 루프와 corpus 길이, tokenizer/vocab, 모델 크기, batch 구성이 다르므로 loss 절대값을 자동 loop 결과와 직접 비교하면 안 된다. 대신 어떤 축이 유망했는지 방향성 근거로 사용한다.

| 실험 | 변경 축 | params | final_val_loss | 관찰 |
| --- | --- | ---: | ---: | --- |
| E01 | context_length 64 | 2,941,824 | 5.146068 | 긴 context보다 짧은 context가 유리했다. |
| E02 | context_length 192 | 2,966,400 | 5.388636 | context를 늘리면 train batch가 줄고 val이 악화됐다. |
| E03 | context_length 256 | 2,978,688 | 5.447002 | 더 긴 context는 더 악화됐다. |
| E04 | vocab_size 2000 | 2,570,112 | 4.811352 | 가장 낮지만 vocab 변경으로 loss scale이 달라 직접 비교 주의. |
| E05 | vocab_size 4000 | 3,338,112 | 5.638393 | vocab 확장은 손실과 perplexity를 키웠다. |
| E06 | vocab_size 5000 | 3,722,112 | 5.882533 | 더 큰 vocab은 더 불리했다. |
| E07 | emb_dim 128 | 1,576,192 | 5.392111 | 작은 embedding은 가볍지만 val은 높았다. |
| E08 | emb_dim 256 | 4,725,248 | 5.229174 | 큰 embedding은 val 개선이 있으나 params가 크게 증가했다. |
| E09 | n_heads 3 | 2,954,112 | 5.288454 | head 수 변화는 큰 차이를 만들지 않았다. |
| E10 | n_heads 6 | 2,954,112 | 5.302649 | 3 heads보다 약간 나빴다. |
| E11 | n_layers 2 | 2,065,536 | 5.321256 | 깊이를 줄여도 큰 붕괴는 없었다. |
| E12 | n_layers 6 | 3,842,688 | 5.312068 | 깊이를 늘려도 이득은 작았다. |
| E13 | n_layers 8 | 4,731,264 | 5.317900 | 더 깊게 가도 개선은 제한적이었다. |
| E14 | ffn_multiplier 2 | 2,362,752 | 5.291719 | FFN 축소가 크게 불리하지 않았다. |
| E15 | ffn_multiplier 6 | 3,545,472 | 5.300615 | FFN 확대 이득은 작았다. |
| E16 | drop_rate 0.0 | 2,954,112 | 5.252427 | 이 수동 세팅에서는 dropout 제거가 좋았다. |
| E17 | drop_rate 0.05 | 2,954,112 | 5.277728 | 낮은 dropout은 0.2보다 좋았다. |
| E18 | drop_rate 0.2 | 2,954,112 | 5.340294 | 강한 dropout은 val을 악화했다. |
| E19 | qkv_bias True | 2,956,416 | 5.295729 | qkv bias 효과는 작았다. |
| E20 | weight_tying True | 2,378,112 | 5.427288 | 이 큰 수동 세팅에서는 tying이 악화됐다. |

수동 실험에서 얻은 방향성은 세 가지다. 첫째, 이 데이터에서는 긴 context보다 짧은 context가 안전하다. 둘째, capacity를 무작정 키워도 validation 개선이 크지 않고 비용이 늘어난다. 셋째, 강한 dropout은 오히려 학습을 방해할 수 있으므로 regularization은 작게 조절해야 한다. 자동 루프는 이 방향을 받아 더 작은 MPS 친화 모델에서 `context_length=48`, `stride` 조절, tied embedding, FFN 축소, activation, 짧은 epoch horizon을 중심으로 탐색했다.

## 4. 자동 루프의 가설 생성 기준

가설은 매 run마다 다음 기준으로 만들어졌다.

1. 한 번에 한 축만 바꾼다. seed 반복은 재현성 검증으로 보고, 구조 변경과 섞지 않는다.
2. `overfit_risk`면 dropout, weight_decay, tying, capacity 축소, stride/context, epoch 단축처럼 과적합을 줄이는 축을 고른다.
3. `underfit_or_too_short`면 epoch 또는 capacity를 늘린다.
4. `generalizing`이면 먼저 seed 반복으로 우연성을 확인하고, 그 다음 activation이나 stride처럼 작고 해석 가능한 축을 바꾼다.
5. MPS에서 1-2초 내외로 끝나는 작은 실험을 우선한다. 현재 유망 계열은 약 413k params, 30k-36k tokens/sec 수준이다.
6. raw val loss가 좋아도 gap과 overfit_score가 커지면 기본 후보가 아니라 seed-specific 또는 rescue 후보로 분류한다.

## 5. 가설 검증과 변경 흐름

초기 기준선 단계, runs 001-006:

- run 001은 매우 짧은 smoke 성격이라 `underfit_or_too_short`였다.
- run 002에서 MPS용 중간 기준선으로 validation은 크게 내려갔지만 gap과 overfit_score가 커져 `overfit_risk`가 됐다.
- dropout 0.20, weight_decay 0.05는 효과가 작았다.
- `tie_embeddings=True`는 gap을 낮췄지만 overfit_score는 충분히 낮추지 못했다.
- `ffn_mult=3` 축소는 parameter는 줄였지만 validation/gap이 악화되어 단독 해결책으로 채택되지 않았다.

seed와 activation 검증 단계, runs 007-020:

- seed 151 반복에서 tie embedding 계열이 `generalizing`으로 바뀌었다.
- `quick_gelu`, `silu`, `gelu_exact`는 seed 151에서 모두 안정적이었고, `quick_gelu`가 근소하게 앞섰다.
- seed 134와 202에서는 같은 activation이 다시 overfit 성향을 보였으므로 activation 효과는 seed 민감성이 있다고 판단했다.
- FFN dropout 위치 변경과 dropout 제거는 seed 151에서는 개선됐지만 seed 134에서는 과적합을 완전히 풀지 못했다.
- 이때부터 가설은 “activation 자체”보다 “seed variance와 학습 window 구성” 쪽으로 이동했다.

중반 안정화 단계:

- 수동 실험의 “짧은 context가 유리하다”는 방향을 반영해 `context_length=48`, `stride=24`, `batch_size=8` 계열로 정착했다.
- capacity는 큰 모델보다 `emb_dim=128`, `n_layers=2`, `ffn_mult=3`의 413,184 parameter 모델이 효율과 안정성 균형이 좋았다.
- 구현은 `attention_impl=sdpa`가 속도 면에서 유리했고, MPS에서 30k tokens/sec 전후를 유지했다.
- activation은 여러 후보 비교 뒤 `mish`가 현재 low-risk band에서 가장 좋은 축으로 남았다.

stride/seed variance 단계:

- `stride=24`는 low-risk seed에서 가장 좋은 validation band를 만들었다.
- 다만 seed 707 같은 fresh seed에서는 run 104처럼 raw val은 좋지만 gap이 커지는 실패가 나왔다.
- `stride=20`은 run 106에서 seed 707의 gap을 크게 낮췄지만 validation 손실을 조금 희생했다.
- 따라서 stride20은 global default가 아니라 high-gap seed rescue 정책으로 분류했다.

epoch horizon 단계, runs 101-111:

- `epochs=2.564103`은 기존 100 update horizon에 해당한다.
- run 102는 seed 151에서 overfit-aware best가 됐다.
- run 103은 seed 202에서 더 낮은 raw val을 만들며 2.564103 epoch horizon의 전이 가능성을 확인했다.
- run 104는 fresh seed 707에서 과적합을 드러내 seed rescue 정책 필요성을 다시 확인했다.
- run 108은 fresh seed 808이 low-risk로 통과해 seed 707이 전체 정책을 무너뜨리는 사례는 아니라고 판단했다.
- runs 109-111은 `epochs=2.692308` 계열, 즉 105 update 상당의 작은 horizon 증가를 검증했다. seed 808, 151, 202에서 모두 low-risk generalizing을 유지했고 raw val은 개선됐다.
- 하지만 run 111은 `2.692308 * 39`의 반올림 경계 때문에 106 effective updates가 실행됐다. 이를 막기 위해 epoch-to-update 환산에 epsilon 보정을 넣었고, 현재 대기 plan은 정확한 `105 / 39 = 2.6923076923076925 epochs`로 재검증하도록 되어 있다.

## 6. 현재 상태

| 항목 | 값 |
| --- | --- |
| 완료 run 수 | 111 |
| 최신 run | 111 |
| 최신 report | `docs/train/runs/run_111.md` |
| dashboard | `docs/train/dashboard.md` |
| trend visual | `docs/train/visuals/loss_overfit_trends.svg` |
| latest visual | `docs/train/visuals/latest_run_metrics.svg` |
| 현재 best | run 102 |
| 최신 raw-val best | run 111 |
| 활성 lock | 없음 |
| 다음 run 번호 | 112 |

주요 후보 비교:

| run | 의미 | epochs | effective updates | final_val_loss | gap | overfit_score | 판단 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 102 | 현재 overfit-aware best | 2.564103 | 100 | 5.534507 | -0.000533 | 0.011694 | 가장 안전한 기본 후보 |
| 103 | seed202 2.564103 epoch 검증 | 2.564103 | 100 | 5.528694 | 0.008664 | 0.040245 | raw val 강함, low-risk |
| 108 | fresh seed808 검증 | 2.564103 | 100 | 5.536325 | 0.003856 | 0.022365 | seed707이 예외일 가능성 지지 |
| 109 | seed808 2.692308 epoch | 2.692308 | 105 | 5.533208 | 0.012854 | 0.049360 | raw val 개선, low-risk |
| 110 | seed151 2.692308 epoch | 2.692308 | 105 | 5.533954 | 0.010798 | 0.045152 | best seed에서도 개선 |
| 111 | seed202 2.692308 epoch | 2.692308 | 106 | 5.525291 | 0.015986 | 0.062211 | raw val 최고, rounding artifact 포함 |

## 7. 현재 연구 판단

현재 연구의 가장 강한 결론은 “mish + stride24 + 413k params + 약 2.56-2.69 epochs” 계열이 가장 유망하다는 것이다. 이 계열은 수동 실험에서 보인 짧은 context 선호, 과한 capacity 확장 비효율, 강한 dropout 비효율이라는 방향성과도 맞다.

아직 확정하지 않은 쟁점은 epoch horizon이다. 2.692308 epochs는 raw validation을 개선했지만 gap과 overfit_score도 조금 올라간다. overfit-aware score 기준으로는 2.564103 epochs의 run 102가 더 안전하다. 따라서 다음 검증은 새 축을 추가하는 것이 아니라, 정확한 105-update 상당 epoch 값으로 run 111의 rounding artifact를 제거하고 같은 seed202 결과를 다시 보는 것이 가장 정보량이 높다.

## 8. 앞으로의 실험 정책

앞으로 `docs/train/next_plan.json`의 `config_overrides`에는 `epochs`를 사용한다. `max_steps`는 넣지 않는다.

다음 연구 방향:

1. 정확한 `epochs=2.6923076923076925`로 seed202 재검증을 수행한다.
2. 결과가 low-risk면 2.69 epoch horizon을 low-risk seed 후보로 승격하고 fresh seed를 하나 더 본다.
3. gap 또는 overfit_score가 상승하면 기본값은 run 102의 2.564103 epochs로 유지한다.
4. fresh seed에서 high-gap failure가 나오면 activation이나 capacity를 바꾸기 전에 stride20 rescue를 먼저 적용한다.
5. 새 capacity 확장, 큰 context, 강한 dropout은 현재 증거상 우선순위가 낮다.

## 9. 이번 정리에서 반영한 운영 변경

- `src/train_loop_agent.py`: 새 plan schema가 `max_steps`를 허용하지 않도록 변경했다.
- `src/train_loop_agent.py`: rule-based hardware baseline도 새 실험에서는 `epochs`를 명시하도록 변경했다.
- `docs/train/next_plan.schema.json`: `config_overrides.allowed_keys`에서 `max_steps`를 제거했다.
- `docs/train/next_plan.json`: 대기 중인 run 112 계획에서 `max_steps` 입력을 제거하고 `epochs`만 남겼다.
- heartbeat 자동화: 다음 실험 계획 작성 시 `epochs`만 authored training-length knob으로 사용하도록 업데이트했다.

정리하면, 현재는 “raw loss를 더 낮추는 2.69 epoch 후보”와 “overfit-aware score가 가장 좋은 2.56 epoch 후보” 사이의 마지막 검증 단계다. 지금은 새 구조를 더 탐색하기보다 epoch horizon을 정확히 정리하고 seed variance 정책을 확정하는 것이 가장 좋은 다음 연구 행동이다.
