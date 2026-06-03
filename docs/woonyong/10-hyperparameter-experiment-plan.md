# GPT 하이퍼파라미터 실험 설계

## 1. 현재 구현 확인

현재 구현은 교육용 mini GPT의 필수 경로를 모두 갖추고 있다.

| 영역 | 현재 구현 |
| --- | --- |
| Tokenizer | UTF-8 byte-level BPE, special token `<pad>/<unk>/<bos>/<eos>`, vocab save/load |
| Dataset | 다음 토큰 예측용 `GPTDataset`, `context_length`, `stride`, `batch_size`, shuffle DataLoader |
| Embedding | token embedding + absolute position embedding + dropout |
| Attention | Q/K/V projection, multi-head split/merge, causal mask, attention dropout, residual dropout |
| GPT block | attention residual, FFN residual, `norm_first`로 Pre-LN/Post-LN 선택 가능 |
| FFN | `Linear -> activation -> Linear`, `ffn_mult`로 hidden 확장 배율 설정 |
| Model | N개 TransformerBlock, final LayerNorm, LM head, `init_std` 초기화 |
| Train | cross entropy loss, eval loss, checkpoint save/load, generation temperature/top-k |
| Fine-tune | GPT backbone 위 sequence classification head, NSMC sentiment train/eval |

## 2. 이번에 추가한 실험 선택지

구조 전체를 바꾸지 않고, 함수 교체나 순서 교체 수준으로만 열었다.

| 옵션 | 위치 | 비교 의미 |
| --- | --- | --- |
| `activation_name` | FFN 내부 activation/gate | `gelu`, `gelu_exact`, `quick_gelu`, `relu`, `silu`, `mish`, `squared_relu`, `identity`, `swiglu`, `geglu` 비교 |
| `ffn_dropout_position` | FFN 내부 dropout 순서 | `after_output`, `after_activation`, `none` 비교 |
| `attention_impl` | attention score/value 계산 함수 | 기존 manual 구현과 PyTorch `scaled_dot_product_attention` 비교 |
| `norm_eps` | LayerNorm | 수치 안정성 민감도 비교 |
| `tie_embeddings` | token embedding / LM head | weight tying이 파라미터 수와 validation loss에 주는 영향 비교 |
| `seed` | model init, BPE/DataLoader shuffle | 같은 조건 재현성과 seed variance 확인 |

기존 옵션인 `emb_dim`, `n_heads`, `n_layers`, `drop_rate`, `qkv_bias`, `ffn_mult`, `norm_first`, `init_std`도 모두 실험 계획에 포함했다.

주의할 점은 `swiglu/geglu`이다. 이 둘은 바깥에서 보면 `FFN(input) -> same shape output`을 유지하지만, 내부에서는 `linear1`이 `value/gate` 두 갈래를 만들고 `value * activation(gate)`를 계산한다. 그래서 순수 activation 교체라기보다 "구조 전체는 유지한 gated FFN 교체"로 해석해야 한다. 결과를 볼 때는 반드시 `parameter_count`와 과적합 gap을 함께 본다.

## 3. 미흡한 부분과 판단

| 미흡점 | 영향 | 이번 대응 |
| --- | --- | --- |
| seed가 모델 debug 초기화 중심이라 DataLoader shuffle 재현성이 약했음 | 같은 하이퍼파라미터라도 batch 순서가 달라져 loss 비교가 흔들림 | `create_dataloader(..., seed=...)` 추가 |
| activation이 GELU 하나로 고정되어 있었음 | LLM에서 자주 비교하는 FFN 비선형성 실험 불가 | activation factory 추가 |
| dropout 위치가 FFN output 뒤로 고정되어 있었음 | noise가 hidden 확장부에 들어갈 때와 residual 직전에 들어갈 때 비교 불가 | `ffn_dropout_position` 추가 |
| attention 구현이 manual 하나였음 | 구현 정확도/속도 비교 축 부족 | `attention_impl="sdpa"` 추가 |
| eval이 먼저 실행되면 inference-mode causal mask가 학습 backward를 막을 수 있었음 | train 중 eval 이후 다음 step에서 RuntimeError 가능 | mask 반환 시 normal tensor clone 처리 및 회귀 테스트 추가 |
| `train_model`은 epoch train loss 중심이라 150회 실험 기록에는 부족 | run별 val loss, 속도, 파라미터 수 비교가 어려움 | 별도 `src/experiments.py` 실행기 추가 |
| BPE를 전체 corpus에서 학습하면 validation 분포가 merge rule에 섞일 수 있음 | validation loss가 실제보다 조금 덜 엄격해질 수 있음 | raw text를 먼저 train/validation으로 나눈 뒤 train text로만 BPE 학습 |
| 20 step smoke run만으로는 과적합 판단이 약함 | 용량이 큰 모델이 아직 외우기 전이라 gap이 작게 보일 수 있음 | 150회 탐색 후 상위 후보는 더 긴 step과 seed 반복으로 재검증 |
| NSMC data가 없는 새 환경에서는 자동 실험이 영어 샘플로 fallback될 수 있음 | 한국어 중심 결과를 보려면 NSMC 파일 존재 여부 확인 필요 | 기본 smoke corpus를 `data/nsmc_lm_train.txt`로 두고, 없으면 `src/learning/the-verdict.txt`로 fallback |

## 4. 150회 실험 서순

150회를 전부 random으로 섞으면 해석이 어렵다. 그래서 다음 순서를 사용한다.

1. `run 001`: baseline
2. `run 002~046`: 한 번에 한 축만 바꾸는 OFAT 실험
3. `run 047~050`: baseline seed 반복
4. `run 051~150`: 여러 축을 같이 바꾸는 random coverage 실험

이 순서가 좋은 이유는 먼저 "어떤 단일 축이 loss와 속도를 움직이는지"를 이해하고, 그 다음 "축들이 같이 바뀌었을 때 가설이 유지되는지"를 볼 수 있기 때문이다.

## 5. 하이퍼파라미터별 가설

| 축 | 가설 |
| --- | --- |
| `vocab_size` | 커지면 sequence가 짧아질 수 있지만 LM head가 커져 작은 데이터에서는 과적합과 속도 저하가 생길 수 있다. |
| `context_length` | 길수록 장거리 문맥을 보지만 attention 비용이 늘고 학습 sample 수가 줄 수 있다. |
| `stride` | 작게 잡으면 겹치는 sample이 늘어 loss는 안정될 수 있지만 데이터 중복이 커진다. |
| `batch_size` | 클수록 gradient noise가 줄지만 같은 step 수에서는 업데이트 다양성이 줄 수 있다. |
| `learning_rate` | 클수록 초반 loss는 빨리 내려가지만 불안정/발산 위험이 커진다. |
| `weight_decay` | 과적합을 줄일 수 있지만 작은 모델/짧은 학습에서는 underfit이 될 수 있다. |
| `emb_dim` | 커질수록 표현력과 파라미터 수가 늘어 train loss는 내려가고 val gap은 커질 수 있다. |
| `n_heads` | head 수가 늘면 관점이 나뉘지만 head_dim이 작아져 한 head의 표현력은 줄 수 있다. |
| `n_layers` | 깊어질수록 표현력은 늘지만 작은 데이터와 짧은 step에서는 최적화가 어려울 수 있다. |
| `drop_rate` | regularization을 주지만 smoke 규모에서는 학습 속도를 늦출 수 있다. |
| `qkv_bias` | attention projection 자유도를 늘리지만 효과는 작을 수 있다. |
| `ffn_mult` | FFN 용량이 늘면 token별 변환 능력이 커지지만 계산량과 과적합 위험도 커진다. |
| `norm_first` | Pre-LN은 깊은 모델에서 안정적일 가능성이 높고, Post-LN은 얕은 기준 구현 비교에 좋다. |
| `norm_eps` | 수치 안정성 축이며 너무 크면 정규화 효과가 둔해질 수 있다. |
| `activation_name` | GELU 계열은 LLM 기준점, SiLU/Mish는 부드러운 gradient, ReLU 계열은 단순하지만 dead region 위험이 있다. `swiglu/geglu`는 외부 shape는 유지하지만 FFN 내부 gate/value 계산을 추가하므로 순수 activation보다 용량과 inductive bias가 함께 바뀐다. |
| `ffn_dropout_position` | activation 뒤 dropout은 hidden feature를 흔들고, output 뒤 dropout은 residual 직전 출력을 흔든다. |
| `attention_impl` | manual과 SDPA는 의미상 같은 attention이므로 loss보다 속도와 수치 차이를 확인한다. |
| `tie_embeddings` | 파라미터를 줄이고 작은 데이터에서 regularization처럼 작동할 수 있다. |
| `init_std` | 너무 크면 logit scale이 커져 불안정하고, 너무 작으면 학습 신호가 약할 수 있다. |
| `grad_clip` | 불안정한 설정의 폭주를 막지만 너무 낮으면 학습 속도를 제한한다. |

## 6. 실행 방법

계획만 확인:

```bash
python -m src.experiments --total-runs 150 --plan-only --output-dir experiments/lm_hparam_plan
```

150회 실행:

```bash
python -m src.experiments --total-runs 150 --device auto --output-dir experiments/lm_hparam
```

빠른 검증:

```bash
python -m src.experiments --total-runs 2 --max-steps 2 --device cpu --output-dir experiments/lm_hparam_smoke
```

결과 파일:

| 파일 | 내용 |
| --- | --- |
| `plan.csv` | 실행 전 150회 설정표 |
| `plan.json` | 같은 계획의 JSON 버전 |
| `results.csv` | run별 loss, 속도, 파라미터 수 |
| `results.jsonl` | 중간에 끊겨도 누적되는 line-by-line 결과 |

엄밀한 검증용으로는 smoke 기본값보다 긴 step을 준다.

```bash
python -m src.experiments --total-runs 150 --max-steps 100 --device auto --output-dir experiments/lm_hparam_100step
```

## 7. 결과 해석 기준

우선순위는 `final_val_loss`가 낮으면서 `final_generalization_gap`과 `overfit_score`가 낮은 설정이다. validation loss가 조금 낮아도 train/val gap이 크게 벌어졌다면 "좋은 모델"이 아니라 "더 잘 외운 모델"일 수 있다. validation loss가 비슷하면 더 작고 빠른 모델이 더 좋은 선택이다.

같은 seed의 단일 축 비교에서 차이가 작다면, `run 047~050`의 seed 반복 폭과 비교한다. 차이가 seed 반복의 흔들림보다 작으면 그 하이퍼파라미터 효과는 약하다고 해석하는 것이 안전하다.

실험 결과에는 다음 과적합 지표를 함께 남긴다.

| 지표 | 의미 |
| --- | --- |
| `initial_generalization_gap` | 학습 전 `initial_val_loss - initial_train_loss` |
| `final_generalization_gap` | 학습 후 `final_val_loss - final_train_loss` |
| `generalization_gap_delta` | 학습하면서 train/val gap이 얼마나 커졌는지 |
| `train_val_improvement_gap` | train loss 개선량이 validation loss 개선량보다 얼마나 큰지 |
| `overfit_score` | gap과 개선량 차이를 합친 과적합 위험 점수. 낮을수록 좋다. |
| `fit_status` | `generalizing`, `overfit_risk`, `underfit_or_too_short`, `val_regressed`, `mixed` 중 하나 |

과적합 판단은 다음 순서로 한다.

1. `final_val_loss`가 baseline보다 낮은지 본다.
2. `final_generalization_gap = final_val_loss - final_train_loss`가 baseline보다 커졌는지 본다.
3. `generalization_gap_delta`가 양수이고 큰지 본다.
4. `train_val_improvement_gap`이 크면 train만 좋아지고 validation은 따라오지 못한 것이다.
5. `fit_status == "overfit_risk"`인 설정은 후보에서 바로 제외하지는 않되, 같은 validation loss라면 더 작은 gap의 설정을 우선한다.

150회 탐색이 끝난 뒤에는 `final_val_loss` 기준 상위권 중 `overfit_score`가 낮은 후보를 골라 같은 설정을 seed 3개 이상으로 다시 돌린다. 한 번의 seed에서만 좋은 설정은 하이퍼파라미터 효과가 아니라 초기화 운일 수 있다.
