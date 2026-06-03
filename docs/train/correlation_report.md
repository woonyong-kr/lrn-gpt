# 상관 증거 보고서

이 문서는 BPE/tokenizer 상태, dataset 상태, model/training 조건이 loss와 과적합에 어떤 상관을 갖는지 보기 위한 관찰 보고서입니다.

중요한 경계:

- 아래 상관계수는 현재 로그에서 나온 관찰 상관입니다. 인과 증명으로 바로 쓰지 않습니다.
- 자동 loop와 `docs/HY/testresult`는 dataset, 하드웨어, 실험 스케일이 다르므로 섞어서 계산하지 않습니다.
- tokenizer 또는 dataset이 달라지는 비교는 token-level loss보다 `final_val_nats_per_char`, `final_val_bits_per_char`, gap/overfit을 우선합니다.
- loss 그래프와 overfit 그래프가 없는 결과는 상관 주장 근거로 쓰지 않습니다.

![상관 증거 그래프](visuals/correlation_evidence.svg)

## 범위별 자료 구성

| 범위 | 행 수 | dataset 그룹 수 | vocab 값 수 | tokenizer scale 수 | 과적합 지표 |
| --- | ---: | ---: | ---: | ---: | --- |
| auto_loop | 112 | 1 | 1 | 1 | computed |
| hy_testresult | 20 | 1 | 4 | 4 | gap_proxy |

## 가장 강한 관찰 상관

| 범위 | 변수군 | 관계 | n | 피어슨 r | 스피어만 rho | 해석 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| hy_testresult | tokenizer | val_chars_per_token -> final_val_loss | 20 | 0.9419 | 0.6217 | 강한 관찰 상관: 통제 반복 우선 후보 |
| hy_testresult | dataset_tokenized | train_token_count -> final_val_loss | 20 | -0.9397 | -0.6217 | 강한 관찰 상관: 통제 반복 우선 후보 |
| hy_testresult | dataset_tokenized | val_token_count -> final_val_loss | 20 | -0.9394 | -0.6217 | 강한 관찰 상관: 통제 반복 우선 후보 |
| hy_testresult | tokenizer | val_tokens_per_char -> final_val_loss | 20 | -0.9394 | -0.6217 | 강한 관찰 상관: 통제 반복 우선 후보 |
| hy_testresult | tokenizer | actual_vocab_size -> final_val_loss | 20 | 0.9178 | 0.6217 | 강한 관찰 상관: 통제 반복 우선 후보 |
| hy_testresult | tokenizer | vocab_size -> final_val_loss | 20 | 0.9178 | 0.6217 | 강한 관찰 상관: 통제 반복 우선 후보 |
| hy_testresult | windowing | context_length -> tokens_per_sec | 20 | 0.7541 | 0.5999 | 강한 관찰 상관: 통제 반복 우선 후보 |
| hy_testresult | windowing | stride -> tokens_per_sec | 20 | 0.7541 | 0.5999 | 강한 관찰 상관: 통제 반복 우선 후보 |
| hy_testresult | windowing | context_length -> final_val_bits_per_char | 20 | 0.6778 | 0.5781 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | windowing | context_length -> final_val_nats_per_char | 20 | 0.6778 | 0.5781 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | windowing | stride -> final_val_bits_per_char | 20 | 0.6778 | 0.5781 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | windowing | stride -> final_val_nats_per_char | 20 | 0.6778 | 0.5781 | 중간 관찰 상관: confound 확인 필요 |
| auto_loop | model_capacity | emb_dim -> final_val_bits_per_char | 112 | -0.6208 | -0.1629 | 중간 관찰 상관: confound 확인 필요 |
| auto_loop | model_capacity | emb_dim -> final_val_loss | 112 | -0.6208 | -0.1629 | 중간 관찰 상관: confound 확인 필요 |
| auto_loop | model_capacity | emb_dim -> final_val_nats_per_char | 112 | -0.6208 | -0.1629 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | model_capacity | n_layers -> tokens_per_sec | 20 | -0.5994 | -0.5539 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | tokenizer | actual_vocab_size -> final_generalization_gap | 20 | 0.5832 | 0.5346 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | tokenizer | actual_vocab_size -> overfit_score | 20 | 0.5832 | 0.5346 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | tokenizer | vocab_size -> final_generalization_gap | 20 | 0.5832 | 0.5346 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | tokenizer | vocab_size -> overfit_score | 20 | 0.5832 | 0.5346 | 중간 관찰 상관: confound 확인 필요 |
| auto_loop | windowing | context_length -> final_val_bits_per_char | 112 | 0.5759 | 0.5938 | 중간 관찰 상관: confound 확인 필요 |
| auto_loop | windowing | context_length -> final_val_loss | 112 | 0.5759 | 0.5938 | 중간 관찰 상관: confound 확인 필요 |
| auto_loop | windowing | context_length -> final_val_nats_per_char | 112 | 0.5759 | 0.5938 | 중간 관찰 상관: confound 확인 필요 |
| auto_loop | windowing | context_length -> overfit_score | 112 | 0.5634 | 0.5194 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | tokenizer | val_chars_per_token -> final_generalization_gap | 20 | 0.5534 | 0.5346 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | tokenizer | val_chars_per_token -> overfit_score | 20 | 0.5534 | 0.5346 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | hardware_runtime | tokens_per_sec -> final_val_bits_per_char | 20 | 0.5351 | 0.3188 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | hardware_runtime | tokens_per_sec -> final_val_nats_per_char | 20 | 0.5351 | 0.3188 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | dataset_tokenized | train_token_count -> final_generalization_gap | 20 | -0.5294 | -0.5346 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | dataset_tokenized | train_token_count -> overfit_score | 20 | -0.5294 | -0.5346 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | dataset_tokenized | val_token_count -> final_generalization_gap | 20 | -0.5288 | -0.5346 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | dataset_tokenized | val_token_count -> overfit_score | 20 | -0.5288 | -0.5346 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | tokenizer | val_tokens_per_char -> final_generalization_gap | 20 | -0.5288 | -0.5346 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | tokenizer | val_tokens_per_char -> overfit_score | 20 | -0.5288 | -0.5346 | 중간 관찰 상관: confound 확인 필요 |
| auto_loop | windowing | context_length -> final_generalization_gap | 112 | 0.4879 | 0.4683 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | windowing | context_length -> final_generalization_gap | 20 | -0.4594 | -0.4862 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | windowing | context_length -> overfit_score | 20 | -0.4594 | -0.4862 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | windowing | stride -> final_generalization_gap | 20 | -0.4594 | -0.4862 | 중간 관찰 상관: confound 확인 필요 |
| hy_testresult | windowing | stride -> overfit_score | 20 | -0.4594 | -0.4862 | 중간 관찰 상관: confound 확인 필요 |
| auto_loop | windowing | stride -> final_val_bits_per_char | 58 | 0.4512 | -0.0373 | 중간 관찰 상관: confound 확인 필요 |
| auto_loop | windowing | stride -> final_val_loss | 58 | 0.4512 | -0.0373 | 중간 관찰 상관: confound 확인 필요 |
| auto_loop | windowing | stride -> final_val_nats_per_char | 58 | 0.4512 | -0.0373 | 중간 관찰 상관: confound 확인 필요 |

## 어떻게 상관을 증명에 가깝게 만들 것인가

| 주장 | 통제 설계 | 성공 근거 |
| --- | --- | --- |
| BPE 상태가 학습 내용을 바꾼다 | dataset, model, optimizer, epochs, seed set을 고정하고 vocab/BPE만 변경한다 | `val_tokens_per_char` 변화와 `final_val_nats_per_char`, 생성 샘플, gap/overfit 변화가 같은 방향으로 반복된다 |
| dataset 상태가 학습 내용을 바꾼다 | tokenizer를 고정한 뒤 dataset만 바꾼다 | tokenization scale이 고정된 상태에서 loss/overfit/generation 변화가 dataset별로 반복된다 |
| dataset과 tokenizer가 상호작용한다 | 2x2 factorial: dataset A/B x tokenizer A/B | tokenizer가 자신의 dataset에서만 유리하거나, cross-tokenizer에서 gap이 증가하는 interaction이 반복된다 |
| model capacity와 tokenizer 효과가 섞인다 | vocab sweep 중 parameter_count를 기록하고 capacity sweep은 별도 phase로 분리한다 | vocab 효과가 `parameter_count` 효과와 분리되어 `nats_per_char`에서 남는다 |

## 다음 실험 설계 규칙

1. correlation claim 하나당 독립변수 하나만 움직인다.
2. 최소 3 seed 반복 후 평균과 분산을 같이 본다.
3. 모든 결과는 train/val loss curve와 gap/overfit curve를 포함한다.
4. vocab/BPE 비교는 token loss가 아니라 char-normalized loss를 primary metric으로 둔다.
5. dataset 변경 실험은 corpus hash, train/val char count, train/val token count를 필수 기록한다.
