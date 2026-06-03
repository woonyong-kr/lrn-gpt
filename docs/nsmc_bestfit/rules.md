# NSMC Best-Fit 규칙

이 흐름은 사람이 매번 다음 실험을 고르는 대신, 완료된 결과를 보고 다음 phase를 추가한다.

## 기본 데이터

- LM train: `/Users/woonyong/workspace/Krafton-Jungle/SW_AI-W13-gpt/data/nsmc_lm_train.txt`
- LM val: `/Users/woonyong/workspace/Krafton-Jungle/SW_AI-W13-gpt/data/nsmc_lm_val.txt`
- tokenizer/cache/run 산출물은 `local/nsmc_bestfit`에 둔다.
- 보고서와 matrix는 `docs/nsmc_bestfit`에 둔다.
- 누적 선형 그래프는 `docs/nsmc_bestfit/linear_graphs/figure_index.md`에 둔다.

## 선택 기준

1. tokenizer가 다른 실험은 token loss만 비교하지 않고 `best_val_bits_per_char`를 우선한다.
2. 같은 수준이면 `final_minus_best_val_loss`가 작은 쪽을 고른다. 후반 rebound가 작다는 뜻이다.
3. 그 다음은 `final_generalization_gap`이 작은 쪽을 고른다. train만 좋아지고 val이 벌어지는 후보를 낮춘다.
4. 그 다음은 `compute_proxy = parameter_count * tokens_seen`이 작은 쪽을 고른다.
5. 마지막으로 `tokens_per_sec_after_warmup`이 높은 쪽에 작은 보너스를 준다.

## phase 순서

1. `phase1_tokenizer`: vocab_size와 min_frequency를 먼저 고른다.
2. `phase2_learning_rate`: tokenizer champion을 고정하고 learning rate를 고른다.
3. `phase3_capacity`: optimizer/tokenizer를 고정하고 모델 폭과 깊이를 고른다.
4. `phase4_regularization`: dropout과 weight decay로 과적합을 제어한다.
5. `phase5_confirm`: 최종 후보를 여러 seed로 반복해서 우연인지 확인한다.

## 한글 tokenizer에 대한 규칙

현재 실행 루프는 GPT 계열과 같은 byte-level BPE를 쓴다. 따라서 `영화`와 ` 영화`는 둘 다 유효한 후보가 될 수 있다.
공백 포함 토큰은 언어모델 효율에는 유리할 수 있으므로 금지하지 않는다.
대신 `min_frequency`와 `vocab_size`를 같이 보고, 문장 전체가 거대한 토큰으로 커지는 후보는 bits/char와 일반화 gap에서 걸러낸다.

## 누적 그래프 규칙

- 모든 완료 실행은 `all_run_results.jsonl`에 원장으로 남긴다.
- 그래프는 원장을 `run_number` 순서로 읽어 누적 선형 그래프로 다시 그린다.
- 핵심 그래프는 quality, tokenizer efficiency, compute, generalization, training history, hyperparameter path를 나눠 본다.
- 가장 큰 전체 대시보드는 `00_all_metrics_linear_dashboard.png`다.
