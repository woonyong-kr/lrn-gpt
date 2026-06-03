# LLM 10x 장기 실행 작업 공간

이 폴더는 이전 `docs/train` 실험과 분리된 LLM 10x 실험의 계획, 실행 행렬, 집계 보고서, 전체 실행 결과 원장을 보관합니다.

## 추적 파일

- `experiment_plan.md`: 전체 sweep 계획과 해석 규칙
- `run_matrix.csv`: 118개 물리 탐색 조건 x 3개 seed = 354개 계획 실행
- `all_run_results.jsonl`: 완료된 모든 물리 run의 원본 `result.json`을 한 줄씩 복제한 누락 방지 원장
- `aggregate_summary.csv`: 조건별 생성 통계
- `aggregate_report.md`: 사람이 읽는 집계 보고서
- `aggregate_meta.json`: 집계 상태와 원장 누락 감사 결과
- `screen_ready_report.md`: n>=3 조건을 발표/검토용으로 요약한 중간 보고서
- `llm_developer_report.md`: LLM 개발자 지표 기준으로 다시 쓴 중간 보고서
- `llm_developer_figures/`: bits/char, compute, throughput, tokenizer 공정성 그래프
- `screen_ready_figures/`, `paper_figures/`: screen-ready 조건과 발표용 figure

## 로컬 실행 파일

큰 실행 산출물은 `local/llm_10x_isolated/` 아래에 쌓이며 git 추적 대상이 아닙니다.

- `local/llm_10x_isolated/cache/vocab_*_minfreq_*/`: tokenizer와 token-id cache
- `local/llm_10x_isolated/cache/vocab_*_minfreq_*/tokenizer_profile.json`: tokenizer 해석용 profile
- `local/llm_10x_isolated/cache/vocab_*_minfreq_*/tokenizer_profile.csv`: profile의 표 형식 요약
- `local/llm_10x_isolated/runs/run_*/`: run별 설정, 지표, history, checkpoint
- `local/llm_10x_isolated/queue_events.jsonl`: 재개 가능한 큐 이벤트 로그
- `local/llm_10x_isolated/queue_status.json`: 현재 큐 상태

## Execution Order

1. Generate or refresh the exploratory run matrix.

```bash
python scripts/build_llm_10x_run_matrix.py
```

For a later 10-seed confirmation matrix:

```bash
python scripts/build_llm_10x_run_matrix.py --seed-mode confirm
```

2. Prepare tokenizer/token-id caches.

```bash
python scripts/llm_10x_prepare_cache.py
```

For smoke tests only:

```bash
python scripts/llm_10x_prepare_cache.py --vocab-size 12000 --min-frequency 2 --char-limit 20000 --force
```

3. Run one pending experiment.

```bash
python scripts/llm_10x_run_next.py
```

Or run the resumable queue:

```bash
python scripts/llm_10x_run_queue.py
```

For smoke tests only:

```bash
python scripts/llm_10x_run_next.py --run-number 1 --max-steps 2 --no-checkpoints --force
```

4. Aggregate completed results.

```bash
python scripts/llm_10x_aggregate.py
```

5. Regenerate reports and figures.

```bash
python scripts/llm_10x_screen_ready_report.py
python scripts/llm_10x_llm_developer_report.py
```

6. Optional generation audit when checkpoints exist.

```bash
python scripts/llm_generation_metrics.py \
  --checkpoint local/llm_10x_isolated/runs/run_0001/checkpoints/best.pt \
  --config-json local/llm_10x_isolated/runs/run_0001/result.json \
  --tokenizer-json local/llm_10x_isolated/cache/vocab_12000_minfreq_2/tokenizer.json
```

## Decision Rule

Exploratory conditions become screen-ready at 3 completed repeats. They are not
claim-ready. Final claims require rerunning selected finalists with 10 seeds.

## Growth Sweep Axes

The matrix changes one axis at a time with the same 3 exploratory seeds:

- learning rate: `0.00005` through `0.003`
- epochs: one `1500`-epoch physical long-run per seed, analyzed at `25` through `1500` milestones
- vocab size: `4000` through `64000`
- tokenizer min frequency: `1` through `55`
- capacity: 31M-class through 240M-class
- context length: `256` through `2048`
- batch size: `1` through `16`
- dropout: `0.00` through `0.40`
- weight decay: `0.00` through `0.50`
- grad clip: `0.0` through `5.0`
- FFN multiplier: `2` through `8`
- activation variants: GELU family, ReLU/SILU/Mish, gated variants, identity
- init std: `0.005` through `0.04`
- structure variants: QKV bias, untied embeddings, post-LN, manual attention

The epoch axis is expanded during aggregation. The physical matrix has 354 runs;
the aggregate report has 390 analysis rows after epoch milestone expansion.

## Decision Metrics

Final selection uses four groups together:

- tokenizer-fair quality: `final_val_bits_per_char`, `final_val_nats_per_char`, `best_val_bits_per_char`
- overfit/rebound risk: `overfit_score`, `final_generalization_gap`, `generalization_gap_delta`, `final_minus_best_val_loss`
- exposure and compute: `tokens_seen`, `estimated_chars_seen`, `tokens_per_param`, `compute_proxy`, `estimated_train_flops`
- system cost: `elapsed_sec`, `seconds_per_epoch`, `tokens_per_sec_after_warmup`

Tokenizer changes must not be ranked by `final_val_loss` alone. Vocab and tokenizer-min-frequency phases use bits/char plus tokenizer profile fields:

- `actual_vocab_size`
- `bpe_merge_count`
- `train_tokens_per_char`, `val_tokens_per_char`
- `train_chars_per_token`, `val_chars_per_token`
- `token_length_histogram`
- `top_50_merge_rules`

## Generated Result Schema

New physical runs write the following LLM metric fields to `result.json`:

- `optimizer_updates`
- `best_tokens_seen`
- `best_val_bits_per_char`, `best_val_nats_per_char`
- `final_train_nats_per_char`, `final_val_nats_per_char`
- `final_minus_best_val_loss`
- `estimated_chars_seen`
- `tokens_per_param`
- `compute_proxy`, `estimated_train_flops`
- `tokens_per_sec_after_warmup`
- `peak_gpu_memory_mb`
- `eval_batches`, `eval_tokens`, `eval_chars`

Epoch events in `history.jsonl` include per-character loss, current best rebound, exposure, and warmup-excluded throughput so 1500-epoch long-runs can be analyzed by milestone instead of final-only loss.
