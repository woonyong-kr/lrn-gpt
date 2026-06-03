# LLM 10x 장기 실행 작업 공간

이 폴더는 이전 `docs/train` 실험과 분리된 LLM 10x 실험의 계획, 실행 행렬, 집계 보고서, 전체 실행 결과 원장을 보관합니다.

## 추적 파일

- `experiment_plan.md`: 전체 sweep 계획과 해석 규칙
- `run_matrix.csv`: 118개 물리 탐색 조건 x 3개 seed = 354개 계획 실행
- `all_run_results.jsonl`: 완료된 모든 물리 run의 원본 `result.json`을 한 줄씩 복제한 누락 방지 원장
- `aggregate_summary.csv`: 조건별 생성 통계
- `aggregate_report.md`: 사람이 읽는 집계 보고서
- `aggregate_meta.json`: 집계 상태와 원장 누락 감사 결과

## 로컬 실행 파일

큰 실행 산출물은 `local/llm_10x_isolated/` 아래에 쌓이며 git 추적 대상이 아닙니다.

- `local/llm_10x_isolated/cache/vocab_*_minfreq_*/`: tokenizer와 token-id cache
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

Final selection uses three groups together:

- quality: `final_val_bits_per_char`, `final_val_loss`, `best_val_loss`
- overfit risk: `overfit_score`, `final_generalization_gap`, `generalization_gap_delta`
- cost: `elapsed_sec`, `seconds_per_epoch`, `tokens_per_sec`
