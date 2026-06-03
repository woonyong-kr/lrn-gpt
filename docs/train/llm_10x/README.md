# LLM 10x Long-Run Workspace

This folder tracks the human-readable plan, run matrix, and aggregate reports
for the duplicate-free LLM-heavy 10x corpus experiments.

## Tracked Files

- `experiment_plan.md`: broad sweep plan and interpretation rules
- `run_matrix.csv`: 118 physical exploratory conditions x 3 seeds = 354 planned runs
- `aggregate_summary.csv`: generated condition-level statistics
- `aggregate_report.md`: generated human-readable report
- `aggregate_meta.json`: generated aggregate status

## Local Runtime Files

Large runtime artifacts live under `local/llm_10x/` and are intentionally not
tracked by git.

- `local/llm_10x/cache/vocab_*_minfreq_*/`: tokenizer and token-id caches
- `local/llm_10x/runs/run_*/`: run configs, metrics, history, checkpoints
- `local/llm_10x/queue_events.jsonl`: resumable queue event log
- `local/llm_10x/queue_status.json`: current queue status

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
