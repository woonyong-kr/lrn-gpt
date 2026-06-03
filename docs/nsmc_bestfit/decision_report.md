# NSMC Best-Fit Decision Report

- updated_at: `2026-06-03T19:30:04.322778+00:00`
- action: `wait_for_pending`
- reason: `pending rows already exist`
- planned_runs: `5`
- completed_runs: `1`
- pending_runs: `4`

## 다음 실행

- run_number: `2`
- condition_id: `TOK_V12000_MF2`
- phase: `phase1_tokenizer`
- purpose: `NSMC byte-level BPE tokenizer vocab=12000 min_frequency=2`
- seed: `123`
- tokenizer: `vocab=12000, min_frequency=2`
- model: `emb=256, heads=8, layers=6, ffn_mult=4`
- optimization: `lr=0.0003, drop=0.1, wd=0.05`

## 현재 champion

- run_number: `1`
- condition_id: `TOK_V8000_MF2`
- phase: `phase1_tokenizer`
- seed: `123`
- score: `6.315407236495098`
- best_val_bits_per_char: `6.187685480587931`
- final_val_bits_per_char: `6.187685480587931`
- final_minus_best_val_loss: `0.0`
- final_generalization_gap: `0.008516311645507812`
- val_tokens_per_char: `0.4995686794956868`
- val_chars_per_token: `2.001726771601249`
- tokens_per_sec_after_warmup: `3817.535533985378`
- parameter_count: `6848000`
- tokens_seen: `10240`
- compute_proxy: `70123520000`

## Phase 상태

| phase | label | planned | completed | best_score | mean_score |
| --- | --- | ---: | ---: | ---: | ---: |
| phase1_tokenizer | tokenizer efficiency | 5 | 1 | 6.31541 | 6.31541 |

## 실행 명령

한 번만 실행:

```bash
python scripts/nsmc_bestfit_step.py --max-runs 1 --device auto --eval-batches 20
```

가벼운 smoke 실행:

```bash
python scripts/nsmc_bestfit_step.py --max-runs 1 --max-steps-per-run 5 --device cpu --eval-batches 2
```
