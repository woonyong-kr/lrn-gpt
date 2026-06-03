# SMOKE runner smoke

## 1. 실험 목적

러너 검증

## 2. 실행 설정

| 항목 | 값 |
| --- | --- |
| 변경점 | 작은 토큰 제한 |
| seed | 123 |
| vocab_size | 300 |
| context_length | 16 |
| emb_dim | 32 |
| n_heads | 2 |
| n_layers | 1 |
| ffn_mult | 4 |
| drop_rate | 0.1 |
| qkv_bias | False |
| weight_tying | False |
| batch_size | 4 |
| num_epochs | 1 |
| lr | 0.0004 |
| weight_decay | 0.1 |
| eval_freq | 2 |
| eval_iter | 1 |
| stride | 16 |
| amp | none |
| device | NVIDIA GeForce RTX 5070 Ti |
| torch | 2.13.0.dev20260602+cu132 |
| parameter_count | 32384 |
| train_chars | 5000 |
| val_chars | 5000 |
| train_tokens | 512 |
| val_tokens | 256 |
| train_batches | 7 |
| val_batches | 3 |
| tokenizer_path | C:\Dev\Crafton-Jungle\04.AI\wk13_6_gpt\data\nsmc_bpe_vocab_300.json |
| tokenizer_loaded | True |
| tokenizer_elapsed_sec | 0.006 |

## 3. 결과 요약

| 지표 | 값 |
| --- | ---: |
| final_train_loss | 5.692559 |
| final_val_loss | 5.677442 |
| best_val_loss | 5.677442 |
| best_step | 7 |
| final_perplexity | 292.200905 |
| elapsed_sec | 0.445 |
| tokens_per_sec | 1018.551 |

## 4. Step별 지표

| epoch | step | tokens_seen | train_loss | val_loss | lr | grad_norm | elapsed_sec | tokens_per_sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 5.728890 | 5.694684 | 0.0004 |  | 0.000 | 0.000 |
| 1 | 2 | 128 | 5.684192 | 5.691557 | 0.0004 | 1.468820 | 0.393 | 326.088 |
| 1 | 4 | 256 | 5.674343 | 5.686496 | 0.0004 | 1.247218 | 0.412 | 621.667 |
| 1 | 6 | 384 | 5.676028 | 5.680644 | 0.0004 | 1.202011 | 0.429 | 895.217 |
| 1 | 7 | 448 | 5.692559 | 5.677442 | 0.0004 | 1.228640 | 0.440 | 1018.551 |

## 4-1. 그래프용 CSV

아래 CSV 블록만 파싱하면 loss curve와 throughput curve를 그릴 수 있다.

```csv
epoch,step,tokens_seen,train_loss,val_loss,lr,grad_norm,elapsed_sec,tokens_per_sec
0,0,0,5.72889,5.69468,0.0004,,0,0
1,2,128,5.68419,5.69156,0.0004,1.46882,0.392532,326.088
1,4,256,5.67434,5.6865,0.0004,1.24722,0.411796,621.667
1,6,384,5.67603,5.68064,0.0004,1.20201,0.428946,895.217
1,7,448,5.69256,5.67744,0.0004,1.22864,0.439841,1018.55
```

## 5. 생성 샘플

prompt: `이 영화는`

```text
이 영화는"���64
```

## 6. Checkpoint

저장하지 않음

## 7. 해석

baseline 결과다. 이후 실험은 이 md의 `final_val_loss`, `best_val_loss`, `tokens_per_sec`, 생성 샘플을 기준으로 비교한다.
