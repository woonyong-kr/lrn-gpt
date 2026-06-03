# E15 ffn_multiplier 6

## 1. 실험 목적

Notion 실험 DB에 정의된 LLM 하이퍼파라미터 변경 효과를 baseline과 비교한다.

## 2. 실행 설정

| 항목 | 값 |
| --- | --- |
| 변경점 | ffn_multiplier를 4에서 6으로 증가 |
| seed | 123 |
| vocab_size | 3000 |
| context_length | 128 |
| emb_dim | 192 |
| n_heads | 4 |
| n_layers | 4 |
| ffn_mult | 6 |
| drop_rate | 0.1 |
| qkv_bias | False |
| weight_tying | False |
| batch_size | 32 |
| num_epochs | 5 |
| lr | 0.0004 |
| weight_decay | 0.1 |
| eval_freq | 500 |
| eval_iter | 20 |
| stride | 128 |
| amp | bf16 |
| device | NVIDIA GeForce RTX 5070 Ti |
| torch | 2.13.0.dev20260602+cu132 |
| parameter_count | 3545472 |
| train_chars | 1379486 |
| val_chars | 120560 |
| train_tokens | 805021 |
| val_tokens | 70386 |
| train_batches | 196 |
| val_batches | 17 |
| tokenizer_path | C:\Dev\Crafton-Jungle\04.AI\wk13_6_gpt\data\nsmc_bpe_vocab_3000.json |
| tokenizer_loaded | True |
| tokenizer_elapsed_sec | 0.009 |

## 3. 결과 요약

| 지표 | 값 |
| --- | ---: |
| final_train_loss | 5.096516 |
| final_val_loss | 5.300615 |
| best_val_loss | 5.300615 |
| best_step | 980 |
| final_perplexity | 200.460067 |
| elapsed_sec | 20.957 |
| tokens_per_sec | 192962.714 |

## 4. Step별 지표

| epoch | step | tokens_seen | train_loss | val_loss | lr | grad_norm | elapsed_sec | tokens_per_sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 8.045025 | 8.042199 | 0.0004 |  | 0.000 | 0.000 |
| 1 | 196 | 802816 | 6.849262 | 6.867748 | 0.0004 | 0.444878 | 4.554 | 176271.865 |
| 2 | 392 | 1605632 | 5.630153 | 5.712702 | 0.0004 | 0.607480 | 8.494 | 189035.640 |
| 3 | 500 | 2048000 | 5.456366 | 5.558895 | 0.0004 | 0.658885 | 10.708 | 191259.395 |
| 3 | 588 | 2408448 | 5.356231 | 5.482859 | 0.0004 | 0.688613 | 12.565 | 191685.208 |
| 4 | 784 | 3211264 | 5.205678 | 5.381299 | 0.0004 | 0.743513 | 16.833 | 190770.823 |
| 5 | 980 | 4014080 | 5.096516 | 5.300615 | 0.0004 | 0.820229 | 20.802 | 192962.714 |

## 4-1. 그래프용 CSV

아래 CSV 블록만 파싱하면 loss curve와 throughput curve를 그릴 수 있다.

```csv
epoch,step,tokens_seen,train_loss,val_loss,lr,grad_norm,elapsed_sec,tokens_per_sec
0,0,0,8.04503,8.0422,0.0004,,0,0
1,196,802816,6.84926,6.86775,0.0004,0.444878,4.55442,176272
2,392,1605632,5.63015,5.7127,0.0004,0.60748,8.49381,189036
3,500,2048000,5.45637,5.55889,0.0004,0.658885,10.708,191259
3,588,2408448,5.35623,5.48286,0.0004,0.688613,12.5646,191685
4,784,3211264,5.20568,5.3813,0.0004,0.743513,16.8331,190771
5,980,4014080,5.09652,5.30062,0.0004,0.820229,20.8024,192963
```

## 5. 생성 샘플

prompt: `이 영화는`

```text
이 영화는 아닌가지고 있는듯..
정말 재미있게봤다
저예전드먼이걸까진이하다가 아저는 드라마였다
나의 유이코메디가의 시걸로 보겠다는 거리로만 남는 장면이 아니라는다큐는 있다는 사람들만하면,. 마지막장면이 아니라, 김민아가, 오히로인, 대사로우러운 영화. 그때, 연기, 이래스스를 말로로를 잘하는듯...
평점을 것만하면 좋다고생각하고...정말 잘 못해준거 없
```

## 6. Checkpoint

저장하지 않음

## 7. 해석

baseline 결과다. 이후 실험은 이 md의 `final_val_loss`, `best_val_loss`, `tokens_per_sec`, 생성 샘플을 기준으로 비교한다.
