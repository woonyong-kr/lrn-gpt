# E19 qkv_bias True

## 1. 실험 목적

Notion 실험 DB에 정의된 LLM 하이퍼파라미터 변경 효과를 baseline과 비교한다.

## 2. 실행 설정

| 항목 | 값 |
| --- | --- |
| 변경점 | qkv_bias를 False에서 True로 변경 |
| seed | 123 |
| vocab_size | 3000 |
| context_length | 128 |
| emb_dim | 192 |
| n_heads | 4 |
| n_layers | 4 |
| ffn_mult | 4 |
| drop_rate | 0.1 |
| qkv_bias | True |
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
| parameter_count | 2956416 |
| train_chars | 1379486 |
| val_chars | 120560 |
| train_tokens | 805021 |
| val_tokens | 70386 |
| train_batches | 196 |
| val_batches | 17 |
| tokenizer_path | C:\Dev\Crafton-Jungle\04.AI\wk13_6_gpt\data\nsmc_bpe_vocab_3000.json |
| tokenizer_loaded | True |
| tokenizer_elapsed_sec | 0.004 |

## 3. 결과 요약

| 지표 | 값 |
| --- | ---: |
| final_train_loss | 5.081880 |
| final_val_loss | 5.295729 |
| best_val_loss | 5.295729 |
| best_step | 980 |
| final_perplexity | 199.483084 |
| elapsed_sec | 22.440 |
| tokens_per_sec | 180117.091 |

## 4. Step별 지표

| epoch | step | tokens_seen | train_loss | val_loss | lr | grad_norm | elapsed_sec | tokens_per_sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 8.042728 | 8.044044 | 0.0004 |  | 0.000 | 0.000 |
| 1 | 196 | 802816 | 6.691806 | 6.717665 | 0.0004 | 0.449179 | 4.673 | 171804.775 |
| 2 | 392 | 1605632 | 5.598263 | 5.684999 | 0.0004 | 0.601598 | 8.956 | 179277.640 |
| 3 | 500 | 2048000 | 5.433573 | 5.542868 | 0.0004 | 0.645975 | 11.368 | 180149.845 |
| 3 | 588 | 2408448 | 5.335416 | 5.468573 | 0.0004 | 0.715774 | 13.458 | 178961.687 |
| 4 | 784 | 3211264 | 5.187281 | 5.371763 | 0.0004 | 0.717166 | 17.936 | 179035.196 |
| 5 | 980 | 4014080 | 5.081880 | 5.295729 | 0.0004 | 0.801879 | 22.286 | 180117.091 |

## 4-1. 그래프용 CSV

아래 CSV 블록만 파싱하면 loss curve와 throughput curve를 그릴 수 있다.

```csv
epoch,step,tokens_seen,train_loss,val_loss,lr,grad_norm,elapsed_sec,tokens_per_sec
0,0,0,8.04273,8.04404,0.0004,,0,0
1,196,802816,6.69181,6.71766,0.0004,0.449179,4.67284,171805
2,392,1605632,5.59826,5.685,0.0004,0.601598,8.95612,179278
3,500,2048000,5.43357,5.54287,0.0004,0.645975,11.3683,180150
3,588,2408448,5.33542,5.46857,0.0004,0.715774,13.4579,178962
4,784,3211264,5.18728,5.37176,0.0004,0.717166,17.9365,179035
5,980,4014080,5.08188,5.29573,0.0004,0.801879,22.2859,180117
```

## 5. 생성 샘플

prompt: `이 영화는`

```text
이 영화는 아닌가 아니였는데 이건 진짜.. 이런영화는 10점이에요? 긴장을 지겨지...
아놔서는 아니야지나의 그닥..
내일의 시걸의 매력을 느끼게 해주며 보이면에서 해석는구나. 정말 재밌게 잘 봤습니다.
췰,. 마지막장면이 아니라, 김민아내 생애기, 김동, 연기, 음악, 그리고, 연출,, 연기, 이용
별 18년만에 1회는 그냥 나았는데, 이 영화가 아니다. 마지막엔 더 감동도 없고 그녀와, 정말
```

## 6. Checkpoint

저장하지 않음

## 7. 해석

baseline 결과다. 이후 실험은 이 md의 `final_val_loss`, `best_val_loss`, `tokens_per_sec`, 생성 샘플을 기준으로 비교한다.
