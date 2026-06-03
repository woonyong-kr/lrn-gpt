# E13 n_layers 8

## 1. 실험 목적

Notion 실험 DB에 정의된 LLM 하이퍼파라미터 변경 효과를 baseline과 비교한다.

## 2. 실행 설정

| 항목 | 값 |
| --- | --- |
| 변경점 | n_layers를 4에서 8로 증가 |
| seed | 123 |
| vocab_size | 3000 |
| context_length | 128 |
| emb_dim | 192 |
| n_heads | 4 |
| n_layers | 8 |
| ffn_mult | 4 |
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
| parameter_count | 4731264 |
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
| final_train_loss | 5.122186 |
| final_val_loss | 5.317900 |
| best_val_loss | 5.317900 |
| best_step | 980 |
| final_perplexity | 203.955083 |
| elapsed_sec | 36.800 |
| tokens_per_sec | 109846.398 |

## 4. Step별 지표

| epoch | step | tokens_seen | train_loss | val_loss | lr | grad_norm | elapsed_sec | tokens_per_sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 8.042649 | 8.043825 | 0.0004 |  | 0.000 | 0.000 |
| 1 | 196 | 802816 | 7.300470 | 7.298753 | 0.0004 | 0.237963 | 7.769 | 103333.377 |
| 2 | 392 | 1605632 | 5.840206 | 5.899497 | 0.0004 | 0.541384 | 15.036 | 106783.398 |
| 3 | 500 | 2048000 | 5.560887 | 5.646330 | 0.0004 | 0.576523 | 19.084 | 107315.702 |
| 3 | 588 | 2408448 | 5.427788 | 5.538263 | 0.0004 | 0.646316 | 22.332 | 107847.007 |
| 4 | 784 | 3211264 | 5.250524 | 5.411654 | 0.0004 | 0.737026 | 29.371 | 109333.757 |
| 5 | 980 | 4014080 | 5.122186 | 5.317900 | 0.0004 | 0.802238 | 36.543 | 109846.398 |

## 4-1. 그래프용 CSV

아래 CSV 블록만 파싱하면 loss curve와 throughput curve를 그릴 수 있다.

```csv
epoch,step,tokens_seen,train_loss,val_loss,lr,grad_norm,elapsed_sec,tokens_per_sec
0,0,0,8.04265,8.04382,0.0004,,0,0
1,196,802816,7.30047,7.29875,0.0004,0.237963,7.76918,103333
2,392,1605632,5.84021,5.8995,0.0004,0.541384,15.0363,106783
3,500,2048000,5.56089,5.64633,0.0004,0.576523,19.0839,107316
3,588,2408448,5.42779,5.53826,0.0004,0.646316,22.3321,107847
4,784,3211264,5.25052,5.41165,0.0004,0.737026,29.3712,109334
5,980,4014080,5.12219,5.3179,0.0004,0.802238,36.5427,109846
```

## 5. 생성 샘플

prompt: `이 영화는`

```text
이 영화는 아니지만..그렇게 재밌는 장면의 의미를 잘 알려야지지만 그래도 너무 재밌는다큐브루와 함께 봤는데...ㅋㅋ
사세를 위험한민과는 것, 그 당시에 미모를 통해한 이야기를 보시킨다. 아프로 만든 작품을 보여주는 영화로 이따위는줄 모르는다.
이런영화.. 이런 영화..그라스런영화..진짜 재미없다!!
좋은 영화
우연히 보게되고 감동을 못보는데?ㅋㅋ
웃김
내요소녀는영화!!
오래서 1점
```

## 6. Checkpoint

저장하지 않음

## 7. 해석

baseline 결과다. 이후 실험은 이 md의 `final_val_loss`, `best_val_loss`, `tokens_per_sec`, 생성 샘플을 기준으로 비교한다.
