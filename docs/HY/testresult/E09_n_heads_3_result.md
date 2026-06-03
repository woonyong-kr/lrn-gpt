# E09 n_heads 3

## 1. 실험 목적

Notion 실험 DB에 정의된 LLM 하이퍼파라미터 변경 효과를 baseline과 비교한다.

## 2. 실행 설정

| 항목 | 값 |
| --- | --- |
| 변경점 | n_heads를 4에서 3으로 감소 |
| seed | 123 |
| vocab_size | 3000 |
| context_length | 128 |
| emb_dim | 192 |
| n_heads | 3 |
| n_layers | 4 |
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
| parameter_count | 2954112 |
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
| final_train_loss | 5.070941 |
| final_val_loss | 5.288454 |
| best_val_loss | 5.288454 |
| best_step | 980 |
| final_perplexity | 198.037073 |
| elapsed_sec | 20.201 |
| tokens_per_sec | 200458.658 |

## 4. Step별 지표

| epoch | step | tokens_seen | train_loss | val_loss | lr | grad_norm | elapsed_sec | tokens_per_sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 8.044979 | 8.045688 | 0.0004 |  | 0.000 | 0.000 |
| 1 | 196 | 802816 | 6.669287 | 6.694404 | 0.0004 | 0.617309 | 4.354 | 184402.586 |
| 2 | 392 | 1605632 | 5.604069 | 5.692151 | 0.0004 | 0.592126 | 8.174 | 196422.056 |
| 3 | 500 | 2048000 | 5.432887 | 5.545979 | 0.0004 | 0.641523 | 10.404 | 196856.095 |
| 3 | 588 | 2408448 | 5.332251 | 5.470461 | 0.0004 | 0.755027 | 12.249 | 196622.430 |
| 4 | 784 | 3211264 | 5.179394 | 5.366281 | 0.0004 | 0.692453 | 16.138 | 198990.944 |
| 5 | 980 | 4014080 | 5.070941 | 5.288454 | 0.0004 | 0.826589 | 20.024 | 200458.658 |

## 4-1. 그래프용 CSV

아래 CSV 블록만 파싱하면 loss curve와 throughput curve를 그릴 수 있다.

```csv
epoch,step,tokens_seen,train_loss,val_loss,lr,grad_norm,elapsed_sec,tokens_per_sec
0,0,0,8.04498,8.04569,0.0004,,0,0
1,196,802816,6.66929,6.6944,0.0004,0.617309,4.3536,184403
2,392,1605632,5.60407,5.69215,0.0004,0.592126,8.1744,196422
3,500,2048000,5.43289,5.54598,0.0004,0.641523,10.4035,196856
3,588,2408448,5.33225,5.47046,0.0004,0.755027,12.2491,196622
4,784,3211264,5.17939,5.36628,0.0004,0.692453,16.1377,198991
5,980,4014080,5.07094,5.28845,0.0004,0.826589,20.0245,200459
```

## 5. 생성 샘플

prompt: `이 영화는`

```text
이 영화는 아니지만... 너무 재밌던데
내 생겼는데도 재미없네여운이 넘어욤... 금물? ^
스토의 급영화...
이런류의 영화,. 영화관이 많이 봤다,. 이래도 아까운 영화!
이영화로만화가 아깝고도 그리우드의 한다.
아~
칛웃음, 스토리는,. 어중간에 관계와 함께 가능은 그리도 정말 최고다.
정말 재미없는 영화였다. 이리기 힘들다
내요...... 너무 좋아해서 더더만하더라구
```

## 6. Checkpoint

저장하지 않음

## 7. 해석

baseline 결과다. 이후 실험은 이 md의 `final_val_loss`, `best_val_loss`, `tokens_per_sec`, 생성 샘플을 기준으로 비교한다.
