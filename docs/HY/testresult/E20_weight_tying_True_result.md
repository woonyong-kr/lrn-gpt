# E20 weight_tying True

## 1. 실험 목적

Notion 실험 DB에 정의된 LLM 하이퍼파라미터 변경 효과를 baseline과 비교한다.

## 2. 실행 설정

| 항목 | 값 |
| --- | --- |
| 변경점 | token embedding과 LM head weight를 공유 |
| seed | 123 |
| vocab_size | 3000 |
| context_length | 128 |
| emb_dim | 192 |
| n_heads | 4 |
| n_layers | 4 |
| ffn_mult | 4 |
| drop_rate | 0.1 |
| qkv_bias | False |
| weight_tying | True |
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
| parameter_count | 2378112 |
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
| final_train_loss | 5.309371 |
| final_val_loss | 5.427288 |
| best_val_loss | 5.427288 |
| best_step | 980 |
| final_perplexity | 227.531400 |
| elapsed_sec | 21.130 |
| tokens_per_sec | 191334.192 |

## 4. Step별 지표

| epoch | step | tokens_seen | train_loss | val_loss | lr | grad_norm | elapsed_sec | tokens_per_sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 8.040151 | 8.042521 | 0.0004 |  | 0.000 | 0.000 |
| 1 | 196 | 802816 | 7.282590 | 7.285177 | 0.0004 | 0.253720 | 4.501 | 178369.336 |
| 2 | 392 | 1605632 | 6.067143 | 6.107281 | 0.0004 | 0.637550 | 8.479 | 189366.777 |
| 3 | 500 | 2048000 | 5.788539 | 5.842728 | 0.0004 | 0.655513 | 10.765 | 190238.011 |
| 3 | 588 | 2408448 | 5.645436 | 5.714779 | 0.0004 | 0.652458 | 12.680 | 189945.880 |
| 4 | 784 | 3211264 | 5.439666 | 5.543079 | 0.0004 | 0.687987 | 16.919 | 189799.501 |
| 5 | 980 | 4014080 | 5.309371 | 5.427288 | 0.0004 | 0.664499 | 20.979 | 191334.192 |

## 4-1. 그래프용 CSV

아래 CSV 블록만 파싱하면 loss curve와 throughput curve를 그릴 수 있다.

```csv
epoch,step,tokens_seen,train_loss,val_loss,lr,grad_norm,elapsed_sec,tokens_per_sec
0,0,0,8.04015,8.04252,0.0004,,0,0
1,196,802816,7.28259,7.28518,0.0004,0.25372,4.50086,178369
2,392,1605632,6.06714,6.10728,0.0004,0.63755,8.47895,189367
3,500,2048000,5.78854,5.84273,0.0004,0.655513,10.7655,190238
3,588,2408448,5.64544,5.71478,0.0004,0.652458,12.6797,189946
4,784,3211264,5.43967,5.54308,0.0004,0.687987,16.9192,189800
5,980,4014080,5.30937,5.42729,0.0004,0.664499,20.9794,191334
```

## 5. 생성 샘플

prompt: `이 영화는`

```text
이 영화는 너무 잘만들어서 정말 정말 재미없다 ㅋㅋ
이 영화는 처음이다 이런 영화는 그냥 그냥 쓰레기 영화관에서 봤다.
아요. 그것들 중요~ 이 영화가 아니다
12편은 볼 수 있을 줄거리지만 정말 좋다고 한편에서 보고있게 느껴진다
난다
난.. 정말 지루해서 볼수있는 영화.. 이런 영화보다 낮아내 생애들로는 건 않았을까? 정말 재미있게 봤는데 ㅎ
이 영화는 이래
별점.
저저저도 그지 모르는 것만만..이 아니다 진짜...정말 재미도 없고
정말 재미있게 봤
```

## 6. Checkpoint

저장하지 않음

## 7. 해석

baseline 결과다. 이후 실험은 이 md의 `final_val_loss`, `best_val_loss`, `tokens_per_sec`, 생성 샘플을 기준으로 비교한다.
