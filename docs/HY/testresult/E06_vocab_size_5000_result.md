# E06 vocab_size 5000

## 1. 실험 목적

Notion 실험 DB에 정의된 LLM 하이퍼파라미터 변경 효과를 baseline과 비교한다.

## 2. 실행 설정

| 항목 | 값 |
| --- | --- |
| 변경점 | vocab_size를 3000에서 5000으로 증가 |
| seed | 123 |
| vocab_size | 5000 |
| context_length | 128 |
| emb_dim | 192 |
| n_heads | 4 |
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
| parameter_count | 3722112 |
| train_chars | 1379486 |
| val_chars | 120560 |
| train_tokens | 712949 |
| val_tokens | 62493 |
| train_batches | 174 |
| val_batches | 15 |
| tokenizer_path | C:\Dev\Crafton-Jungle\04.AI\wk13_6_gpt\data\nsmc_bpe_vocab_5000.json |
| tokenizer_loaded | False |
| tokenizer_elapsed_sec | 1601.617 |

## 3. 결과 요약

| 지표 | 값 |
| --- | ---: |
| final_train_loss | 5.552528 |
| final_val_loss | 5.882533 |
| best_val_loss | 5.882533 |
| best_step | 870 |
| final_perplexity | 358.716782 |
| elapsed_sec | 19.039 |
| tokens_per_sec | 188826.050 |

## 4. Step별 지표

| epoch | step | tokens_seen | train_loss | val_loss | lr | grad_norm | elapsed_sec | tokens_per_sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 8.554554 | 8.553164 | 0.0004 |  | 0.000 | 0.000 |
| 1 | 174 | 712704 | 7.738937 | 7.729345 | 0.0004 | 0.233472 | 4.042 | 176342.195 |
| 2 | 348 | 1425408 | 6.638267 | 6.702500 | 0.0004 | 0.562244 | 7.678 | 185657.508 |
| 3 | 500 | 2048000 | 6.051611 | 6.198396 | 0.0004 | 0.620245 | 10.828 | 189133.840 |
| 3 | 522 | 2138112 | 5.989175 | 6.156444 | 0.0004 | 0.626847 | 11.408 | 187426.836 |
| 4 | 696 | 2850816 | 5.728675 | 5.967237 | 0.0004 | 0.786555 | 15.140 | 188299.250 |
| 5 | 870 | 3563520 | 5.552528 | 5.882533 | 0.0004 | 0.812397 | 18.872 | 188826.050 |

## 4-1. 그래프용 CSV

아래 CSV 블록만 파싱하면 loss curve와 throughput curve를 그릴 수 있다.

```csv
epoch,step,tokens_seen,train_loss,val_loss,lr,grad_norm,elapsed_sec,tokens_per_sec
0,0,0,8.55455,8.55316,0.0004,,0,0
1,174,712704,7.73894,7.72934,0.0004,0.233472,4.0416,176342
2,348,1425408,6.63827,6.7025,0.0004,0.562244,7.67762,185658
3,500,2048000,6.05161,6.1984,0.0004,0.620245,10.8283,189134
3,522,2138112,5.98917,6.15644,0.0004,0.626847,11.4077,187427
4,696,2850816,5.72867,5.96724,0.0004,0.786555,15.1398,188299
5,870,3563520,5.55253,5.88253,0.0004,0.812397,18.872,188826
```

## 5. 생성 샘플

prompt: `이 영화는`

```text
이 영화는 너무 멋잇음
너무좋아요..난한 내용에 나음...
전형적인 미국영화는 처음
미스텔에 빠져들..
너무 재밌다. 그리고 아래요일때로는 장면들을 비행어지고 있어야지하는 배우들도 다양이슨의 미리나... 영화나싶은 이것이 아닌것 같다.
별점에서 자고.ㅠ. 제니가 나서서 다시보고싶어요
그냥저런로움을 바이 잘해가는줄거는 너무 지루함은 이것도 아닌거리가라.
아놔서양한 설정과거라
```

## 6. Checkpoint

저장하지 않음

## 7. 해석

baseline 결과다. 이후 실험은 이 md의 `final_val_loss`, `best_val_loss`, `tokens_per_sec`, 생성 샘플을 기준으로 비교한다.
