# E02 context_length 192

## 1. 실험 목적

128보다 긴 문맥 길이가 validation loss와 처리량에 미치는 영향 확인

## 2. 실행 설정

| 항목 | 값 |
| --- | --- |
| 변경점 | context_length를 128에서 192로 증가. |
| seed | 123 |
| vocab_size | 3000 |
| context_length | 192 |
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
| stride | 192 |
| amp | bf16 |
| device | NVIDIA GeForce RTX 5070 Ti |
| torch | 2.13.0.dev20260602+cu132 |
| parameter_count | 2966400 |
| train_chars | 1379486 |
| val_chars | 120560 |
| train_tokens | 805021 |
| val_tokens | 70386 |
| train_batches | 131 |
| val_batches | 11 |
| tokenizer_path | C:\Dev\Crafton-Jungle\04.AI\wk13_6_gpt\data\nsmc_bpe_vocab_3000.json |
| tokenizer_loaded | True |
| tokenizer_elapsed_sec | 0.004 |

## 3. 결과 요약

| 지표 | 값 |
| --- | ---: |
| final_train_loss | 5.203910 |
| final_val_loss | 5.388636 |
| best_val_loss | 5.388636 |
| best_step | 655 |
| final_perplexity | 218.904658 |
| elapsed_sec | 14.254 |
| tokens_per_sec | 285631.866 |

## 4. Step별 지표

| epoch | step | tokens_seen | train_loss | val_loss | lr | grad_norm | elapsed_sec | tokens_per_sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 8.044989 | 8.043985 | 0.0004 |  | 0.000 | 0.000 |
| 1 | 131 | 804864 | 7.225841 | 7.235859 | 0.0004 | 0.311191 | 3.002 | 268095.204 |
| 2 | 262 | 1609728 | 5.898838 | 5.950554 | 0.0004 | 0.509377 | 5.586 | 288148.197 |
| 3 | 393 | 2414592 | 5.488586 | 5.596244 | 0.0004 | 0.548150 | 8.285 | 291438.572 |
| 4 | 500 | 3072000 | 5.344141 | 5.481361 | 0.0004 | 0.548111 | 10.618 | 289319.772 |
| 4 | 524 | 3219456 | 5.322625 | 5.461768 | 0.0004 | 0.563521 | 11.279 | 285435.697 |
| 5 | 655 | 4024320 | 5.203910 | 5.388636 | 0.0004 | 0.575668 | 14.089 | 285631.866 |

## 4-1. 그래프용 CSV

아래 CSV 블록만 파싱하면 loss curve와 throughput curve를 그릴 수 있다.

```csv
epoch,step,tokens_seen,train_loss,val_loss,lr,grad_norm,elapsed_sec,tokens_per_sec
0,0,0,8.04499,8.04399,0.0004,,0,0
1,131,804864,7.22584,7.23586,0.0004,0.311191,3.00216,268095
2,262,1609728,5.89884,5.95055,0.0004,0.509377,5.58646,288148
3,393,2414592,5.48859,5.59624,0.0004,0.54815,8.28508,291439
4,500,3072000,5.34414,5.48136,0.0004,0.548111,10.618,289320
4,524,3219456,5.32263,5.46177,0.0004,0.563521,11.2791,285436
5,655,4024320,5.20391,5.38864,0.0004,0.575668,14.0892,285632
```

## 5. 생성 샘플

prompt: `이 영화는`

```text
이 영화는 왜 평점이 낮지 마세요.
재밌었어요^^
너무 지루해..아들아직도 그 자체가..그때마다 이정도면이제에서는 장면들 다소룡을 위해 이런영화.
정말 재미있어요!!
이걸이면에 반개도 아깝다.
정말 재미있음 ㅋㅋ
아직비디오용한 내용은 왜 못생긴데?
어.ㅡ 긴장되고...이런건 없을 보는것만 있어나와의 사랑은듯..
재밌게만 보지 모르겠습니다.
전문제일관계속에 나의 미드워드리
```

## 6. Checkpoint

저장하지 않음

## 7. 해석

baseline 결과다. 이후 실험은 이 md의 `final_val_loss`, `best_val_loss`, `tokens_per_sec`, 생성 샘플을 기준으로 비교한다.
