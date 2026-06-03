# E14 ffn_multiplier 2

## 1. 실험 목적

Notion 실험 DB에 정의된 LLM 하이퍼파라미터 변경 효과를 baseline과 비교한다.

## 2. 실행 설정

| 항목 | 값 |
| --- | --- |
| 변경점 | ffn_multiplier를 4에서 2로 감소 |
| seed | 123 |
| vocab_size | 3000 |
| context_length | 128 |
| emb_dim | 192 |
| n_heads | 4 |
| n_layers | 4 |
| ffn_mult | 2 |
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
| parameter_count | 2362752 |
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
| final_train_loss | 5.066985 |
| final_val_loss | 5.291719 |
| best_val_loss | 5.291719 |
| best_step | 980 |
| final_perplexity | 198.684635 |
| elapsed_sec | 20.449 |
| tokens_per_sec | 197805.724 |

## 4. Step별 지표

| epoch | step | tokens_seen | train_loss | val_loss | lr | grad_norm | elapsed_sec | tokens_per_sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 8.045649 | 8.045464 | 0.0004 |  | 0.000 | 0.000 |
| 1 | 196 | 802816 | 6.810543 | 6.835532 | 0.0004 | 0.448973 | 4.314 | 186081.385 |
| 2 | 392 | 1605632 | 5.599275 | 5.687649 | 0.0004 | 0.590700 | 8.143 | 197169.193 |
| 3 | 500 | 2048000 | 5.425949 | 5.539857 | 0.0004 | 0.646980 | 10.366 | 197567.233 |
| 3 | 588 | 2408448 | 5.323781 | 5.462722 | 0.0004 | 0.740881 | 12.295 | 195883.173 |
| 4 | 784 | 3211264 | 5.172599 | 5.366411 | 0.0004 | 0.712394 | 16.309 | 196898.743 |
| 5 | 980 | 4014080 | 5.066985 | 5.291719 | 0.0004 | 0.820193 | 20.293 | 197805.724 |

## 4-1. 그래프용 CSV

아래 CSV 블록만 파싱하면 loss curve와 throughput curve를 그릴 수 있다.

```csv
epoch,step,tokens_seen,train_loss,val_loss,lr,grad_norm,elapsed_sec,tokens_per_sec
0,0,0,8.04565,8.04546,0.0004,,0,0
1,196,802816,6.81054,6.83553,0.0004,0.448973,4.31433,186081
2,392,1605632,5.59927,5.68765,0.0004,0.5907,8.14342,197169
3,500,2048000,5.42595,5.53986,0.0004,0.64698,10.3661,197567
3,588,2408448,5.32378,5.46272,0.0004,0.740881,12.2953,195883
4,784,3211264,5.1726,5.36641,0.0004,0.712394,16.3092,196899
5,980,4014080,5.06699,5.29172,0.0004,0.820193,20.293,197806
```

## 5. 생성 샘플

prompt: `이 영화는`

```text
이 영화는 아닌가 아니지만..그나마.. 이거 볼거리나네? 0점이에요....이건뭐지나?
그대로된영화.. 이리티가.. ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ
역시.. 평점은 너무 재미있게 봤습니다~♥~^^
아무리웃기분하고 봤는데.. 정말 지루해서 못할정도로.. 이런 영화보다 낮아가면 진짜 최고였는데..ㅎㅎ
아이 좀더군요. 그냥 ㄷㄷ
내도 아까운영화인데 18년만에 영화보고..이기세가 아깝네요. 정말 좋다고 하지 말고 재미도 없고.
저럭저
```

## 6. Checkpoint

저장하지 않음

## 7. 해석

baseline 결과다. 이후 실험은 이 md의 `final_val_loss`, `best_val_loss`, `tokens_per_sec`, 생성 샘플을 기준으로 비교한다.
