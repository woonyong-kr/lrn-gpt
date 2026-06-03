# E00 기준 모델

## 1. 실험 목적

LLM 하이퍼파라미터 비교의 기준이 되는 baseline 사전학습 결과를 확보한다.

## 2. 실행 설정

| 항목 | 값 |
| --- | --- |
| 변경점 | baseline |
| seed | 123 |
| vocab_size | 3000 |
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
| parameter_count | 2954112 |
| train_chars | 1379486 |
| val_chars | 120560 |
| train_tokens | 805021 |
| val_tokens | 70386 |
| train_batches | 196 |
| val_batches | 17 |
| tokenizer_path | C:\Dev\Crafton-Jungle\04.AI\wk13_6_gpt\data\nsmc_bpe_vocab_3000.json |
| tokenizer_loaded | False |
| tokenizer_elapsed_sec | 1019.986 |

## 3. 결과 요약

| 지표 | 값 |
| --- | ---: |
| final_train_loss | 5.081426 |
| final_val_loss | 5.301974 |
| best_val_loss | 5.301974 |
| best_step | 980 |
| final_perplexity | 200.732742 |
| elapsed_sec | 20.732 |
| tokens_per_sec | 195309.463 |

## 4. Step별 지표

| epoch | step | tokens_seen | train_loss | val_loss | lr | grad_norm | elapsed_sec | tokens_per_sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 8.044979 | 8.045686 | 0.0004 |  | 0.000 | 0.000 |
| 1 | 196 | 802816 | 6.668664 | 6.694039 | 0.0004 | 0.644256 | 4.266 | 188200.403 |
| 2 | 392 | 1605632 | 5.605541 | 5.693040 | 0.0004 | 0.590515 | 8.088 | 198511.036 |
| 3 | 500 | 2048000 | 5.434109 | 5.545916 | 0.0004 | 0.633486 | 10.344 | 197987.898 |
| 3 | 588 | 2408448 | 5.333419 | 5.471000 | 0.0004 | 0.733200 | 12.328 | 195362.149 |
| 4 | 784 | 3211264 | 5.187092 | 5.374210 | 0.0004 | 0.686664 | 16.501 | 194607.697 |
| 5 | 980 | 4014080 | 5.081426 | 5.301974 | 0.0004 | 0.772134 | 20.552 | 195309.463 |

## 4-1. 그래프용 CSV

아래 CSV 블록만 파싱하면 loss curve와 throughput curve를 그릴 수 있다.

```csv
epoch,step,tokens_seen,train_loss,val_loss,lr,grad_norm,elapsed_sec,tokens_per_sec
0,0,0,8.04498,8.04569,0.0004,,0,0
1,196,802816,6.66866,6.69404,0.0004,0.644256,4.26575,188200
2,392,1605632,5.60554,5.69304,0.0004,0.590515,8.08838,198511
3,500,2048000,5.43411,5.54592,0.0004,0.633486,10.3441,197988
3,588,2408448,5.33342,5.471,0.0004,0.7332,12.3281,195362
4,784,3211264,5.18709,5.37421,0.0004,0.686664,16.5012,194608
5,980,4014080,5.08143,5.30197,0.0004,0.772134,20.5524,195309
```

## 5. 생성 샘플

prompt: `이 영화는`

```text
이 영화는 아닌가 아니였다고 할수가 없었음. 정말 재밌게 봤는데 완전히 재미잠까?
정말 잼있게봤으면좋은 영화였다.
내놔
가 좋아요 ....
추억지금봐도 재밋어~♥
카드로서는 좋았으나, 이러지,.ㅡ,이코메디.,아가,2편한 이야기,2,2편까지,2의 OST도는,2편의 반복을 더 좋았다. 다 보고있다.
굿굿굿
다 진짜 재미없고 재미도 없고 감동받아서 낮
```

## 6. Checkpoint

저장하지 않음

## 7. 해석

baseline 결과다. 이후 실험은 이 md의 `final_val_loss`, `best_val_loss`, `tokens_per_sec`, 생성 샘플을 기준으로 비교한다.
