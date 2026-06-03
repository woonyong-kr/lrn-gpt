# E11 n_layers 2

## 1. 실험 목적

Notion 실험 DB에 정의된 LLM 하이퍼파라미터 변경 효과를 baseline과 비교한다.

## 2. 실행 설정

| 항목 | 값 |
| --- | --- |
| 변경점 | n_layers를 4에서 2로 감소 |
| seed | 123 |
| vocab_size | 3000 |
| context_length | 128 |
| emb_dim | 192 |
| n_heads | 4 |
| n_layers | 2 |
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
| parameter_count | 2065536 |
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
| final_train_loss | 5.098812 |
| final_val_loss | 5.321256 |
| best_val_loss | 5.321256 |
| best_step | 980 |
| final_perplexity | 204.640668 |
| elapsed_sec | 12.499 |
| tokens_per_sec | 324062.829 |

## 4. Step별 지표

| epoch | step | tokens_seen | train_loss | val_loss | lr | grad_norm | elapsed_sec | tokens_per_sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 8.042977 | 8.042758 | 0.0004 |  | 0.000 | 0.000 |
| 1 | 196 | 802816 | 6.375834 | 6.409427 | 0.0004 | 0.507520 | 2.699 | 297481.158 |
| 2 | 392 | 1605632 | 5.534979 | 5.631564 | 0.0004 | 0.659019 | 5.142 | 312233.460 |
| 3 | 500 | 2048000 | 5.392826 | 5.511490 | 0.0004 | 0.668619 | 6.501 | 315039.966 |
| 3 | 588 | 2408448 | 5.306192 | 5.447584 | 0.0004 | 0.753076 | 7.586 | 317491.344 |
| 4 | 784 | 3211264 | 5.175719 | 5.370174 | 0.0004 | 0.707767 | 9.982 | 321702.982 |
| 5 | 980 | 4014080 | 5.098812 | 5.321256 | 0.0004 | 0.787043 | 12.387 | 324062.829 |

## 4-1. 그래프용 CSV

아래 CSV 블록만 파싱하면 loss curve와 throughput curve를 그릴 수 있다.

```csv
epoch,step,tokens_seen,train_loss,val_loss,lr,grad_norm,elapsed_sec,tokens_per_sec
0,0,0,8.04298,8.04276,0.0004,,0,0
1,196,802816,6.37583,6.40943,0.0004,0.50752,2.69871,297481
2,392,1605632,5.53498,5.63156,0.0004,0.659019,5.14241,312233
3,500,2048000,5.39283,5.51149,0.0004,0.668619,6.50076,315040
3,588,2408448,5.30619,5.44758,0.0004,0.753076,7.58587,317491
4,784,3211264,5.17572,5.37017,0.0004,0.707767,9.98208,321703
5,980,4014080,5.09881,5.32126,0.0004,0.787043,12.3867,324063
```

## 5. 생성 샘플

prompt: `이 영화는`

```text
이 영화는 나았어요...조연과는 드라마.18년도 아니지만, 그래도 전에 있는지.
아프네? 그때마다 이영화를 보며 볼줄알았지만...아프리오의 연기력있다음?!
이때가 아깝다!!
좋은 영화를 볼 수 있다. ㄷㄷ
별한 연기자단말고도 감동. 이어버리고 싶어.ㅡ. 그리고 그 자체가 될것들 중에 너무나왔는데 ㅉㅉ
내인생을 바람하더라구요...아름이 아니다. 그리고 이뇌를 하면
```

## 6. Checkpoint

저장하지 않음

## 7. 해석

baseline 결과다. 이후 실험은 이 md의 `final_val_loss`, `best_val_loss`, `tokens_per_sec`, 생성 샘플을 기준으로 비교한다.
