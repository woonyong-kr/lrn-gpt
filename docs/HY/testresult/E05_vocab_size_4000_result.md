# E05 vocab_size 4000

## 1. 실험 목적

Notion 실험 DB에 정의된 LLM 하이퍼파라미터 변경 효과를 baseline과 비교한다.

## 2. 실행 설정

| 항목 | 값 |
| --- | --- |
| 변경점 | vocab_size를 3000에서 4000으로 증가 |
| seed | 123 |
| vocab_size | 4000 |
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
| parameter_count | 3338112 |
| train_chars | 1379486 |
| val_chars | 120560 |
| train_tokens | 750427 |
| val_tokens | 65634 |
| train_batches | 183 |
| val_batches | 16 |
| tokenizer_path | C:\Dev\Crafton-Jungle\04.AI\wk13_6_gpt\data\nsmc_bpe_vocab_4000.json |
| tokenizer_loaded | False |
| tokenizer_elapsed_sec | 1336.408 |

## 3. 결과 요약

| 지표 | 값 |
| --- | ---: |
| final_train_loss | 5.341958 |
| final_val_loss | 5.638393 |
| best_val_loss | 5.638393 |
| best_step | 915 |
| final_perplexity | 281.010876 |
| elapsed_sec | 19.324 |
| tokens_per_sec | 195553.071 |

## 4. Step별 지표

| epoch | step | tokens_seen | train_loss | val_loss | lr | grad_norm | elapsed_sec | tokens_per_sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 8.334116 | 8.333532 | 0.0004 |  | 0.000 | 0.000 |
| 1 | 183 | 749568 | 7.479642 | 7.496120 | 0.0004 | 0.308975 | 4.070 | 184152.156 |
| 2 | 366 | 1499136 | 6.096871 | 6.179417 | 0.0004 | 0.607411 | 7.731 | 193903.874 |
| 3 | 500 | 2048000 | 5.737448 | 5.899409 | 0.0004 | 0.680461 | 10.437 | 196224.950 |
| 3 | 549 | 2248704 | 5.675533 | 5.836553 | 0.0004 | 0.637005 | 11.569 | 194379.116 |
| 4 | 732 | 2998272 | 5.489716 | 5.705013 | 0.0004 | 0.709487 | 15.327 | 195626.260 |
| 5 | 915 | 3747840 | 5.341958 | 5.638393 | 0.0004 | 0.775044 | 19.165 | 195553.071 |

## 4-1. 그래프용 CSV

아래 CSV 블록만 파싱하면 loss curve와 throughput curve를 그릴 수 있다.

```csv
epoch,step,tokens_seen,train_loss,val_loss,lr,grad_norm,elapsed_sec,tokens_per_sec
0,0,0,8.33412,8.33353,0.0004,,0,0
1,183,749568,7.47964,7.49612,0.0004,0.308975,4.07037,184152
2,366,1499136,6.09687,6.17942,0.0004,0.607411,7.73134,193904
3,500,2048000,5.73745,5.89941,0.0004,0.680461,10.437,196225
3,549,2248704,5.67553,5.83655,0.0004,0.637005,11.5687,194379
4,732,2998272,5.48972,5.70501,0.0004,0.709487,15.3265,195626
5,915,3747840,5.34196,5.63839,0.0004,0.775044,19.1653,195553
```

## 5. 생성 샘플

prompt: `이 영화는`

```text
이 영화는 처음엔 너무 잘 안고 말은거라기로드라.
그레이드리카프닝, 배우들은 다정이 있는건지,...?
진짜정말좋음?그리언가락영화중에 제일 좋아할듯~
아.. ㅇㅇㅇㅇ
이런걸
다!!!
이걸작중훈훈, !!
평점에 속았어요!
영화는 아닌가? 이건 아니죠~!!
유.ㅡ
우러난다 ㅋㅋ
이거보고 왜 만든거의 이야기가볍게 만드네!!
영화가락해요
너무 재미있어요~^^
이영화라의 명성
```

## 6. Checkpoint

저장하지 않음

## 7. 해석

baseline 결과다. 이후 실험은 이 md의 `final_val_loss`, `best_val_loss`, `tokens_per_sec`, 생성 샘플을 기준으로 비교한다.
