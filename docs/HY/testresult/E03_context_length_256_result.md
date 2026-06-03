# E03 context_length 256

## 1. 실험 목적

Notion 실험 DB에 정의된 LLM 하이퍼파라미터 변경 효과를 baseline과 비교한다.

## 2. 실행 설정

| 항목 | 값 |
| --- | --- |
| 변경점 | context_length를 128에서 256으로 증가 |
| seed | 123 |
| vocab_size | 3000 |
| context_length | 256 |
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
| stride | 256 |
| amp | bf16 |
| device | NVIDIA GeForce RTX 5070 Ti |
| torch | 2.13.0.dev20260602+cu132 |
| parameter_count | 2978688 |
| train_chars | 1379486 |
| val_chars | 120560 |
| train_tokens | 805021 |
| val_tokens | 70386 |
| train_batches | 98 |
| val_batches | 8 |
| tokenizer_path | C:\Dev\Crafton-Jungle\04.AI\wk13_6_gpt\data\nsmc_bpe_vocab_3000.json |
| tokenizer_loaded | True |
| tokenizer_elapsed_sec | 0.004 |

## 3. 결과 요약

| 지표 | 값 |
| --- | ---: |
| final_train_loss | 5.287089 |
| final_val_loss | 5.447002 |
| best_val_loss | 5.447002 |
| best_step | 490 |
| final_perplexity | 232.061290 |
| elapsed_sec | 12.006 |
| tokens_per_sec | 340329.108 |

## 4. Step별 지표

| epoch | step | tokens_seen | train_loss | val_loss | lr | grad_norm | elapsed_sec | tokens_per_sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 8.043876 | 8.043230 | 0.0004 |  | 0.000 | 0.000 |
| 1 | 98 | 802816 | 7.292839 | 7.293815 | 0.0004 | 0.178364 | 2.596 | 309228.047 |
| 2 | 196 | 1605632 | 6.212404 | 6.252933 | 0.0004 | 0.456761 | 4.825 | 332799.826 |
| 3 | 294 | 2408448 | 5.645193 | 5.728631 | 0.0004 | 0.526149 | 7.126 | 337957.243 |
| 4 | 392 | 3211264 | 5.423755 | 5.542152 | 0.0004 | 0.445380 | 9.461 | 339424.123 |
| 5 | 490 | 4014080 | 5.287089 | 5.447002 | 0.0004 | 0.499240 | 11.795 | 340329.108 |

## 4-1. 그래프용 CSV

아래 CSV 블록만 파싱하면 loss curve와 throughput curve를 그릴 수 있다.

```csv
epoch,step,tokens_seen,train_loss,val_loss,lr,grad_norm,elapsed_sec,tokens_per_sec
0,0,0,8.04388,8.04323,0.0004,,0,0
1,98,802816,7.29284,7.29381,0.0004,0.178364,2.59619,309228
2,196,1605632,6.2124,6.25293,0.0004,0.456761,4.82462,332800
3,294,2408448,5.64519,5.72863,0.0004,0.526149,7.12649,337957
4,392,3211264,5.42376,5.54215,0.0004,0.44538,9.46092,339424
5,490,4014080,5.28709,5.447,0.0004,0.49924,11.7947,340329
```

## 5. 생성 샘플

prompt: `이 영화는`

```text
이 영화는 처음을 알겠다리게봤다.
그닥터리받과 연출력과 미치 않지만
별점
내가 영화!
영화의 자막히 미모가 이리언이에 너무 좋아할듯... 이러는 이 영화가 끝나는 드라마가..이걸 영화에서는 안간만에 다가고는 있는데 평점을 바라들
내 생긴하지만 재밌음
재미없다
어용! 아들로맨의 명성우성에 빠진 않아서 보겠어디어지는 않았다!!
내가 본다면..너무 재밋네
```

## 6. Checkpoint

저장하지 않음

## 7. 해석

baseline 결과다. 이후 실험은 이 md의 `final_val_loss`, `best_val_loss`, `tokens_per_sec`, 생성 샘플을 기준으로 비교한다.
