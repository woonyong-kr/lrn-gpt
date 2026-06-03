# LLM 10x 집계 보고서

## 현재 상태

- 계획된 조건 수: `130`
- 계획된 분석 행 수: `390`
- 계획된 실제 실행 수: `354`
- 완료된 실행 수: `63`
- 완료된 실제 실행 수: `66`
- 화면 검토 가능 조건(`탐색 조건 n >= 3`): `20`
- 주장 근거 가능 조건(`n >= 10`): `0`
- 전체 실행 결과 원장: `/Users/woonyong/workspace/Krafton-Jungle/SW_AI-W13-gpt/docs/llm_10x/all_run_results.jsonl`
- 원장 행 수: `66`
- 원장 누락 감사: `PASS`
- result.json 읽기 실패: `0`
- 중복 완료 run_number: `0`
- matrix 밖 완료 결과: `0`
- matrix/result 불일치: `0`

탐색 조건은 3회 반복이 끝나면 화면 검토는 가능하지만, 최종 주장은 선별된 후보에 대해 10회 반복이 필요합니다.

원장 누락 감사가 `PASS`가 아니면 완료된 개별 결과가 집계 원장에 빠졌거나, 읽을 수 없는 결과 파일/중복/계획표 불일치가 있다는 뜻입니다.

## 조건별 요약

| phase | 조건 | 단계 | 축 | 값 | n | 화면 검토 가능 | 주장 가능 | 검증 bits/char 중앙값 | 검증 IQR | 과적합 중앙값 | gap 중앙값 | 시간 h 중앙값 | warm tok/s 중앙값 | paired delta 중앙값 | 기준선 승률 |
| --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phase1_lr | LR0050 | exploratory | learning_rate | 5e-05 | 3 | yes | no | 4.23066 | 0.026017 | 0 | -0.0667027 | 0.0900198 | 2047.55 | 0.408168 | 0 |
| phase1_lr | LR0070 | exploratory | learning_rate | 7e-05 | 3 | yes | no | 4.07058 | 0.0471174 | 0 | -0.157885 | 0.0903226 | 2040.69 | 0.248087 | 0 |
| phase1_lr | LR0100 | exploratory | learning_rate | 0.0001 | 3 | yes | no | 3.97997 | 0.0489901 | 0 | -0.222133 | 0.091119 | 2022.85 | 0.157471 | 0 |
| phase1_lr | LR0150 | exploratory | learning_rate | 0.00015 | 3 | yes | no | 3.8874 | 0.0582243 | 0 | -0.288826 | 0.102418 | 1799.68 | 0.0649084 | 0 |
| phase1_lr | LR0200 | exploratory | learning_rate | 0.0002 | 3 | yes | no | 3.84309 | 0.0511927 | 0 | -0.325747 | 0.0899853 | 2048.33 | 0.0205978 | 0 |
| phase1_lr | LR0300 | exploratory | learning_rate | 0.0003 | 3 | yes | no | 3.8225 | 0.0667094 | 0 | -0.326963 | 0.0793457 | 2323 |  |  |
| phase1_lr | LR0500 | exploratory | learning_rate | 0.0005 | 3 | yes | no | 3.93704 | 0.0546196 | 0 | -0.254186 | 0.0910264 | 2024.91 | 0.0531401 | 0 |
| phase1_lr | LR0700 | exploratory | learning_rate | 0.0007 | 3 | yes | no | 3.9932 | 0.0796012 | 0 | -0.209725 | 0.0791643 | 2328.32 | 0.0836382 | 0 |
| phase1_lr | LR1000 | exploratory | learning_rate | 0.001 | 3 | yes | no | 4.04166 | 0.128937 | 0 | -0.1822 | 0.0803922 | 2292.76 | 0.16339 | 0 |
| phase1_lr | LR1500 | exploratory | learning_rate | 0.0015 | 3 | yes | no | 4.03809 | 0.112721 | 0 | -0.195723 | 0.101713 | 1812.15 | 0.15287 | 0 |
| phase1_lr | LR2000 | exploratory | learning_rate | 0.002 | 3 | yes | no | 4.22121 | 0.156875 | 0 | -0.0590315 | 0.107182 | 1719.69 | 0.310081 | 0 |
| phase1_lr | LR3000 | exploratory | learning_rate | 0.003 | 3 | yes | no | 4.27561 | 0.0409851 | 0 | -0.0301692 | 0.0782318 | 2356.07 | 0.4165 | 0 |
| phase2_epoch | E0025 | exploratory | epochs | 25 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase2_epoch | E0050 | exploratory | epochs | 50 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase2_epoch | E0075 | exploratory | epochs | 75 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase2_epoch | E0100 | exploratory | epochs | 100 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase2_epoch | E0150 | exploratory | epochs | 150 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase2_epoch | E0200 | exploratory | epochs | 200 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase2_epoch | E0300 | exploratory | epochs | 300 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase2_epoch | E0400 | exploratory | epochs | 400 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase2_epoch | E0600 | exploratory | epochs | 600 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase2_epoch | E0800 | exploratory | epochs | 800 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase2_epoch | E1000 | exploratory | epochs | 1000 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase2_epoch | E1200 | exploratory | epochs | 1200 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase2_epoch | E1500 | exploratory | epochs | 1500 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase3_vocab | V04000 | exploratory | vocab_size | 4000 | 1 | no | no | 4.20008 | 0 | 0 | -0.221544 | 0.0968913 | 2242.88 | 0.378235 | 0 |
| phase3_vocab | V06000 | exploratory | vocab_size | 6000 | 3 | yes | no | 4.07186 | 0.0476026 | 0 | -0.289637 | 0.103867 | 1944.36 | 0.251785 | 0 |
| phase3_vocab | V08000 | exploratory | vocab_size | 8000 | 3 | yes | no | 3.92661 | 0.0509132 | 0 | -0.270098 | 0.0793286 | 2431.07 | 0.0800639 | 0 |
| phase3_vocab | V10000 | exploratory | vocab_size | 10000 | 3 | yes | no | 3.86029 | 0.0392103 | 0 | -0.339356 | 0.0996168 | 1884.56 | 0.0613315 | 0.333333 |
| phase3_vocab | V12000 | exploratory | vocab_size | 12000 | 3 | yes | no | 3.82184 | 0.0676903 | 0 | -0.327853 | 0.113235 | 1627.76 |  |  |
| phase3_vocab | V16000 | exploratory | vocab_size | 16000 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase3_vocab | V20000 | exploratory | vocab_size | 20000 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase3_vocab | V24000 | exploratory | vocab_size | 24000 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase3_vocab | V32000 | exploratory | vocab_size | 32000 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase3_vocab | V40000 | exploratory | vocab_size | 40000 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase3_vocab | V48000 | exploratory | vocab_size | 48000 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase3_vocab | V56000 | exploratory | vocab_size | 56000 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase3_vocab | V64000 | exploratory | vocab_size | 64000 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase4_tokenizer_granularity | MF001 | exploratory | tokenizer_min_frequency | 1 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase4_tokenizer_granularity | MF002 | exploratory | tokenizer_min_frequency | 2 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase4_tokenizer_granularity | MF003 | exploratory | tokenizer_min_frequency | 3 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase4_tokenizer_granularity | MF004 | exploratory | tokenizer_min_frequency | 4 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase4_tokenizer_granularity | MF005 | exploratory | tokenizer_min_frequency | 5 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase4_tokenizer_granularity | MF008 | exploratory | tokenizer_min_frequency | 8 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase4_tokenizer_granularity | MF013 | exploratory | tokenizer_min_frequency | 13 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase4_tokenizer_granularity | MF021 | exploratory | tokenizer_min_frequency | 21 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase4_tokenizer_granularity | MF034 | exploratory | tokenizer_min_frequency | 34 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase4_tokenizer_granularity | MF055 | exploratory | tokenizer_min_frequency | 55 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M031 | exploratory | capacity | 31M baseline | 2 | no | no | 3.8678 | 0.0452167 | 0 | -0.294681 | 0.103501 | 1878.77 |  |  |
| phase5_capacity | M044 | exploratory | capacity | 44M deeper | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M057 | exploratory | capacity | 57M deepest 512-width | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M066 | exploratory | capacity | 66M wider | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M098 | exploratory | capacity | 98M wide-deep | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M130 | exploratory | capacity | 130M widest-deep 768-width | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M110 | exploratory | capacity | 110M 1024-width | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M160 | exploratory | capacity | 160M 1024-width deep | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M210 | exploratory | capacity | 210M 1024-width deepest | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M240 | exploratory | capacity | 240M 1280-width deep | 0 | no | no |  |  |  |  |  |  |  |  |
| phase6_context | CTX0256 | exploratory | context_length | 256 | 3 | yes | no | 3.6053 | 0.0226216 | 0 | -0.534429 | 0.0801919 | 2298.49 |  |  |
| phase6_context | CTX0384 | exploratory | context_length | 384 | 3 | yes | no | 3.71494 | 0.0343983 | 0 | -0.378739 | 0.0853584 | 2159.37 |  |  |
| phase6_context | CTX0512 | exploratory | context_length | 512 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase6_context | CTX0640 | exploratory | context_length | 640 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase6_context | CTX0768 | exploratory | context_length | 768 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase6_context | CTX1024 | exploratory | context_length | 1024 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase6_context | CTX1280 | exploratory | context_length | 1280 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase6_context | CTX1536 | exploratory | context_length | 1536 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase6_context | CTX2048 | exploratory | context_length | 2048 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase7_batch_size | BS001 | exploratory | batch_size | 1 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase7_batch_size | BS002 | exploratory | batch_size | 2 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase7_batch_size | BS004 | exploratory | batch_size | 4 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase7_batch_size | BS008 | exploratory | batch_size | 8 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase7_batch_size | BS016 | exploratory | batch_size | 16 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase8_dropout | DO000 | exploratory | drop_rate | 0 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase8_dropout | DO020 | exploratory | drop_rate | 0.02 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase8_dropout | DO030 | exploratory | drop_rate | 0.03 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase8_dropout | DO050 | exploratory | drop_rate | 0.05 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase8_dropout | DO080 | exploratory | drop_rate | 0.08 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase8_dropout | DO100 | exploratory | drop_rate | 0.1 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase8_dropout | DO120 | exploratory | drop_rate | 0.12 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase8_dropout | DO150 | exploratory | drop_rate | 0.15 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase8_dropout | DO200 | exploratory | drop_rate | 0.2 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase8_dropout | DO300 | exploratory | drop_rate | 0.3 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase8_dropout | DO400 | exploratory | drop_rate | 0.4 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase9_weight_decay | WD000 | exploratory | weight_decay | 0 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase9_weight_decay | WD005 | exploratory | weight_decay | 0.005 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase9_weight_decay | WD010 | exploratory | weight_decay | 0.01 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase9_weight_decay | WD020 | exploratory | weight_decay | 0.02 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase9_weight_decay | WD030 | exploratory | weight_decay | 0.03 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase9_weight_decay | WD050 | exploratory | weight_decay | 0.05 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase9_weight_decay | WD075 | exploratory | weight_decay | 0.075 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase9_weight_decay | WD100 | exploratory | weight_decay | 0.1 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase9_weight_decay | WD150 | exploratory | weight_decay | 0.15 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase9_weight_decay | WD200 | exploratory | weight_decay | 0.2 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase9_weight_decay | WD300 | exploratory | weight_decay | 0.3 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase9_weight_decay | WD500 | exploratory | weight_decay | 0.5 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase10_grad_clip | GC000 | exploratory | grad_clip | 0 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase10_grad_clip | GC025 | exploratory | grad_clip | 0.25 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase10_grad_clip | GC050 | exploratory | grad_clip | 0.5 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase10_grad_clip | GC100 | exploratory | grad_clip | 1 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase10_grad_clip | GC200 | exploratory | grad_clip | 2 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase10_grad_clip | GC500 | exploratory | grad_clip | 5 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase11_ffn_mult | FFN02 | exploratory | ffn_mult | 2 | 3 | yes | no | 3.86642 | 0.0862475 | 0 | -0.292341 | 0.0635804 | 2899.01 |  |  |
| phase11_ffn_mult | FFN03 | exploratory | ffn_mult | 3 | 3 | yes | no | 3.85629 | 0.0289667 | 0 | -0.295934 | 0.0619347 | 2976.04 |  |  |
| phase11_ffn_mult | FFN04 | exploratory | ffn_mult | 4 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase11_ffn_mult | FFN05 | exploratory | ffn_mult | 5 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase11_ffn_mult | FFN06 | exploratory | ffn_mult | 6 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase11_ffn_mult | FFN08 | exploratory | ffn_mult | 8 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase12_activation | ACT_GELU | exploratory | activation_name | gelu | 0 | no | no |  |  |  |  |  |  |  |  |
| phase12_activation | ACT_GELUEXACT | exploratory | activation_name | gelu_exact | 0 | no | no |  |  |  |  |  |  |  |  |
| phase12_activation | ACT_QUICKGELU | exploratory | activation_name | quick_gelu | 0 | no | no |  |  |  |  |  |  |  |  |
| phase12_activation | ACT_RELU | exploratory | activation_name | relu | 0 | no | no |  |  |  |  |  |  |  |  |
| phase12_activation | ACT_SILU | exploratory | activation_name | silu | 0 | no | no |  |  |  |  |  |  |  |  |
| phase12_activation | ACT_SWISH | exploratory | activation_name | swish | 0 | no | no |  |  |  |  |  |  |  |  |
| phase12_activation | ACT_MISH | exploratory | activation_name | mish | 0 | no | no |  |  |  |  |  |  |  |  |
| phase12_activation | ACT_SQUAREDRELU | exploratory | activation_name | squared_relu | 0 | no | no |  |  |  |  |  |  |  |  |
| phase12_activation | ACT_IDENTITY | exploratory | activation_name | identity | 0 | no | no |  |  |  |  |  |  |  |  |
| phase12_activation | ACT_SWIGLU | exploratory | activation_name | swiglu | 0 | no | no |  |  |  |  |  |  |  |  |
| phase12_activation | ACT_GEGLU | exploratory | activation_name | geglu | 0 | no | no |  |  |  |  |  |  |  |  |
| phase13_init_std | INIT005 | exploratory | init_std | 0.005 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase13_init_std | INIT010 | exploratory | init_std | 0.01 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase13_init_std | INIT015 | exploratory | init_std | 0.015 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase13_init_std | INIT020 | exploratory | init_std | 0.02 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase13_init_std | INIT030 | exploratory | init_std | 0.03 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase13_init_std | INIT040 | exploratory | init_std | 0.04 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase14_structure | STRUCT_BASE | exploratory | structure | baseline | 0 | no | no |  |  |  |  |  |  |  |  |
| phase14_structure | STRUCT_QKVB | exploratory | structure | qkv_bias=True | 0 | no | no |  |  |  |  |  |  |  |  |
| phase14_structure | STRUCT_UNTIED | exploratory | structure | tie_embeddings=False | 0 | no | no |  |  |  |  |  |  |  |  |
| phase14_structure | STRUCT_POSTLN | exploratory | structure | norm_first=False | 0 | no | no |  |  |  |  |  |  |  |  |
| phase14_structure | STRUCT_MANUALATT | exploratory | structure | attention_impl=manual | 0 | no | no |  |  |  |  |  |  |  |  |
| phase14_structure | STRUCT_QKVB_UNTIED | exploratory | structure | qkv_bias=True,tie_embeddings=False | 0 | no | no |  |  |  |  |  |  |  |  |
