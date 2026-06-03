# LLM 10x Aggregate Report

## Status

- planned conditions: `130`
- planned analysis rows: `390`
- planned physical runs: `354`
- completed runs: `0`
- screen-ready conditions (`n >= 3 exploratory`): `0`
- claim-ready conditions (`n >= 10`): `0`

Exploratory conditions can be screen-ready at 3 completed repeats, but final claims still need 10 repeats on selected finalists.

## Condition Summary

| phase | condition | stage | axis | value | n | screen-ready | claim-ready | median val bits/char | val IQR | overfit median | gap median | time h median | tok/s median | paired delta median | beats baseline |
| --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phase1_lr | LR0050 | exploratory | learning_rate | 5e-05 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase1_lr | LR0070 | exploratory | learning_rate | 7e-05 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase1_lr | LR0100 | exploratory | learning_rate | 0.0001 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase1_lr | LR0150 | exploratory | learning_rate | 0.00015 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase1_lr | LR0200 | exploratory | learning_rate | 0.0002 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase1_lr | LR0300 | exploratory | learning_rate | 0.0003 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase1_lr | LR0500 | exploratory | learning_rate | 0.0005 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase1_lr | LR0700 | exploratory | learning_rate | 0.0007 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase1_lr | LR1000 | exploratory | learning_rate | 0.001 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase1_lr | LR1500 | exploratory | learning_rate | 0.0015 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase1_lr | LR2000 | exploratory | learning_rate | 0.002 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase1_lr | LR3000 | exploratory | learning_rate | 0.003 | 0 | no | no |  |  |  |  |  |  |  |  |
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
| phase3_vocab | V04000 | exploratory | vocab_size | 4000 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase3_vocab | V06000 | exploratory | vocab_size | 6000 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase3_vocab | V08000 | exploratory | vocab_size | 8000 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase3_vocab | V10000 | exploratory | vocab_size | 10000 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase3_vocab | V12000 | exploratory | vocab_size | 12000 | 0 | no | no |  |  |  |  |  |  |  |  |
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
| phase5_capacity | M031 | exploratory | capacity | 31M baseline | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M044 | exploratory | capacity | 44M deeper | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M057 | exploratory | capacity | 57M deepest 512-width | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M066 | exploratory | capacity | 66M wider | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M098 | exploratory | capacity | 98M wide-deep | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M130 | exploratory | capacity | 130M widest-deep 768-width | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M110 | exploratory | capacity | 110M 1024-width | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M160 | exploratory | capacity | 160M 1024-width deep | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M210 | exploratory | capacity | 210M 1024-width deepest | 0 | no | no |  |  |  |  |  |  |  |  |
| phase5_capacity | M240 | exploratory | capacity | 240M 1280-width deep | 0 | no | no |  |  |  |  |  |  |  |  |
| phase6_context | CTX0256 | exploratory | context_length | 256 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase6_context | CTX0384 | exploratory | context_length | 384 | 0 | no | no |  |  |  |  |  |  |  |  |
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
| phase11_ffn_mult | FFN02 | exploratory | ffn_mult | 2 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase11_ffn_mult | FFN03 | exploratory | ffn_mult | 3 | 0 | no | no |  |  |  |  |  |  |  |  |
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
