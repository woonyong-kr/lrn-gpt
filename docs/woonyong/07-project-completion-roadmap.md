# mini GPT 프로젝트 완성 로드맵

이 문서는 `SW_AI-W13-gpt` 프로젝트를 완전히 구현하기 위해 해야 할 일을 미리 정리한 작업 문서다. 나중에 공부 내용, 구현 메모, 실험 결과가 추가되면 이 문서를 기준으로 병합한다.

## 목표

최종 목표는 다음 세 가지다.

1. `src/`의 모든 `TODO`와 `NotImplementedError`를 제거한다.
2. `pytest tests/ -v` 전체 테스트를 통과한다.
3. `gpt-lab.ipynb`에서 BPE, 모델, 사전 학습, 감성 분류 미세 조정 흐름을 실행할 수 있게 만든다.

## 책 기준 학습 범위

필수 구현을 끝내려면 『밑바닥부터 만들면서 배우는 LLM』 기준으로 6장까지 보면 된다.

| 범위 | 필요도 | 프로젝트 연결 |
| --- | --- | --- |
| 2장 | 필수 | 텍스트 토큰화, BPE, dataset, embedding |
| 3장 | 필수 | self-attention, causal mask, multi-head attention |
| 4장 | 필수 | LayerNorm, GELU, FFN, TransformerBlock, GPTModel |
| 5장 | 필수 | loss, generation, checkpoint, pretraining loop |
| 6장 | 필수 | 감성 분류 dataset, classifier head, fine-tuning |
| 7장 | 선택 | 지시 미세튜닝, instruction following 확장 |
| 부록 D | 선택 | warmup, cosine decay, gradient clipping, weight decay |

현재 과제의 미세 튜닝은 지시 미세튜닝이 아니라 NSMC 감성 분류 미세 튜닝이다. 따라서 필수 완성 기준은 6장까지다.

## 구현 순서

### 1. BPE tokenizer

파일:

- `src/bpe.py`
- `tests/test_bpe.py`

구현할 것:

- [ ] special token ID 고정
- [ ] UTF-8 byte-level vocabulary 초기화
- [ ] corpus 기반 BPE merge rule 학습
- [ ] encode
- [ ] decode
- [ ] save
- [ ] load

완료 기준:

```bash
pytest tests/test_bpe.py -v
```

학습 문서:

- [06-byte-pair-encoding.md](06-byte-pair-encoding.md)
- [00-embedding-and-word2vec.md](00-embedding-and-word2vec.md)

### 2. Dataset과 InputEmbedding

파일:

- `src/dataset.py`
- `src/embeddings.py`
- `tests/test_dataset.py`

구현할 것:

- [ ] token ID sequence에서 input/target pair 만들기
- [ ] context length 기준 sliding window 구성
- [ ] DataLoader 생성
- [ ] token embedding
- [ ] position embedding
- [ ] embedding dropout

완료 기준:

```bash
pytest tests/test_dataset.py -v
```

핵심 shape:

```text
input_ids: (B, T)
target_ids: (B, T)
embedding output: (B, T, d_model)
```

### 3. Multi-head causal self-attention

파일:

- `src/attention.py`
- `tests/test_attention.py`

구현할 것:

- [ ] Q/K/V projection
- [ ] head split
- [ ] scaled dot-product attention
- [ ] causal mask
- [ ] attention dropout
- [ ] output projection
- [ ] attention weight 반환 옵션

완료 기준:

```bash
pytest tests/test_attention.py -v
```

학습 문서:

- [01-transformer-gpt3.md](01-transformer-gpt3.md)

### 4. GPTModel

파일:

- `src/model.py`
- `tests/test_model.py`

구현할 것:

- [ ] LayerNorm
- [ ] GELU
- [ ] FeedForward
- [ ] TransformerBlock
- [ ] GPTModel
- [ ] next-token cross entropy loss
- [ ] greedy generation

완료 기준:

```bash
pytest tests/test_model.py -v
```

핵심 shape:

```text
idx: (B, T)
hidden states: (B, T, d_model)
logits: (B, T, vocab_size)
loss: scalar
```

### 5. 사전 학습 유틸리티

파일:

- `src/train.py`
- `tests/test_train.py`

구현할 것:

- [ ] batch loss 계산
- [ ] loader 평균 loss 계산
- [ ] checkpoint 저장
- [ ] checkpoint 로드
- [ ] temperature sampling
- [ ] top-k sampling
- [ ] generate helper
- [ ] train loop
- [ ] loss plot

완료 기준:

```bash
pytest tests/test_train.py -v
```

선택 확장:

- [ ] learning rate warmup
- [ ] cosine decay
- [ ] gradient clipping
- [ ] weight decay

### 6. 감성 분류 미세 튜닝

파일:

- `src/finetune.py`
- `tests/test_finetune.py`

구현할 것:

- [ ] NSMC TSV 읽기
- [ ] train/validation/test split
- [ ] JSONL 저장 옵션
- [ ] sentiment dataset
- [ ] padding/truncation
- [ ] GPT backbone 기반 classifier
- [ ] train epoch
- [ ] evaluate

완료 기준:

```bash
pytest tests/test_finetune.py -v
```

학습 문서:

- [02-fine-tuning.md](02-fine-tuning.md)
- [03-bert-and-gpt.md](03-bert-and-gpt.md)
- [04-bert-harmful-content-detection.md](04-bert-harmful-content-detection.md)

### 7. 전체 통합

구현할 것:

- [ ] 전체 테스트 실행
- [ ] `gpt-lab.ipynb` smoke test 실행
- [ ] `REPORT.md` 구현 현황 업데이트
- [ ] README와 실제 구현 차이 확인
- [ ] 불필요한 실험 파일 정리

완료 기준:

```bash
pytest tests/ -v
```

## 통합 체크리스트

| 항목 | 상태 | 메모 |
| --- | --- | --- |
| BPE tokenizer | 대기 | 2장, byte-level BPE |
| Dataset/InputEmbedding | 대기 | sliding window와 position embedding |
| Attention | 대기 | causal mask 검증 중요 |
| GPTModel | 대기 | logits/loss shape 확인 |
| Train utils | 대기 | checkpoint와 generation |
| Sentiment fine-tuning | 대기 | classification head |
| 전체 테스트 | 대기 | `pytest tests/ -v` |
| 노트북 실행 | 대기 | Colab/로컬 smoke test |
| 보고서 작성 | 대기 | `REPORT.md` |

## 나중에 내용 추가 시 병합 규칙

새로운 공부 내용이나 구현 메모가 생기면 아래 기준으로 병합한다.

| 추가 내용 | 병합 위치 |
| --- | --- |
| BPE 원리, tokenizer 실험 | [06-byte-pair-encoding.md](06-byte-pair-encoding.md) |
| embedding, word2vec, vector 검색 | [00-embedding-and-word2vec.md](00-embedding-and-word2vec.md) |
| Transformer, attention, GPT-3 구조 | [01-transformer-gpt3.md](01-transformer-gpt3.md) |
| pretraining, fine-tuning 차이 | [02-fine-tuning.md](02-fine-tuning.md) |
| BERT와 GPT 비교 | [03-bert-and-gpt.md](03-bert-and-gpt.md) |
| 유해 콘텐츠 감지 | [04-bert-harmful-content-detection.md](04-bert-harmful-content-detection.md) |
| zero-shot, few-shot, prompting | [05-zero-shot-few-shot.md](05-zero-shot-few-shot.md) |
| 구현 순서, 테스트 현황, 남은 작업 | 이 문서 |

병합할 때는 같은 개념을 여러 문서에 반복해서 쓰지 않는다. 이미 설명한 문서가 있으면 링크를 걸고, 새 문서에는 해당 주제의 응용이나 구현상 주의점만 추가한다.

## 구현 중 기록할 것

각 단계가 끝날 때 아래 항목을 짧게 남긴다.

```text
날짜:
구현 파일:
통과한 테스트:
막혔던 점:
해결 방법:
다음 작업:
```

예시:

```text
날짜: 2026-05-27
구현 파일: src/attention.py
통과한 테스트: pytest tests/test_attention.py -v
막혔던 점: causal mask shape broadcasting
해결 방법: mask를 (1, 1, T, T)로 만들어 score에 적용
다음 작업: src/model.py TransformerBlock 구현
```

## 최종 완료 정의

프로젝트 완료는 다음 조건을 모두 만족할 때로 본다.

- [ ] 모든 `NotImplementedError` 제거
- [ ] 모든 단위 테스트 통과
- [ ] 작은 corpus로 BPE encode/decode 복원 확인
- [ ] 한 batch pretraining smoke test 성공
- [ ] generation이 token shape를 유지하며 동작
- [ ] sentiment classifier forward/loss/eval 동작
- [ ] `REPORT.md`에 구현 현황과 테스트 결과 기록

