# woonyong LLM 학습 문서

이 폴더는 mini GPT 과제를 공부하면서 개념을 잃어버리지 않도록 만든 학습 노트다. 핵심 수식은 한 번만 깊게 설명하고, 나머지 문서는 그 개념을 응용하는 방식으로 연결했다.

## 읽는 순서

| 순서 | 문서 | 질문 |
| --- | --- | --- |
| 0 | [00-embedding-and-word2vec.md](00-embedding-and-word2vec.md) | 텍스트, 오디오, 비디오는 어떻게 벡터가 되는가 |
| 1 | [01-transformer-gpt3.md](01-transformer-gpt3.md) | Transformer와 GPT-3는 어떤 구조로 다음 토큰을 예측하는가 |
| 2 | [02-fine-tuning.md](02-fine-tuning.md) | 지시 미세튜닝과 분류 미세튜닝은 무엇이 다른가 |
| 3 | [03-bert-and-gpt.md](03-bert-and-gpt.md) | BERT는 무엇이고 GPT와 어떤 점이 다른가 |
| 4 | [04-bert-harmful-content-detection.md](04-bert-harmful-content-detection.md) | BERT로 유해 콘텐츠 감지는 어떤 원리로 동작하는가 |
| 5 | [05-zero-shot-few-shot.md](05-zero-shot-few-shot.md) | 제로샷과 퓨샷 학습은 무엇이 다른가 |
| 6 | [06-byte-pair-encoding.md](06-byte-pair-encoding.md) | BPE는 언제 세고 언제 병합하는가 |
| 7 | [07-project-completion-roadmap.md](07-project-completion-roadmap.md) | 프로젝트를 완성하려면 무엇을 구현해야 하는가 |

## 중복을 줄이는 기준

- Transformer의 수식, causal mask, GPT 계열 구조는 `01-transformer-gpt3.md`에서만 자세히 설명한다.
- 벡터, 임베딩, word2vec, 멀티모달 embedding, token/position embedding의 기본 직관은 `00-embedding-and-word2vec.md`에서 설명한다.
- parameter count, hidden layer 수, `d_model`, GPT-3 공개 구조의 관계는 `01-transformer-gpt3.md`에서 설명한다.
- fine-tuning의 목적 함수와 학습 데이터 형태는 `02-fine-tuning.md`에서 정리한다.
- BERT의 encoder-only 구조와 GPT와의 차이는 `03-bert-and-gpt.md`에서 설명한다.
- 유해 콘텐츠 감지는 BERT 구조를 다시 설명하지 않고, 분류 파이프라인과 운영상 주의점에 집중한다.
- 제로샷과 퓨샷은 모델 구조보다 "학습 여부와 예시 제공 방식"에 초점을 둔다.
- BPE tokenizer의 학습 단계, merge rule, vocab size, `<|unk|>` 차이는 `06-byte-pair-encoding.md`에서 정리한다.
- 구현 순서, 테스트 현황, 나중에 추가될 메모의 병합 위치는 `07-project-completion-roadmap.md`에서 관리한다.

## 공부할 때 붙잡을 큰 그림

언어 모델을 볼 때는 항상 네 질문으로 분해하면 길을 잃지 않는다.

1. 입력은 어떤 토큰 시퀀스인가.
2. 각 토큰은 어떤 벡터가 되는가.
3. 각 위치는 어떤 위치를 볼 수 있는가.
4. 최종 출력은 다음 토큰인가, 클래스인가, 토큰별 태그인가.
5. 학습 가능한 파라미터는 어느 행렬과 table에 들어 있는가.

Transformer, GPT, BERT, fine-tuning, zero-shot, few-shot은 서로 다른 이름처럼 보이지만 결국 이 네 질문의 답이 조금씩 다르다. 호기심이 많을수록 용어를 외우기보다 "이 모델은 무엇을 보게 허락받았고, 무엇을 맞히도록 벌점을 받는가"를 따라가면 된다.

## 참고 논문

- Vaswani et al., 2017, [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Brown et al., 2020, [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- Devlin et al., 2018, [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- Ouyang et al., 2022, [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- Mikolov et al., 2013, [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
