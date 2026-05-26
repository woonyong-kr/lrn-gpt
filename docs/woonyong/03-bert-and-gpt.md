# BERT와 GPT의 차이

이 문서는 BERT가 무엇인지 설명하고, GPT와 구조와 학습 목표가 어떻게 다른지 정리한다. Transformer 수식 자체는 [01-transformer-gpt3.md](01-transformer-gpt3.md)에서 설명했으므로 여기서는 encoder-only와 decoder-only의 차이에 집중한다.

## BERT 한 문장 요약

BERT는 Transformer encoder만 사용해 문장의 양쪽 문맥을 동시에 보는 언어 이해 모델이다. 이름도 Bidirectional Encoder Representations from Transformers다.

GPT가 "이전 토큰을 보고 다음 토큰을 생성하는 모델"이라면, BERT는 "문장 전체를 양방향으로 보고 각 token이나 문장의 의미 표현을 만드는 모델"이다.

## BERT의 구조

BERT는 Transformer encoder stack으로 구성된다.

```text
input tokens
  -> token embedding + segment embedding + position embedding
  -> bidirectional self-attention encoder layers x N
  -> contextual hidden states
  -> task head
```

GPT와 달리 BERT의 self-attention은 causal mask를 사용하지 않는다. 한 위치가 왼쪽 문맥과 오른쪽 문맥을 모두 볼 수 있다.

```text
      key position
      0  1  2  3
q 0   O  O  O  O
u 1   O  O  O  O
e 2   O  O  O  O
r 3   O  O  O  O
y
position
```

이 구조 덕분에 BERT는 문장 분류, 문장쌍 관계 판단, 토큰 분류, 질의응답처럼 "입력 전체를 이해한 뒤 답을 고르는 문제"에 강하다.

## BERT의 입력 토큰

BERT는 특별 토큰을 사용한다.

| 토큰 | 역할 |
| --- | --- |
| `[CLS]` | 문장 전체를 대표하는 위치 |
| `[SEP]` | 문장 구분 |
| `[MASK]` | pretraining 때 일부 토큰을 가리는 위치 |
| `[PAD]` | 길이를 맞추기 위한 padding |

문장 하나를 분류할 때는 보통 이렇게 넣는다.

```text
[CLS] 이 영화는 정말 좋았다 [SEP]
```

문장 두 개의 관계를 볼 때는 이렇게 넣는다.

```text
[CLS] 문장 A [SEP] 문장 B [SEP]
```

원래 BERT는 token embedding, segment embedding, position embedding을 더한다.

```text
input embedding = token embedding + segment embedding + position embedding
```

Segment embedding은 문장 A와 문장 B를 구분하는 역할을 한다.

## BERT의 pretraining

BERT는 주로 두 가지 목표로 사전 학습되었다.

### Masked Language Modeling, MLM

입력 토큰 중 일부를 `[MASK]`로 가리고, 원래 토큰을 맞힌다.

```text
입력:  나는 오늘 [MASK] 먹었다
정답:  밥을
```

중요한 점은 BERT가 `[MASK]` 위치의 왼쪽과 오른쪽을 모두 볼 수 있다는 것이다.

```text
P(masked_token | left_context, right_context)
```

이 목표 덕분에 BERT는 양방향 문맥 표현을 배운다.

### Next Sentence Prediction, NSP

두 문장이 실제로 이어지는 문장인지 맞히는 목표다.

```text
[CLS] 문장 A [SEP] 문장 B [SEP]
label: IsNext 또는 NotNext
```

이 목표는 문장쌍 관계를 배우기 위해 도입되었다. 이후 RoBERTa 같은 후속 모델들은 NSP를 제거하거나 다른 학습 전략을 사용하기도 했다. 하지만 BERT의 원래 설계를 이해할 때는 MLM과 NSP를 함께 기억하면 된다.

## BERT fine-tuning

BERT는 task마다 큰 구조 변경 없이 작은 head를 붙여 fine-tuning할 수 있다.

문장 분류:

```text
[CLS] sentence [SEP]
  -> BERT
  -> h_CLS
  -> classifier
  -> label
```

토큰 분류:

```text
token hidden states
  -> token-wise classifier
  -> label per token
```

질의응답:

```text
[CLS] question [SEP] passage [SEP]
  -> BERT
  -> start position logits
  -> end position logits
```

BERT 논문에서 강조한 장점은 task-specific architecture를 크게 바꾸지 않고도 여러 NLP task에 적용할 수 있다는 점이다.

## GPT 구조

GPT는 Transformer decoder-only 모델이다. GPT의 self-attention은 causal mask를 사용한다.

```text
input tokens
  -> token embedding + position embedding
  -> masked self-attention decoder blocks x N
  -> hidden states
  -> LM head
  -> next-token logits
```

GPT의 학습 목표는 next-token prediction이다.

```text
P(x_t | x_1, ..., x_{t-1})
```

그래서 GPT는 자연어 생성에 강하다. 긴 답변, 코드, 요약, 번역, 대화처럼 token sequence를 만들어야 하는 문제에 잘 맞는다.

## BERT와 GPT의 핵심 차이

| 항목 | BERT | GPT |
| --- | --- | --- |
| Transformer 종류 | encoder-only | decoder-only |
| attention mask | 양방향, 미래도 볼 수 있음 | causal, 미래를 볼 수 없음 |
| pretraining 목표 | masked token 예측, 문장 관계 예측 | 다음 토큰 예측 |
| 대표 출력 | 문장/토큰 표현 | 다음 토큰 확률 |
| 강한 분야 | 분류, 검색, 문장쌍 판단, 토큰 태깅 | 생성, 대화, 코드, in-context learning |
| 대표 위치 | `[CLS]` hidden state | 마지막 token 또는 각 위치 hidden state |
| 생성 능력 | 기본 구조로는 약함 | 핵심 능력 |
| 입력 전체 이해 | 매우 강함 | 왼쪽에서 오른쪽으로 누적 이해 |

## 왜 BERT는 생성에 약한가

BERT는 양방향으로 문맥을 본다. 이건 이해에는 강하지만, 왼쪽에서 오른쪽으로 한 토큰씩 생성하는 일에는 맞지 않는다.

생성은 다음 조건을 만족해야 한다.

```text
현재까지 생성한 토큰만 보고 다음 토큰을 뽑는다.
```

BERT는 pretraining 때 `[MASK]`를 보고 빈칸을 맞히도록 학습되었다. 하지만 실제 생성에서는 `[MASK]`가 계속 주어지는 것이 아니고, 생성해야 할 위치 오른쪽 문맥도 없다. 그래서 BERT를 그대로 GPT처럼 쓰기는 어렵다.

반대로 GPT는 처음부터 다음 토큰을 예측하도록 학습되었기 때문에 생성에 자연스럽다.

## 왜 BERT는 분류에 강한가

분류에서는 입력 전체를 한 번에 보고 label을 정하면 된다.

```text
text -> label
```

BERT는 모든 위치가 서로를 양방향으로 볼 수 있으므로, 문장의 앞뒤 단서를 모두 반영한 표현을 만든다.

예를 들어 다음 문장을 보자.

```text
"재미없을 줄 알았는데 끝까지 몰입했다"
```

"재미없을"만 보면 부정처럼 보인다. 하지만 뒤쪽의 "몰입했다"를 보면 전체 감성은 긍정에 가깝다. BERT는 양쪽 문맥을 동시에 볼 수 있어 이런 뒤쪽 반전을 잘 반영한다.

GPT도 마지막 위치에서는 앞의 모든 토큰을 볼 수 있으므로 분류가 가능하다. 하지만 구조적으로 BERT의 `[CLS]` 표현은 입력 전체 이해와 분류에 더 직접적으로 맞춰져 있다.

## 같은 Transformer인데 왜 성격이 다른가

핵심은 "무엇을 보게 했는가"와 "무엇을 맞히게 했는가"다.

BERT:

```text
보는 것: 왼쪽 + 오른쪽 전체 문맥
맞히는 것: 가려진 토큰, 문장 관계, task label
결과: 이해 중심 표현
```

GPT:

```text
보는 것: 왼쪽 문맥
맞히는 것: 다음 토큰
결과: 생성 중심 모델
```

모델 구조 이름보다 attention mask와 objective가 더 중요하다.

## BERT와 GPT를 선택하는 감각

| 문제 | 더 자연스러운 선택 |
| --- | --- |
| 영화 리뷰 긍정/부정 분류 | BERT 또는 GPT classifier |
| 문장 유사도 판단 | BERT 계열 |
| 개체명 인식 | BERT 계열 |
| 긴 답변 생성 | GPT 계열 |
| 코드 작성 | GPT 계열 |
| prompt 예시를 보고 즉석에서 task 수행 | GPT 계열 |
| 문서 검색용 embedding | BERT 계열 또는 encoder 계열 |

실무에서는 둘을 섞기도 한다. 예를 들어 BERT 계열 encoder로 문서를 검색하고, GPT 계열 decoder로 답변을 생성할 수 있다.

## 이 레포와 연결

이 레포는 GPT 구현 과제다. 따라서 핵심은 causal self-attention과 next-token prediction이다.

하지만 `src/finetune.py`에서 감성 분류를 다루므로, BERT식 분류 감각도 도움이 된다.

| 레포 구현 | BERT와 비교 |
| --- | --- |
| GPT backbone | BERT encoder가 아니라 decoder-only |
| causal mask | BERT에는 없음 |
| sentiment classifier | BERT의 `[CLS]` classifier와 역할은 비슷 |
| 마지막 hidden state | GPT에서 문장 대표 벡터로 사용 가능 |

즉, 이 과제의 분류 모델은 "BERT 같은 encoder classifier"가 아니라 "GPT backbone 위에 classifier head를 붙인 모델"이다.

## 이해 점검 질문

1. BERT가 양방향 문맥을 볼 수 있는 이유는 무엇인가.
2. GPT가 미래 토큰을 보면 왜 학습 문제가 쉬워져서 망가지는가.
3. `[CLS]` hidden state는 왜 문장 분류에 쓰이는가.
4. BERT의 MLM과 GPT의 next-token prediction은 정답을 보는 방식이 어떻게 다른가.
5. 유해 콘텐츠 감지에는 왜 BERT 계열 encoder가 자주 쓰이는가.

## 참고

- Devlin et al., 2018, [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- Vaswani et al., 2017, [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Brown et al., 2020, [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

