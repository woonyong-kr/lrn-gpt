# 지시 미세튜닝과 분류 미세튜닝

이 문서는 fine-tuning을 "모델의 weight를 실제로 바꾸는 학습"으로 보고 정리한다. zero-shot, few-shot prompting처럼 prompt만 바꾸는 방식은 [05-zero-shot-few-shot.md](05-zero-shot-few-shot.md)에서 따로 다룬다.

## 사전 학습과 미세 튜닝의 차이

LLM 학습은 크게 "범용 언어 능력을 만드는 단계"와 "특정 목적에 맞게 조정하는 단계"로 나눌 수 있다.

첫 번째는 레이블 없는 대규모 텍스트 말뭉치로 하는 사전 학습이다. GPT 계열 모델은 책, 웹 문서, 코드, 대화문 같은 거대한 텍스트를 보고 다음 토큰을 맞히도록 학습한다.

```text
입력: 오늘 날씨가 정말
정답: 좋다
```

여기서는 사람이 문장마다 "긍정", "부정", "정답", "오답" 같은 레이블을 붙이지 않아도 된다. 텍스트 자체가 학습 신호가 된다. 앞부분을 입력으로 쓰고 바로 다음 토큰을 정답으로 삼을 수 있기 때문이다.

GPT의 사전 학습 목표는 다음처럼 쓸 수 있다.

```text
P(다음 토큰 | 이전 토큰들)
```

이 과정에서 모델은 문법, 단어의 의미, 문체, 상식, 코드 패턴, 번역 패턴, 질의응답 형식, 추론처럼 보이는 텍스트 패턴을 넓게 배운다. 다만 사전 학습만 끝난 모델은 사용자의 지시를 항상 친절하게 따르거나, 특정 서비스 정책대로 분류하거나, 정해진 출력 형식을 안정적으로 지키도록 직접 훈련된 상태는 아니다.

두 번째는 레이블 또는 정답이 있는 데이터로 하는 미세 튜닝이다. 이미 사전 학습된 모델을 가져와 특정 목적에 맞게 추가 학습한다.

감성 분류라면 데이터는 다음처럼 생긴다.

```json
{"text": "이 영화 정말 좋았다", "label": "긍정"}
{"text": "시간이 아까웠다", "label": "부정"}
```

이때 모델은 다음 토큰을 자유롭게 예측하는 대신, 입력 문장을 보고 정해진 label을 맞히도록 학습한다.

```text
P(label | 입력 문장)
```

지시 미세튜닝이라면 데이터는 다음처럼 생긴다.

```json
{
  "instruction": "다음 문장을 요약해라.",
  "input": "Transformer는 self-attention을 사용한다...",
  "output": "Transformer는 self-attention 기반 모델이다."
}
```

이 경우 모델은 label 하나를 고르는 것이 아니라, 사용자의 지시를 읽고 어떤 형식으로 답해야 하는지를 배운다.

| 구분 | 사전 학습 | 미세 튜닝 |
| --- | --- | --- |
| 데이터 | 레이블 없는 대규모 텍스트 말뭉치 | 레이블 또는 정답이 있는 task 데이터 |
| 학습 신호 | 다음 토큰 자체가 정답 | 사람이 붙인 label, 정답 답변, 선호 데이터 |
| 대표 목표 | `P(next token | previous tokens)` | `P(label | input)` 또는 `P(answer | instruction, input)` |
| 목적 | 범용 언어 능력 획득 | 특정 작업이나 행동 방식에 맞게 조정 |
| 데이터 규모 | 매우 큼 | 상대적으로 작음 |
| 예시 | next-token prediction | 감성 분류, 유해 콘텐츠 감지, 지시 따르기 |
| 결과 | base language model | instruction model, classifier, domain model |

비유하면 사전 학습은 세상의 책을 많이 읽으며 언어 감각을 익히는 과정이고, 미세 튜닝은 특정 시험 유형이나 업무 방식에 맞춰 따로 훈련받는 과정이다.

```text
대규모 텍스트 사전 학습
  -> 기본 언어 능력 획득
  -> 지시 미세튜닝
  -> 사용자 지시를 더 잘 따름
  -> 필요하면 분류/도메인 미세튜닝
  -> 특정 업무에 최적화
```

## Fine-tuning의 기본 의미

Pretraining은 범용 능력을 만드는 단계다. GPT라면 대규모 텍스트에서 다음 토큰을 맞히고, BERT라면 마스킹된 토큰이나 문장 관계를 맞히며 언어의 일반 패턴을 배운다.

Fine-tuning은 이미 배운 모델을 특정 목적에 맞게 추가 학습하는 단계다.

```text
pretrained model
  -> task-specific data
  -> additional gradient updates
  -> specialized model
```

핵심은 gradient update가 일어난다는 점이다. prompt에 예시를 넣는 것만으로는 fine-tuning이 아니다.

## 두 갈래

Fine-tuning은 크게 두 질문으로 나눌 수 있다.

1. 모델이 답변 문장을 잘 생성하게 만들 것인가.
2. 모델이 입력을 보고 정해진 label을 잘 고르게 만들 것인가.

첫 번째가 지시 미세튜닝이고, 두 번째가 분류 미세튜닝이다.

| 구분 | 지시 미세튜닝 | 분류 미세튜닝 |
| --- | --- | --- |
| 목표 | 사용자의 지시를 따르는 답변 생성 | 입력을 label로 분류 |
| 출력 | 자연어 응답, 코드, 요약, 설명 | class logits, probability |
| 학습 데이터 | instruction, input, output | text, label |
| 대표 loss | next-token cross entropy | label cross entropy 또는 binary cross entropy |
| 평가 | 도움됨, 정확성, 형식 준수, 안전성 | accuracy, F1, AUROC, precision, recall |
| 모델 head | language modeling head 유지 | classification head 추가 |
| 예시 | "다음 문장을 요약해라" -> 요약문 | "이 리뷰는 긍정인가 부정인가" -> 긍정 |

## 지시 미세튜닝

지시 미세튜닝은 모델이 "사용자가 원하는 행동 형식"을 배우게 만드는 학습이다.

Base GPT는 다음 토큰 예측을 잘한다. 하지만 사용자의 질문에 친절하게 답하고, 지시를 지키고, 불필요한 내용을 줄이고, 안전 기준을 따르는 것은 별도의 행동 양식이다. 지시 미세튜닝은 이 행동 양식을 데이터로 가르친다.

### 데이터 형태

가장 단순한 supervised instruction tuning 데이터는 다음처럼 생긴다.

```json
{
  "instruction": "다음 문장을 한 문장으로 요약해라.",
  "input": "Transformer는 self-attention을 사용해 sequence를 병렬로 처리한다...",
  "output": "Transformer는 self-attention으로 문맥을 병렬 처리하는 모델이다."
}
```

채팅 모델이라면 message 형식으로 저장할 수 있다.

```json
{
  "messages": [
    {"role": "user", "content": "BERT와 GPT의 차이를 설명해줘."},
    {"role": "assistant", "content": "BERT는 encoder-only이고 GPT는 decoder-only다..."}
  ]
}
```

학습할 때는 prompt 부분과 answer 부분을 하나의 token sequence로 만들고, 보통 assistant answer 구간의 next-token loss를 최소화한다.

```text
prompt tokens + answer tokens
                 ^^^^^^^^^^^^^
                 loss를 주로 계산하는 구간
```

prompt 구간에도 loss를 줄 수 있지만, 실무에서는 모델이 사용자의 입력을 "그대로 재생"하도록 학습되는 것을 피하려고 answer 구간에만 loss를 주는 경우가 많다.

### 지시 미세튜닝의 종류

| 종류 | 설명 | 목적 |
| --- | --- | --- |
| Supervised fine-tuning, SFT | 사람이 작성한 좋은 답변을 정답으로 학습 | 기본 지시 수행 능력 형성 |
| Multi-task instruction tuning | 요약, 번역, 질의응답, 분류 등 여러 task를 instruction 형식으로 섞어 학습 | 다양한 지시 일반화 |
| Dialogue fine-tuning | user/assistant 대화 형식으로 학습 | 대화 흐름, 맥락 유지 |
| Safety fine-tuning | 안전 정책에 맞는 거절, 경고, 대안 답변을 학습 | 유해 응답 감소 |
| RLHF | 사람이 선호한 답변 순위를 reward model로 학습하고 강화학습 적용 | 사람 선호와 정렬 |
| Preference tuning | 선택된 답변과 거절된 답변 쌍으로 선호를 직접 학습 | RLHF보다 단순한 정렬 학습 |

InstructGPT 논문은 SFT 이후 사람의 선호 순위를 수집하고, reward model과 RLHF를 사용해 사용자 의도에 더 잘 맞는 답변을 만들었다. 핵심 메시지는 "모델을 크게 만드는 것만으로는 지시를 잘 따르는 모델이 되지 않는다"는 점이다.

### 지시 미세튜닝의 loss

GPT 계열 instruction tuning은 여전히 language modeling이다. 출력이 label 하나가 아니라 token sequence이기 때문이다.

```text
loss = - sum_t log P(answer_t | prompt, answer_<t)
```

즉, 답변의 첫 토큰은 prompt만 보고 예측하고, 두 번째 토큰은 prompt와 답변 첫 토큰을 보고 예측한다.

### 지시 미세튜닝이 배우는 것

지시 미세튜닝은 단순히 지식량을 늘리는 과정이 아니다. 주로 다음 행동을 배운다.

- 질문에 맞는 답변 형식 선택
- 불확실할 때 단정하지 않기
- 요구한 언어와 말투 지키기
- 단계별 설명 제공
- 코드, 표, 요약, 비교 등 형식 맞추기
- 위험한 요청을 안전하게 처리하기

그래서 지시 미세튜닝은 "지식 주입"보다 "행동 정렬"에 가깝다.

## 분류 미세튜닝

분류 미세튜닝은 입력을 보고 정해진 label을 예측하게 만드는 학습이다.

이 레포의 `src/finetune.py`는 GPT backbone 위에 감성 분류용 classifier를 붙이는 구조다.

```text
review text
  -> tokenizer
  -> GPT backbone
  -> representative hidden vector
  -> linear classifier
  -> positive/negative logits
```

### 데이터 형태

NSMC 감성 분류라면 데이터는 다음처럼 단순하다.

```json
{"text": "이 영화 정말 좋았다", "label": 1}
{"text": "시간이 아까웠다", "label": 0}
```

모델은 문장을 보고 label을 맞힌다.

### 분류 head

GPT는 원래 각 위치에서 다음 토큰 logits를 만든다.

```text
hidden states: (B, T, d_model)
LM logits:     (B, T, vocab_size)
```

분류 미세튜닝에서는 vocabulary logits가 아니라 class logits가 필요하다.

```text
class logits: (B, num_labels)
```

그래서 hidden state 중 하나를 문장 대표 벡터로 뽑고, linear classifier를 붙인다.

```text
h = hidden_states[:, -1, :]
logits = h W + b
```

Decoder-only GPT에서는 마지막 토큰 hidden state를 대표 벡터로 쓰는 경우가 많다. causal mask 때문에 마지막 위치는 앞의 모든 토큰을 볼 수 있기 때문이다.

BERT에서는 보통 `[CLS]` 위치 hidden state를 대표 벡터로 사용한다. BERT는 bidirectional encoder라 `[CLS]`가 문장 전체를 양방향으로 볼 수 있다.

### 분류 미세튜닝의 종류

| 종류 | 출력 | 예시 |
| --- | --- | --- |
| Binary classification | label 0 또는 1 | 긍정/부정, 스팸/정상 |
| Multi-class classification | 여러 class 중 하나 | 뉴스 카테고리, 의도 분류 |
| Multi-label classification | 여러 label이 동시에 참일 수 있음 | 유해성: 욕설, 혐오, 위협 동시 감지 |
| Token classification | 각 token마다 label | 개체명 인식, 품사 태깅 |
| Pair classification | 두 문장 관계 label | 자연어 추론, 의미 유사도 |
| Regression | 연속값 | 별점 예측, 독성 점수 |

### 분류 loss

class가 하나만 정답인 multi-class 문제는 cross entropy를 사용한다.

```text
loss = -log softmax(logits)[true_label]
```

여러 label이 동시에 정답일 수 있는 multi-label 문제는 label별 sigmoid와 binary cross entropy를 사용한다.

```text
p_c = sigmoid(logit_c)
loss = - sum_c [y_c log p_c + (1 - y_c) log(1 - p_c)]
```

유해 콘텐츠 감지는 보통 multi-label classification으로 설계되는 일이 많다. 한 문장이 욕설이면서 동시에 위협일 수도 있기 때문이다.

## 지시 미세튜닝과 분류 미세튜닝의 가장 깊은 차이

두 방식의 차이는 "정답의 모양"에서 나온다.

지시 미세튜닝의 정답은 문장이다. 답변은 길이가 가변적이고, 같은 질문에도 좋은 답변이 여러 개 있을 수 있다. 그래서 token sequence probability를 학습한다.

분류 미세튜닝의 정답은 label이다. 정답 공간이 작고 고정되어 있다. 그래서 class probability를 학습한다.

```text
지시 미세튜닝:
P(answer tokens | instruction, input)

분류 미세튜닝:
P(label | input text)
```

둘 다 pretrained model을 사용하지만, 모델에게 요구하는 행동은 매우 다르다.

## 전체 미세튜닝과 효율적 미세튜닝

Fine-tuning은 모든 parameter를 업데이트할 수도 있고, 일부 parameter만 업데이트할 수도 있다.

| 방식 | 설명 | 장점 | 단점 |
| --- | --- | --- | --- |
| Full fine-tuning | 모든 weight 업데이트 | 성능 잠재력 큼 | 비용과 저장 공간 큼 |
| Frozen backbone | backbone 고정, head만 학습 | 빠르고 안정적 | 표현 적응 한계 |
| Layer-wise tuning | 일부 layer만 업데이트 | 비용과 성능 절충 | 어떤 layer를 열지 선택 필요 |
| Adapter | 작은 모듈을 layer 사이에 추가 | task별 저장 비용 작음 | 구조 변경 필요 |
| LoRA | 큰 weight는 고정하고 low-rank update만 학습 | LLM fine-tuning에서 효율적 | rank, target module 선택 필요 |
| Prompt tuning | 연속 prompt vector만 학습 | parameter 효율 높음 | 작은 모델에서는 약할 수 있음 |

이 레포의 과제 수준에서는 full fine-tuning 또는 classifier head 중심 fine-tuning을 먼저 이해하는 것이 좋다. LoRA나 adapter는 나중에 큰 모델을 다룰 때 비용 문제를 해결하기 위해 배우면 된다.

## 언제 무엇을 써야 하는가

| 상황 | 추천 |
| --- | --- |
| 특정 label을 안정적으로 예측해야 함 | 분류 미세튜닝 |
| 확률 threshold와 운영 지표가 중요함 | 분류 미세튜닝 |
| 자연어 답변 형식이 중요함 | 지시 미세튜닝 |
| 같은 모델이 여러 종류의 요청을 처리해야 함 | 지시 미세튜닝 |
| 데이터가 적고 빠르게 실험해야 함 | zero-shot/few-shot prompting 먼저 시도 |
| 규칙 기반으로 충분히 처리 가능함 | fine-tuning보다 규칙 또는 작은 classifier |

## 흔한 오해

### "Fine-tuning하면 지식이 완벽히 추가된다"

Fine-tuning은 특정 데이터 분포에 맞게 weight를 조정한다. 작은 데이터로 새로운 사실을 주입하면 과적합하거나 기존 능력이 손상될 수 있다. 지식 추가가 목적이면 retrieval, 데이터베이스, 검색 결합이 더 적합할 때가 많다.

### "분류도 instruction tuning으로 하면 된다"

가능은 하다. 예를 들어 "이 문장이 긍정이면 긍정, 부정이면 부정이라고 답해라"처럼 만들 수 있다. 하지만 운영에서 정확한 확률, threshold, calibration, F1이 필요하면 classification head를 붙인 분류 모델이 관리하기 쉽다.

### "지시 미세튜닝은 RLHF와 같다"

같지 않다. SFT는 정답 답변을 따라 쓰는 supervised learning이다. RLHF는 여러 답변의 선호 순위를 이용해 reward model을 만들고, 그 reward를 높이도록 모델을 추가 최적화한다.

## 이 레포와 연결

`src/finetune.py`의 `GPTForSequenceClassification`은 지시 미세튜닝이 아니라 분류 미세튜닝이다.

구현 관점에서 생각할 흐름은 다음과 같다.

1. NSMC 리뷰를 tokenizer로 token ID로 바꾼다.
2. max_length에 맞게 padding 또는 truncation한다.
3. GPT backbone으로 hidden states를 만든다.
4. 마지막 유효 token hidden state 또는 마지막 위치 hidden state를 문장 대표 벡터로 사용한다.
5. classifier가 긍정/부정 logits를 낸다.
6. cross entropy loss로 학습한다.

지시 미세튜닝을 구현하려면 `text,label` 데이터가 아니라 `prompt,completion` 데이터가 필요하고, classifier head 대신 LM head를 유지해야 한다.

## 이해 점검 질문

1. 지시 미세튜닝의 정답은 왜 label이 아니라 token sequence인가.
2. 분류 미세튜닝에서 GPT의 마지막 token hidden state를 대표 벡터로 쓸 수 있는 이유는 무엇인가.
3. multi-class와 multi-label은 loss가 왜 달라지는가.
4. RLHF는 SFT와 무엇이 다른가.
5. prompt에 예시를 넣는 few-shot과 fine-tuning의 결정적 차이는 무엇인가.

## 참고

- Ouyang et al., 2022, [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- Brown et al., 2020, [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
