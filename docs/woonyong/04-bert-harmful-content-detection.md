# BERT로 유해 콘텐츠를 감지하는 원리

이 문서는 BERT 구조를 다시 반복하지 않고, BERT를 분류 모델로 fine-tuning해 유해 콘텐츠를 감지하는 원리를 설명한다. BERT와 GPT의 구조 차이는 [03-bert-and-gpt.md](03-bert-and-gpt.md)를 먼저 보면 된다.

## 문제 정의

유해 콘텐츠 감지는 입력 텍스트가 특정 위험 범주에 해당하는지 판단하는 문제다.

```text
input text -> harmful category probability
```

예시는 다음과 같다.

| 범주 | 의미 |
| --- | --- |
| toxic | 공격적이거나 해로운 표현 |
| insult | 모욕 |
| threat | 위협 |
| hate | 특정 집단에 대한 혐오 |
| sexual | 노골적 성적 표현 |
| self-harm | 자해 관련 위험 표현 |
| spam | 원치 않는 홍보 또는 반복 게시 |

서비스마다 범주는 다르다. 중요한 점은 한 문장이 여러 범주에 동시에 해당할 수 있다는 것이다. 그래서 유해 콘텐츠 감지는 multi-label classification으로 설계되는 경우가 많다.

## 전체 파이프라인

```text
raw text
  -> normalization
  -> tokenizer
  -> BERT encoder
  -> [CLS] hidden vector
  -> classification head
  -> category probabilities
  -> threshold decision
  -> moderation action
```

각 단계의 역할은 분명하다.

| 단계 | 역할 |
| --- | --- |
| normalization | 깨진 문자, 반복 공백, URL, 멘션 등 정리 |
| tokenizer | 텍스트를 WordPiece token ID로 변환 |
| BERT encoder | 문맥을 반영한 token representation 생성 |
| `[CLS]` vector | 문장 전체 대표 표현 |
| classification head | 범주별 logit 계산 |
| threshold | 확률을 실제 조치로 변환 |
| moderation action | 허용, 경고, 숨김, 검토 요청 등 |

## 학습 데이터

학습 데이터는 텍스트와 label로 구성된다.

```json
{
  "text": "너 같은 사람은 여기 오지 마",
  "labels": {
    "toxic": 1,
    "insult": 1,
    "threat": 0,
    "hate": 0
  }
}
```

label이 하나만 있는 binary classification이면 단순하다.

```json
{"text": "정상 댓글입니다", "label": 0}
{"text": "모욕적 댓글입니다", "label": 1}
```

하지만 실제 moderation에서는 범주가 여러 개이고, 한 문장이 여러 범주에 걸칠 수 있다. 그래서 label vector를 사용한다.

```text
y = [toxic, insult, threat, hate]
y = [1, 1, 0, 0]
```

## BERT가 문맥을 보는 방식

유해 콘텐츠 감지는 단어 목록만으로 풀기 어렵다. 같은 단어도 문맥에 따라 의미가 달라진다.

```text
"죽어"                    -> 공격 또는 자해 유도일 수 있음
"그 캐릭터는 마지막에 죽어" -> 줄거리 설명일 수 있음
"죽어도 포기하지 마"       -> 격려 표현일 수 있음
```

BERT는 각 token이 문장 전체를 양방향으로 참고한다. 그래서 특정 단어 하나가 아니라 주변 문맥, 부정어, 대상, 인용 여부, 문장 목적을 함께 반영한 표현을 만든다.

```text
token hidden state = token itself + left context + right context
```

이 문맥 표현이 classifier로 들어가기 때문에 단순 키워드 필터보다 더 유연하다.

## Classification head

BERT의 출력은 각 token 위치의 hidden state다.

```text
hidden states shape: (B, T, d_model)
```

문장 전체를 분류할 때는 보통 `[CLS]` 위치의 hidden state를 사용한다.

```text
h = hidden_states[:, 0, :]
```

그 위에 linear layer를 붙인다.

```text
logits = h W + b
```

binary classification이면 logit 하나를 sigmoid로 바꾼다.

```text
p = sigmoid(logit)
```

multi-class classification이면 softmax를 쓴다.

```text
p = softmax(logits)
```

multi-label classification이면 범주마다 sigmoid를 따로 쓴다.

```text
p_c = sigmoid(logit_c)
```

multi-label에서 softmax를 쓰지 않는 이유는 범주가 서로 배타적이지 않기 때문이다. 한 문장이 toxic이면서 threat일 수 있다.

## Loss

Binary 또는 multi-label classification에서는 binary cross entropy를 쓴다.

```text
loss = - sum_c [y_c log p_c + (1 - y_c) log(1 - p_c)]
```

`y_c`는 범주 `c`가 정답이면 1, 아니면 0이다. 모델은 정답 범주의 확률을 높이고, 아닌 범주의 확률을 낮추도록 학습된다.

Multi-class classification에서는 cross entropy를 쓴다.

```text
loss = -log softmax(logits)[true_class]
```

유해 콘텐츠처럼 복수 label 가능성이 있는 문제에서는 binary cross entropy가 더 자연스럽다.

## Threshold가 중요한 이유

모델은 보통 확률을 낸다.

```text
toxic: 0.82
insult: 0.71
threat: 0.18
hate: 0.09
```

하지만 서비스는 확률이 아니라 결정을 내려야 한다.

```text
toxic >= 0.80 -> 숨김
toxic >= 0.60 -> 사람 검토
toxic < 0.60 -> 통과
```

Threshold를 낮추면 더 많이 잡지만 정상 글도 더 많이 막는다. Threshold를 높이면 정상 글을 덜 막지만 유해 글을 놓칠 수 있다.

| 기준 | 낮은 threshold | 높은 threshold |
| --- | --- | --- |
| recall | 높음 | 낮음 |
| precision | 낮을 수 있음 | 높음 |
| false positive | 증가 | 감소 |
| false negative | 감소 | 증가 |
| 사용자 경험 | 정상 글 차단 위험 | 유해 글 노출 위험 |

운영에서는 범주별로 threshold를 다르게 둔다. 예를 들어 threat나 self-harm은 recall을 높게 잡고 사람 검토를 붙이는 것이 안전할 수 있다.

## BERT가 실제로 배우는 단서

BERT classifier는 사람이 직접 규칙을 넣지 않아도 학습 데이터에서 단서를 배운다.

가능한 단서는 다음과 같다.

- 특정 모욕 표현
- 위협 동사와 대상
- 집단 지칭 표현과 부정적 서술
- 부정어와 완화 표현
- 인용이나 신고 문맥
- 반복, 과도한 강조, 비속어 변형
- 문장 전체의 감정 강도

하지만 모델이 "윤리"를 이해한다고 과장하면 안 된다. 모델은 label이 붙은 데이터 분포에서 통계적 패턴을 학습한다. 그래서 데이터가 편향되면 판단도 편향될 수 있다.

## 키워드 필터와의 차이

키워드 필터:

```text
금칙어 포함 -> 차단
```

BERT classifier:

```text
문장 전체 문맥 -> 확률 판단
```

키워드 필터는 빠르고 설명하기 쉽다. 하지만 우회 표현, 오탐, 문맥 구분에 약하다.

BERT는 문맥 판단에 강하지만, 학습 데이터와 threshold 관리가 필요하고, 완벽한 설명 가능성을 제공하지 않는다.

좋은 moderation 시스템은 보통 둘을 섞는다.

```text
명백한 규칙 위반 -> 규칙 기반 즉시 처리
애매한 문장 -> BERT classifier
고위험 또는 불확실 -> 사람 검토
```

## 편향과 공정성 문제

유해 콘텐츠 감지에서 가장 조심해야 할 부분은 편향이다.

예를 들어 학습 데이터에서 특정 정체성 단어가 유해 댓글에 자주 등장했다면, 모델은 그 단어 자체를 위험 신호로 과하게 배울 수 있다. 그러면 차별을 비판하는 문장이나 피해 경험을 말하는 문장까지 잘못 차단할 수 있다.

문제 사례:

- 정체성 단어가 들어간 정상 발언을 toxic으로 오탐
- 특정 방언이나 커뮤니티 표현을 공격적으로 오해
- 욕설을 인용해 신고하는 문장을 욕설 발화로 오해
- 풍자, 농담, 친한 사이의 표현을 실제 공격으로 오해
- 은어와 우회 표현을 놓침

그래서 유해 콘텐츠 classifier는 accuracy 하나만 보면 안 된다.

확인해야 할 지표:

- 범주별 precision, recall, F1
- false positive 예시
- false negative 예시
- identity term이 포함된 문장의 오탐률
- 언어, 방언, 집단별 성능 차이
- threshold별 운영 결과

## 운영에서의 의사결정

모델 점수는 최종 판결이 아니라 의사결정 재료다.

```text
score low    -> 통과
score medium -> 경고 또는 사람 검토
score high   -> 임시 숨김 또는 차단
```

서비스의 위험도에 따라 조치를 다르게 설계해야 한다.

| 상황 | 권장 접근 |
| --- | --- |
| 댓글이 잠깐 노출되어도 큰 위험이 낮음 | 높은 precision 중심 |
| 폭력 위협, 자해 위험 | 높은 recall + 사람 검토 |
| 법적 리스크가 큰 콘텐츠 | 보수적 threshold + 감사 로그 |
| 커뮤니티 표현이 다양한 서비스 | 도메인별 재학습과 오탐 분석 |

## GPT와 BERT의 moderation 역할 차이

BERT classifier는 입력 텍스트의 위험도를 점수화하는 데 적합하다.

GPT 계열 모델은 다음 역할에 더 자연스럽다.

- moderation 이유 설명
- 사용자에게 수정 제안
- 정책 문서 기반 판단 보조
- 사람이 볼 검토 요약 생성

하지만 GPT가 생성한 설명이 항상 정확한 것은 아니다. 최종 정책 판단은 classifier 점수, 규칙, 사람 검토, 감사 가능한 근거를 함께 사용해야 한다.

## 이 레포의 NSMC 분류와 연결

NSMC 감성 분류는 유해 콘텐츠 감지보다 단순하지만 구조는 비슷하다.

```text
text -> backbone -> representative vector -> classifier -> label
```

차이는 label의 의미와 운영 난이도다.

| 항목 | NSMC 감성 분류 | 유해 콘텐츠 감지 |
| --- | --- | --- |
| label | 긍정/부정 | toxic, insult, threat 등 |
| label 수 | 보통 binary | multi-label 가능 |
| 위험도 | 낮음 | 높음 |
| threshold | 단순 | 범주별 조정 필요 |
| 오탐 비용 | 실험 성능 저하 | 표현의 자유, 안전, 신뢰 문제 |

그래서 유해 콘텐츠 감지는 모델 구현보다 데이터 품질, 평가, 운영 정책이 훨씬 중요하다.

## 이해 점검 질문

1. 유해 콘텐츠 감지에서 softmax보다 sigmoid가 자연스러운 경우는 언제인가.
2. threshold를 낮추면 precision과 recall은 각각 어떻게 변하는가.
3. BERT가 키워드 필터보다 문맥에 강한 이유는 무엇인가.
4. 정체성 단어가 포함된 정상 문장을 오탐하는 문제를 어떻게 발견할 수 있는가.
5. 모델 점수를 바로 차단 결정으로 쓰면 어떤 문제가 생길 수 있는가.

## 참고

- Devlin et al., 2018, [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- Ouyang et al., 2022, [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

