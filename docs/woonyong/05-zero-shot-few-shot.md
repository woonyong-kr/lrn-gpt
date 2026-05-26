# 제로샷 학습과 퓨샷 학습

이 문서는 zero-shot과 few-shot을 "모델 weight를 업데이트하지 않고 prompt 안에서 task를 지정하는 방식" 중심으로 설명한다. 실제 weight를 바꾸는 fine-tuning은 [02-fine-tuning.md](02-fine-tuning.md)에서 다룬다.

## 먼저 구분해야 할 것

Zero-shot, few-shot이라는 말은 문맥에 따라 두 가지로 쓰인다.

1. Prompting 관점: prompt 안에 예시를 몇 개 넣는가.
2. Training 관점: task-specific 학습 예시를 몇 개 사용해 weight를 업데이트하는가.

GPT-3 논문에서 유명해진 zero-shot, one-shot, few-shot은 주로 prompting 관점이다. 즉, 모델 weight는 그대로 두고 prompt만 바꾼다.

이 문서에서는 prompting 관점을 중심으로 설명한다.

## Zero-shot

Zero-shot은 task 예시 없이 지시만 주는 방식이다.

```text
다음 문장의 감성을 긍정 또는 부정으로 분류해라.

문장: "이 영화는 생각보다 재미있었다."
정답:
```

모델은 사전 학습과 지시 이해 능력을 바탕으로 답을 낸다. prompt 안에는 정답 예시가 없다.

### Zero-shot이 가능한 이유

대규모 언어 모델은 pretraining 동안 수많은 텍스트 패턴을 본다. "질문에 답하기", "문장 분류하기", "번역하기", "요약하기" 같은 형식도 텍스트 안에 자주 등장한다. instruction tuning까지 거치면 모델은 자연어 지시를 task specification으로 해석하는 능력이 더 강해진다.

Zero-shot은 모델에게 새로운 weight를 학습시키는 것이 아니라, 이미 배운 능력을 자연어 지시로 꺼내 쓰는 것이다.

## One-shot

One-shot은 prompt 안에 예시 하나를 넣는다.

```text
다음 문장의 감성을 긍정 또는 부정으로 분류해라.

문장: "정말 지루했다."
정답: 부정

문장: "배우들의 연기가 좋았다."
정답:
```

예시 하나는 답변 형식과 label mapping을 알려준다.

## Few-shot

Few-shot은 prompt 안에 예시 여러 개를 넣는다.

```text
다음 문장의 감성을 긍정 또는 부정으로 분류해라.

문장: "정말 지루했다."
정답: 부정

문장: "다시 보고 싶다."
정답: 긍정

문장: "전개가 너무 어색했다."
정답: 부정

문장: "음악과 연출이 인상적이었다."
정답:
```

모델은 예시들을 보고 이 task의 입력 형식, 출력 형식, label 기준을 문맥 안에서 추론한다. 이것을 in-context learning이라고 부른다.

중요한 점은 few-shot prompting에서도 weight는 바뀌지 않는다는 것이다.

```text
few-shot prompting:
prompt changes, weights fixed

fine-tuning:
training data changes weights
```

## 비교표

| 구분 | Zero-shot | One-shot | Few-shot | Fine-tuning |
| --- | --- | --- | --- | --- |
| prompt 예시 | 0개 | 1개 | 여러 개 | 필요 없음 또는 별도 |
| weight 업데이트 | 없음 | 없음 | 없음 | 있음 |
| 준비 비용 | 낮음 | 낮음 | 낮음-중간 | 중간-높음 |
| 형식 안정성 | 낮을 수 있음 | 조금 좋아짐 | 더 좋아짐 | 가장 안정적일 수 있음 |
| 데이터 필요 | 거의 없음 | 예시 1개 | 예시 여러 개 | 학습 데이터 |
| 장점 | 빠른 실험 | label mapping 전달 | 기준 전달력 좋음 | 성능과 운영 안정성 |
| 단점 | 애매한 task에 약함 | 예시 하나에 과민 | context window 사용 | 학습 비용과 과적합 |

## GPT-3에서의 의미

GPT-3 논문은 모델을 task별로 fine-tuning하지 않고, prompt 안에 지시와 예시를 넣어 다양한 task를 수행하는 능력을 보여주었다.

핵심은 다음이다.

```text
모델 weight는 고정
task 설명과 예시는 prompt에 포함
모델은 다음 토큰 예측 방식으로 답 생성
```

GPT-3의 few-shot은 사람이 몇 개의 예시를 보고 새 문제 풀이 방식을 파악하는 것과 비슷해 보인다. 하지만 내부적으로는 gradient update가 아니라 Transformer context 안에서 패턴을 읽고 다음 토큰을 예측하는 과정이다.

## Zero-shot이 잘 되는 경우

Zero-shot은 task가 자연어로 명확하고, 모델이 이미 많이 본 형식일수록 잘 된다.

잘 맞는 예:

- 간단한 요약
- 일반 번역
- 쉬운 감성 분류
- 문체 변환
- 짧은 질의응답
- 형식이 단순한 정보 추출

어려운 예:

- 도메인 규칙이 특수한 분류
- label 정의가 애매한 문제
- 최신 내부 정책 기반 판단
- 수치 계산이 많이 필요한 문제
- 예시 없이는 출력 형식을 알기 어려운 문제

## Few-shot이 도움이 되는 이유

Few-shot 예시는 세 가지를 알려준다.

첫째, 입력과 출력 형식이다.

```text
문장: ...
정답: ...
```

둘째, label의 의미다.

```text
"별로였다" -> 부정
"볼만했다" -> 긍정
```

셋째, 애매한 기준이다.

```text
"기대보다는 아쉬웠지만 나쁘지 않았다" -> 긍정 또는 중립
```

예시가 많아질수록 모델은 이 prompt 안에서 기준을 더 잘 잡을 수 있다. 하지만 예시가 너무 많으면 context window를 많이 쓰고, 잘못 고른 예시가 모델을 오도할 수 있다.

## Prompt 예시 설계

Few-shot prompt를 만들 때는 예시를 아무거나 넣지 않는 것이 중요하다.

좋은 예시 조건:

- 입력 형식이 실제 문제와 같다.
- label이 명확하다.
- 쉬운 예시와 어려운 예시가 섞여 있다.
- class별 예시 수가 균형 있다.
- 출력 형식이 일관된다.
- 모호한 경우를 일부 포함해 기준을 보여준다.

나쁜 예시:

- label이 틀린 예시
- 형식이 제각각인 예시
- 한 class만 많은 예시
- 실제 문제와 도메인이 다른 예시
- 너무 긴 예시로 context를 낭비하는 예시

## Zero-shot classification과 label description

Zero-shot 분류에서는 label 이름만 주는 것보다 label 설명을 같이 주면 좋아질 때가 많다.

약한 prompt:

```text
다음 댓글을 toxic, normal 중 하나로 분류해라.
```

더 나은 prompt:

```text
다음 댓글을 분류해라.

toxic: 상대를 모욕하거나 위협하거나 혐오를 표현하는 댓글
normal: 비판이 있더라도 모욕이나 위협이 없는 댓글
```

이렇게 하면 모델이 label 단어 자체보다 label의 의미를 보고 판단할 수 있다.

## Few-shot과 데이터 누수

Few-shot 평가에서 조심해야 할 것이 데이터 누수다.

평가 문제와 너무 비슷한 예시를 prompt에 넣으면 실제 일반화 성능보다 높게 보일 수 있다. 특히 공개 benchmark에서는 대규모 모델이 pretraining 중 benchmark 일부를 봤을 가능성도 있다.

그래서 성능을 평가할 때는 다음을 확인해야 한다.

- prompt 예시와 test sample이 중복되지 않는가.
- 예시 순서를 바꿔도 성능이 유지되는가.
- label 이름을 바꿔도 성능이 유지되는가.
- 쉬운 sample만 골라 평가하지 않았는가.
- 여러 random seed 또는 여러 prompt template에서 평균을 보는가.

## Zero-shot, few-shot, fine-tuning 선택

| 상황 | 먼저 시도할 것 |
| --- | --- |
| 빠른 아이디어 검증 | zero-shot |
| 출력 형식이 자주 흔들림 | few-shot |
| label 기준이 애매함 | few-shot with edge cases |
| 높은 정확도와 재현성이 필요 | fine-tuning |
| 비용을 낮추고 싶음 | prompt 개선 후 작은 classifier 검토 |
| 운영 threshold가 필요 | classification fine-tuning |

좋은 순서는 보통 다음과 같다.

```text
zero-shot
  -> few-shot
  -> prompt template 개선
  -> 데이터 수집
  -> fine-tuning
  -> 운영 평가
```

처음부터 fine-tuning으로 가면 문제 정의가 틀렸을 때 비용이 크다. 먼저 prompt로 task 정의를 탐색하면 어떤 데이터가 필요한지 더 잘 보인다.

## Chain-of-thought와 few-shot

추론 문제에서는 few-shot 예시에 풀이 과정을 포함할 수 있다.

```text
문제: 철수는 사과 2개를 가지고 있고 3개를 더 샀다. 총 몇 개인가?
풀이: 처음 2개에 3개를 더하면 5개다.
정답: 5
```

이런 예시는 모델에게 "답만 내지 말고 중간 과정을 따라가라"는 패턴을 준다. 다만 모든 서비스에서 긴 추론을 사용자에게 그대로 보여주는 것이 좋은 것은 아니다. 정확도, 비용, 보안, 사용자 경험을 함께 봐야 한다.

## 헷갈리기 쉬운 용어

### Few-shot prompting

Prompt 안에 예시 몇 개를 넣는다. 모델 weight는 바뀌지 않는다.

### Few-shot fine-tuning

아주 적은 학습 데이터로 실제 fine-tuning을 한다. 모델 weight가 바뀐다.

### Zero-shot transfer

어떤 task에 직접 학습하지 않았지만 다른 학습에서 얻은 능력이 전이되어 수행한다.

### In-context learning

Prompt 안의 예시와 지시를 문맥으로 읽고, 그 자리에서 task 패턴을 따라 하는 현상이다.

## 이해 점검 질문

1. GPT-3의 few-shot은 왜 fine-tuning이 아닌가.
2. Zero-shot prompt에서 label 설명을 넣으면 왜 도움이 될 수 있는가.
3. Few-shot 예시를 class별로 균형 있게 넣어야 하는 이유는 무엇인가.
4. Prompt 예시 순서가 성능에 영향을 준다면 어떤 문제가 생기는가.
5. 운영용 moderation classifier를 zero-shot만으로 만들면 어떤 위험이 있는가.

## 참고

- Brown et al., 2020, [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- Ouyang et al., 2022, [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

