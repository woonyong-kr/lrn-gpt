# 어텐션 메커니즘 정리

이 문서는 간소화된 self-attention, self-attention, causal attention, multi-head attention의 차이를 정리한다. 목표는 "각 토큰이 다른 토큰을 얼마나 참고해서 자기 벡터를 새로 만들까"라는 한 질문으로 네 개념을 이어서 보는 것이다.

## 왜 책은 RNN encoder-decoder부터 설명하는가

책이 attention을 바로 설명하지 않고 RNN encoder-decoder를 먼저 보여주는 이유는 attention이 해결하려는 불편함을 느끼게 하려는 것이다.

전통적인 encoder-decoder RNN에서는 입력 문장을 왼쪽에서 오른쪽으로 하나씩 읽는다.

```text
나는       -> h1
어제       -> h2
도서관에서 -> h3
빌린       -> h4
책을       -> h5
오늘       -> h6
반납했다   -> h7
```

여기서 중요한 점은 `h7`이 단순히 마지막 단어 `반납했다`만의 벡터가 아니라는 것이다. RNN의 hidden state는 앞에서 읽은 정보를 계속 누적한다.

```text
h1 = "나는"까지 읽은 요약
h2 = "나는 어제"까지 읽은 요약
h3 = "나는 어제 도서관에서"까지 읽은 요약
...
h7 = "나는 어제 도서관에서 빌린 책을 오늘 반납했다" 전체를 읽은 요약
```

그래서 더 와닿는 표현으로 말하면 다음과 같다.

```text
입력 문장 전체를 마지막 hidden state 하나에 압축한다.
```

즉 encoder-decoder RNN은 길이가 다른 문장을 고정 크기의 context vector 하나로 바꾼다.

```text
짧은 문장 -> context vector 하나
긴 문장   -> context vector 하나
매우 긴 문장 -> context vector 하나
```

이 context vector가 decoder로 넘어가고, decoder는 이 벡터 하나를 바탕으로 출력 문장을 만든다.

```text
입력 문장 전체
  -> encoder RNN
  -> 마지막 hidden state 하나
  -> decoder RNN
  -> 출력 문장
```

문제는 긴 문장도 하나의 벡터에 눌러 담아야 한다는 점이다.

```text
누가 했는지
언제 했는지
어디서 했는지
무엇을 했는지
어떤 수식어가 붙었는지
```

이 정보들이 모두 마지막 hidden state 하나에 들어가야 한다. 그래서 문장이 길어질수록 앞부분 정보가 흐려지거나 빠질 수 있다. 이것을 정보 병목이라고 볼 수 있다.

Attention은 이 병목을 이렇게 바꾼다.

```text
문장 전체를 마지막 hidden state 하나에만 압축하지 말자.
h1, h2, h3, ..., h7을 모두 보관하자.
decoder가 출력 단어를 만들 때 필요한 입력 위치를 다시 보게 하자.
```

예를 들어 영어 번역에서 `book`을 만들 때는 `책을` 위치를 강하게 참고하면 된다.

```text
book 생성 시:
나는: 조금
도서관에서: 조금
책을: 많이
반납했다: 조금
```

`returned`를 만들 때는 `반납했다`를 강하게 참고하면 된다.

```text
returned 생성 시:
오늘: 조금
책을: 조금
반납했다: 많이
```

정리하면, attention은 처음부터 "멋진 새 수식"으로 나온 것이 아니라 RNN encoder-decoder의 다음 제약에서 출발한다.

```text
제약:
입력 문장 전체를 벡터 하나에 압축해야 한다.

attention의 생각:
벡터 하나만 믿지 말고, 필요할 때 입력의 각 위치를 다시 보자.
```

Transformer는 여기서 한 걸음 더 간다. RNN을 보조하기 위해 attention을 붙이는 수준이 아니라, 아예 RNN 없이 attention을 중심 구조로 사용한다. 그래서 `Attention Is All You Need`라는 제목이 나온다.

## 먼저 attention이란 무엇인가

Attention은 한 위치의 token vector를 새로 만들 때, 문장 안의 다른 token vector들을 얼마나 참고할지 계산하는 방식이다.

예를 들어 문장이 있다.

```text
나는 오늘 밥을 먹었다
```

`먹었다`라는 위치의 벡터를 만들 때 `밥을`은 강하게 참고하고, `오늘`은 중간 정도로 참고하고, `나는`은 문장 구조를 위해 조금 참고할 수 있다.

```text
먹었다가 참고하는 정도:
나는: 0.10
오늘: 0.20
밥을: 0.60
먹었다: 0.10
```

이 값을 attention weight라고 볼 수 있다. 그 다음 각 token vector를 이 비율만큼 섞는다.

```text
새로운 "먹었다" 벡터
= 0.10 * 나는 벡터
+ 0.20 * 오늘 벡터
+ 0.60 * 밥을 벡터
+ 0.10 * 먹었다 벡터
```

즉 attention은 "어디를 볼까"와 "얼마나 섞을까"를 계산하는 장치다.

### `context_vec_2` 예제는 어떤 단계인가

책의 다음 코드는 attention 전체가 아니라, 이미 계산된 attention weight를 사용해서 두 번째 token의 문맥 벡터를 만드는 마지막 단계다.

```python
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)

for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
```

여기서 `query = inputs[1]`은 두 번째 token을 기준으로 보겠다는 뜻이다. 책 그림에서는 `x^(2)`, 예제 문장에서는 `"journey"` 위치에 해당한다.

```text
inputs[0] = x^(1) = "your"의 벡터
inputs[1] = x^(2) = "journey"의 벡터
inputs[2] = x^(3) = "starts"의 벡터
...
```

`attn_weights_2[i]`는 두 번째 token이 i번째 token을 얼마나 참고할지 나타내는 비율이다.

```text
attn_weights_2[0] = alpha_21
attn_weights_2[1] = alpha_22
attn_weights_2[2] = alpha_23
...
```

그래서 반복문은 다음 수식을 코드로 쓴 것이다.

```text
z^(2)
= alpha_21 * x^(1)
 + alpha_22 * x^(2)
 + alpha_23 * x^(3)
 + ...
 + alpha_2T * x^(T)
```

작은 숫자로 보면 더 직접적이다.

```text
x^(1) = [1, 0]
x^(2) = [0, 1]
x^(3) = [1, 1]

attn_weights_2 = [0.2, 0.5, 0.3]
```

그러면 `context_vec_2`는 처음에 0벡터로 시작한다.

```text
context_vec_2 = [0, 0]
```

반복문 1회차:

```text
context_vec_2 += 0.2 * [1, 0]
context_vec_2  = [0.2, 0.0]
```

반복문 2회차:

```text
context_vec_2 += 0.5 * [0, 1]
context_vec_2  = [0.2, 0.5]
```

반복문 3회차:

```text
context_vec_2 += 0.3 * [1, 1]
context_vec_2  = [0.5, 0.8]
```

최종 결과는 다음이다.

```text
z^(2) = context_vec_2 = [0.5, 0.8]
```

이 값은 두 번째 token 하나의 새 벡터다. 다만 원래 `x^(2)`만 복사한 것이 아니라, 모든 token 벡터를 attention weight만큼 섞어서 만든 새 표현이다.

```text
원래 두 번째 token:
x^(2)

attention 후 두 번째 token:
z^(2) = 문장 전체를 참고한 x^(2)의 새 표현
```

그래서 이 코드를 어텐션 연산이라고 부르는 이유는 다음과 같다.

```text
attention score 계산
  -> softmax로 attention weight 계산
  -> attention weight로 모든 token vector를 가중합
  -> context vector 생성
```

질문에 나온 코드는 이 중 마지막 단계다.

### 모든 token에 대해 한 번에 계산하기

앞의 `context_vec_2` 예제는 두 번째 token 하나만 계산했다. 그런데 self-attention은 실제로는 모든 token 위치에 대해 같은 일을 한다.

```text
x^(1) -> z^(1)
x^(2) -> z^(2)
x^(3) -> z^(3)
...
x^(T) -> z^(T)
```

즉 두 번째 token의 context vector만 만드는 것이 아니라, 모든 token의 새 context vector를 만든다.

질문에 나온 코드를 올바른 형태로 정리하면 다음과 같다.

```python
attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

attn_weights = torch.softmax(attn_scores, dim=-1)
row_2_sum = attn_weights[1].sum()
all_context_vecs = attn_weights @ inputs
```

여기서 `inputs`가 6개 token의 벡터라고 하자.

```text
inputs shape = (6, 3)
```

이 말은 다음 뜻이다.

```text
token 개수 = 6개
각 token vector 차원 = 3차원
```

`attn_scores`는 `(6, 6)`이다.

```text
attn_scores shape = (6, 6)
```

왜 `(6, 6)`이냐면, 6개 token 각각이 6개 token 전부와 한 번씩 비교되기 때문이다.

```text
attn_scores[i, j] = dot(x_i, x_j)
```

행 `i`는 i번째 token을 기준으로 본다는 뜻이다.

```text
attn_scores[0, :] = 0번째 token이 모든 token을 본 점수
attn_scores[1, :] = 1번째 token이 모든 token을 본 점수
attn_scores[2, :] = 2번째 token이 모든 token을 본 점수
...
```

열 `j`는 참고 대상 token이다.

```text
attn_scores[i, 0] = i번째 token이 0번째 token을 보는 점수
attn_scores[i, 1] = i번째 token이 1번째 token을 보는 점수
attn_scores[i, 2] = i번째 token이 2번째 token을 보는 점수
...
```

그 다음 softmax를 적용한다.

```python
attn_weights = torch.softmax(attn_scores, dim=-1)
```

`dim=-1`은 마지막 차원에 softmax를 적용한다는 뜻이다. `attn_scores`가 `(6, 6)`이므로 마지막 차원은 각 행의 열 방향이다. 그래서 각 행마다 따로 softmax가 적용된다.

```text
attn_weights[0, :]의 합 = 1
attn_weights[1, :]의 합 = 1
attn_weights[2, :]의 합 = 1
...
```

`row_2_sum`은 책에서 두 번째 token `x^(2)`의 attention weight 합을 확인하는 코드에 가깝다. Python index로는 두 번째 token이 `1`번 row다.

```python
row_2_sum = attn_weights[1].sum()
```

이 값이 1이 되는 이유는 softmax가 한 token이 다른 token들을 참고하는 비율을 합 1로 만들기 때문이다.

마지막 줄이 간소화된 self-attention의 최종 단계다.

```python
all_context_vecs = attn_weights @ inputs
```

shape로 보면 다음과 같다.

```text
attn_weights shape = (6, 6)
inputs shape       = (6, 3)

all_context_vecs shape = (6, 3)
```

행렬곱은 다음 의미다.

```text
all_context_vecs[0] = z^(1)
all_context_vecs[1] = z^(2)
all_context_vecs[2] = z^(3)
...
```

즉 `all_context_vecs`는 모든 token의 새 context vector를 모은 결과다.

```text
all_context_vecs =
[
  z^(1),
  z^(2),
  z^(3),
  z^(4),
  z^(5),
  z^(6),
]
```

`all_context_vecs[1]`은 앞에서 for문으로 직접 만들었던 `context_vec_2`와 같은 값이다.

```text
context_vec_2
= attn_weights[1, 0] * x^(1)
 + attn_weights[1, 1] * x^(2)
 + attn_weights[1, 2] * x^(3)
 + ...
 + attn_weights[1, 5] * x^(6)

all_context_vecs[1] = context_vec_2
```

정리하면 간소화된 self-attention은 여기서 끝난다.

```text
1. 입력 벡터 X를 준비한다.
2. 모든 token pair의 dot product를 계산한다.
   attn_scores = X @ X.T
3. 각 row에 softmax를 적용한다.
   attn_weights = softmax(attn_scores, dim=-1)
4. attention weight로 입력 벡터들을 섞는다.
   all_context_vecs = attn_weights @ X
```

수식으로 쓰면 다음이다.

```text
Z = softmax(X X^T) X
```

이것이 간소화된 self-attention이다. 아직 `W_Q`, `W_K`, `W_V`가 없기 때문에 학습 가능한 attention projection은 아니다. 실제 Transformer에서는 다음처럼 확장된다.

```text
Q = X W_Q
K = X W_K
V = X W_V

Z = softmax(Q K^T / sqrt(d_k)) V
```

## 공통 입력

앞 단계에서 이미 input embedding을 만들었다고 하자.

```text
token embedding + position embedding
  -> input embeddings
```

shape로 쓰면 다음과 같다.

```text
X shape: (B, T, d_model)
```

| 기호 | 뜻 |
| --- | --- |
| `B` | batch size |
| `T` | sequence length, token 개수 |
| `d_model` | token vector 차원 |
| `X` | Transformer에 들어온 입력 벡터들 |

attention은 이 `X`를 보고 각 위치의 새 벡터를 만든다.

```text
X
  -> attention
  -> context vectors
```

## 1. 간소화된 self-attention

간소화된 self-attention은 학습 가능한 `W_Q`, `W_K`, `W_V` 없이 input vector 자체로 attention을 계산하는 교육용 버전이다.

```text
Query = X
Key = X
Value = X
```

어떤 위치 `i`가 다른 위치 `j`를 얼마나 볼지는 두 벡터의 dot product로 계산한다.

```text
score(i, j) = x_i · x_j
```

그 다음 softmax로 합이 1인 비율로 바꾼다.

```text
attention_weight(i, :) = softmax(score(i, :))
```

마지막으로 모든 value vector를 가중합한다.

```text
context_i = sum_j attention_weight(i, j) * x_j
```

간소화된 self-attention의 의미는 다음이다.

```text
내 벡터와 비슷한 벡터를 더 많이 참고한다.
```

장점은 구조를 이해하기 쉽다는 것이다. 단점은 attention 자체에 학습 가능한 projection이 없어서, 모델이 "무엇을 query로 볼지", "무엇을 key로 볼지", "무엇을 value로 가져올지"를 따로 배울 수 없다는 점이다.

```text
간소화된 self-attention:
입력 X 그대로 유사도 계산
학습 가능한 Q/K/V 가중치 없음
개념 이해용
```

## 2. Self-attention

일반적인 self-attention은 input vector `X`에서 `Q`, `K`, `V`를 각각 학습 가능한 선형층으로 만든다.

```text
Q = X W_Q
K = X W_K
V = X W_V
```

여기서 `W_Q`, `W_K`, `W_V`는 학습되는 파라미터다.

| 이름 | 의미 |
| --- | --- |
| Query | 지금 위치가 찾고 싶은 정보의 모양 |
| Key | 각 위치가 가진 검색용 표식 |
| Value | 실제로 가져와 섞을 내용 |

비유하면 다음과 같다.

```text
Query:
내가 지금 찾는 질문

Key:
각 token이 내놓는 색인표

Value:
그 token에서 실제로 가져올 정보
```

attention score는 query와 key의 dot product로 만든다.

```text
scores = Q K^T
```

보통 차원이 커질수록 dot product 값이 커지므로 `sqrt(d_k)`로 나눈다.

```text
scores = Q K^T / sqrt(d_k)
```

그 다음 softmax와 value 가중합을 한다.

```text
attention_weights = softmax(scores)
context = attention_weights V
```

여기서 `context = attention_weights V`는 `V`를 수정한다는 뜻이 아니다.
attention weight라는 비율로 여러 value vector를 섞어서 **현재 token의 새 context vector**를 만든다는 뜻이다.

예를 들어 현재 token이 1번이고, 참조 후보가 6개라면 attention weight는 6개짜리 벡터다.

```text
attention_weights shape = (6,)
values shape            = (6, 2)

attention_weights @ values
= (6,) @ (6, 2)
= (2,)
```

`(6,)`은 다음처럼 6개 후보를 얼마나 참고할지 나타낸다.

```text
attention_weights =
[0.17, 0.16, 0.16, 0.16, 0.16, 0.19]
```

`values`는 각 후보 token이 실제로 제공할 내용 벡터다.

```text
values =
[
  v0,
  v1,
  v2,
  v3,
  v4,
  v5,
]

각 v_i shape = (2,)
```

행렬곱 `attention_weights @ values`는 아래 가중합을 한 번에 계산한다.

```text
context
= 0.17 * v0
 + 0.16 * v1
 + 0.16 * v2
 + 0.16 * v3
 + 0.16 * v4
 + 0.19 * v5
```

그래서 결과는 후보 6개가 아니라, 현재 token 하나에 대한 새 벡터 하나다.

```text
context shape = (2,)
```

이 context vector는 최종 답이 아니다. Transformer block 안에서 attention이 만든 중간 표현이다. 실제 GPT에서는 이후 output projection, residual connection, layer norm, feed-forward network를 지나 다음 block으로 넘어가고, 마지막에는 vocab logits로 변환되어 다음 token 예측에 사용된다.

최종 수식은 다음이다.

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

self-attention이라고 부르는 이유는 `Q`, `K`, `V`가 모두 같은 입력 `X`에서 나오기 때문이다.

```text
같은 문장 안의 token들이
같은 문장 안의 다른 token들을 참고한다.
```

mask가 없다면 각 위치는 모든 위치를 볼 수 있다.

```text
0번 token -> 0, 1, 2, 3 모두 볼 수 있음
1번 token -> 0, 1, 2, 3 모두 볼 수 있음
2번 token -> 0, 1, 2, 3 모두 볼 수 있음
3번 token -> 0, 1, 2, 3 모두 볼 수 있음
```

BERT 같은 encoder-only 모델은 이런 양방향 self-attention을 사용한다. 문장을 이해하거나 분류할 때는 오른쪽 문맥까지 보는 것이 도움이 되기 때문이다.

## 3. Causal attention

Causal attention은 self-attention에 미래 token을 보지 못하게 하는 mask를 추가한 것이다. GPT에서 쓰는 핵심 방식이다.

GPT는 다음 token을 맞히는 모델이다.

```text
input:  [나는, 오늘, 밥을]
target: [오늘, 밥을, 먹었다]
```

만약 `오늘` 위치가 뒤의 `밥을`, `먹었다`를 볼 수 있으면 정답을 훔쳐보는 셈이다. 그래서 각 위치는 자기 자신과 과거 위치만 볼 수 있어야 한다.

길이 4일 때 허용되는 관계는 다음과 같다.

```text
      key position
      0  1  2  3
q 0   O  X  X  X
u 1   O  O  X  X
e 2   O  O  O  X
r 3   O  O  O  O
y
position
```

구현에서는 softmax 전에 미래 위치 score를 아주 작은 값으로 바꾼다.

```text
scores[i, j] = -inf if j > i
```

softmax를 통과하면 해당 위치의 attention weight는 0이 된다.

```text
미래 token의 attention weight = 0
```

정리하면 causal attention은 다음이다.

```text
self-attention
+ 미래 token을 가리는 mask
= causal attention
```

그래서 causal attention은 보통 masked self-attention이라고도 부른다.

```text
일반 self-attention:
모든 token을 볼 수 있음

causal self-attention:
과거와 현재 token만 볼 수 있음
```

## 4. Multi-head attention

Multi-head attention은 attention을 여러 개의 head로 나눠 병렬로 수행하는 방식이다.

하나의 attention head만 있으면 한 종류의 관계에 치우칠 수 있다. 하지만 문장 안 관계는 여러 종류다.

```text
바로 앞 단어와의 관계
주어와 동사의 관계
부정어와 서술어의 관계
대명사와 선행사의 관계
코드에서 괄호와 들여쓰기의 관계
```

Multi-head attention은 head마다 다른 `W_Q`, `W_K`, `W_V`를 가진다.

```text
head_1 = Attention(X W_Q1, X W_K1, X W_V1)
head_2 = Attention(X W_Q2, X W_K2, X W_V2)
...
head_h = Attention(X W_Qh, X W_Kh, X W_Vh)
```

그 다음 head 결과를 이어 붙인다.

```text
concat = Concat(head_1, head_2, ..., head_h)
```

마지막으로 출력 projection을 통과시켜 다시 `d_model` 차원으로 맞춘다.

```text
output = concat W_O
```

shape 감각은 다음과 같다.

```text
X shape:
(B, T, d_model)

head output shape:
(B, T, d_head)

concat heads shape:
(B, T, h * d_head)

보통 h * d_head = d_model

final output shape:
(B, T, d_model)
```

예를 들어 `d_model = 768`, `head = 12`이면 head 하나의 차원은 보통 64다.

```text
d_head = 768 / 12 = 64
```

중요한 점은 multi-head attention이 causal 여부를 결정하는 말은 아니라는 것이다.

```text
multi-head self-attention:
여러 head를 쓰는 self-attention

multi-head causal self-attention:
여러 head를 쓰고, 미래 token도 mask로 가리는 attention
```

GPT에서 실제로 많이 말하는 것은 `masked multi-head self-attention`이다.

```text
masked = 미래 token을 가림
multi-head = 여러 관점의 attention head를 사용
self-attention = 같은 sequence 안에서 서로를 봄
```

## 네 가지 차이 한눈에 보기

| 구분 | 핵심 아이디어 | 학습 파라미터 | 미래 token 보기 | 주 용도 |
| --- | --- | --- | --- | --- |
| 간소화된 self-attention | 입력 벡터끼리 직접 유사도 계산 | attention용 Q/K/V 없음 | 보통 모두 봄 | 개념 이해 |
| self-attention | `Q`, `K`, `V`를 학습해서 관계 계산 | `W_Q`, `W_K`, `W_V` | mask 없으면 모두 봄 | BERT encoder, Transformer encoder |
| causal attention | self-attention에 미래 가림 mask 추가 | `W_Q`, `W_K`, `W_V` | 미래 못 봄 | GPT next-token prediction |
| multi-head attention | attention을 여러 관점으로 병렬 수행 | head별 `W_Q/K/V`, `W_O` | mask 설정에 따라 다름 | Transformer 대부분 |

## 단계별로 쌓아 보기

개념은 이렇게 쌓인다.

```text
1. 간소화된 self-attention
   X로 바로 유사도 계산

2. self-attention
   X에서 Q, K, V를 학습해서 만든 뒤 attention 계산

3. causal attention
   self-attention score에서 미래 위치를 -inf로 가림

4. multi-head attention
   attention을 여러 head로 여러 번 수행하고 합침
```

GPT의 attention은 이 중에서 다음 조합이다.

```text
self-attention
+ causal mask
+ multi-head
= multi-head causal self-attention
```

## 왜 attention이 중요한가

LLM이 다음 token을 잘 맞히려면 현재 위치의 token만 봐서는 안 된다. 문맥 안의 다른 token들을 참고해야 한다.

```text
그 영화는 길었지만 지루하지 않았다
```

`지루하지`를 제대로 이해하려면 `않았다`와 연결되어야 한다. attention은 이런 연결 강도를 학습한다.

또한 self-attention은 모든 위치의 관계를 행렬 곱으로 한 번에 계산할 수 있다.

```text
Q K^T -> (T, T) 관계표
```

그래서 RNN처럼 왼쪽에서 오른쪽으로 하나씩 순차 처리하는 구조보다 학습 병렬화에 유리하다. 대신 `T x T` 관계표가 필요하므로 sequence length가 길어질수록 메모리와 계산량은 크게 늘어난다.

```text
attention cost roughly grows with T^2
```

## 이 레포와 연결

구현할 때는 다음 파일과 연결된다.

```text
src/embeddings.py
  -> token embedding + position embedding
  -> X 생성

src/attention.py
  -> causal multi-head self-attention 구현

src/model.py
  -> attention + FFN + residual + layer norm을 Transformer block으로 묶음
```

구현 순서로 보면 다음을 확인하면 된다.

```text
1. Q, K, V shape가 맞는가
2. Q K^T 결과가 (B, heads, T, T)인가
3. causal mask가 미래 위치를 막는가
4. softmax 뒤 미래 위치 weight가 0인가
5. head concat 뒤 shape가 다시 (B, T, d_model)인가
```

## 이해 점검 질문

1. 간소화된 self-attention에는 왜 `W_Q`, `W_K`, `W_V`가 없는가.
2. self-attention에서 Query, Key, Value는 각각 어떤 역할인가.
3. causal mask가 없으면 GPT 학습에서 어떤 문제가 생기는가.
4. multi-head attention은 causal attention과 같은 말인가, 다른 말인가.
5. GPT의 attention을 한 줄로 쓰면 왜 multi-head causal self-attention인가.

## 참고

- Vaswani et al., 2017, [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
