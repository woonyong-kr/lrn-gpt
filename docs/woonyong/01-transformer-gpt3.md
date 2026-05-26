# Transformer와 GPT-3 구조

이 문서는 `Attention Is All You Need`의 Transformer 구조를 먼저 정리한 뒤, GPT-3 기준의 decoder-only Transformer가 어떻게 다음 토큰을 예측하는지 설명한다. 이 레포의 `src/attention.py`, `src/embeddings.py`, `src/model.py`를 구현할 때 바로 연결되는 개념을 중심으로 쓴다.

## 한 문장으로 보는 전체 구조

Transformer는 문장을 순서대로 처리하는 RNN 대신, 모든 토큰 위치가 서로를 한 번에 참조하는 self-attention을 사용해 문맥 벡터를 만든다. GPT 계열은 그중 decoder 쪽 구조만 사용하고, causal mask로 미래 토큰을 가린 상태에서 다음 토큰 확률을 예측한다.

```text
token ids
  -> token embedding + position embedding
  -> [LayerNorm -> masked multi-head self-attention -> residual
      LayerNorm -> feed-forward network -> residual] x N
  -> final LayerNorm
  -> linear projection to vocab
  -> next-token probability
```

## 표기

| 기호 | 뜻 |
| --- | --- |
| `B` | batch size |
| `T` | sequence length, context length |
| `V` | vocabulary size |
| `d_model` | token hidden vector dimension |
| `h` | number of attention heads |
| `d_k` | dimension per head, 보통 `d_model / h` |
| `X` | 입력 hidden states, shape `(B, T, d_model)` |
| `Q, K, V` | query, key, value |

수식에서 `V`는 vocabulary size로도 쓰이고 value matrix로도 쓰인다. 혼동이 생기면 value matrix는 `Value`라고 읽으면 된다.

## 원 논문 Transformer 구조

`Attention Is All You Need`의 Transformer는 원래 번역을 위한 encoder-decoder 모델이다.

```text
source sentence
  -> encoder stack
  -> encoded memory
  -> decoder stack
  -> target sentence
```

Encoder layer는 두 부분으로 구성된다.

1. Multi-head self-attention
2. Position-wise feed-forward network

Decoder layer는 세 부분으로 구성된다.

1. Masked multi-head self-attention
2. Encoder-decoder attention
3. Position-wise feed-forward network

GPT는 이 구조에서 decoder의 핵심만 가져온다. 번역처럼 별도의 source sentence를 받지 않으므로 encoder와 encoder-decoder attention은 없다. 대신 입력 문장 자체를 왼쪽에서 오른쪽으로 보며 다음 토큰을 예측한다.

| 구분 | 원 논문 Transformer | GPT 계열 |
| --- | --- | --- |
| 목적 | 입력 문장을 출력 문장으로 변환 | 다음 토큰 예측 |
| 구조 | encoder-decoder | decoder-only |
| attention | encoder self-attention, decoder masked self-attention, cross-attention | masked self-attention |
| 미래 토큰 | decoder에서 가림 | 모든 layer에서 가림 |
| 대표 학습 | 번역 병렬 데이터 | 대규모 텍스트 next-token prediction |

## 입력 표현

신경망은 문자열을 직접 읽지 못한다. 먼저 텍스트를 토큰 ID로 바꾸고, 토큰 ID를 벡터로 바꾼다.

```text
"나는 밥을 먹었다"
  -> tokenizer
  -> [token_id_0, token_id_1, token_id_2, ...]
  -> embedding lookup
  -> vectors
```

이 레포는 byte-level BPE tokenizer를 직접 구현하는 과제다. BPE는 자주 함께 등장하는 byte나 subword pair를 병합하면서 vocabulary를 만든다. GPT-3도 BPE 계열 토큰화를 사용한다. BPE의 장점은 처음 보는 단어도 byte/subword 조합으로 표현할 수 있다는 점이다.

토큰 임베딩만 있으면 모델은 같은 토큰을 항상 같은 벡터로 본다. 하지만 문장에서는 위치가 중요하다.

```text
"개가 사람을 물었다"
"사람이 개를 물었다"
```

두 문장은 등장 단어가 비슷해도 의미가 다르다. 그래서 입력에는 token embedding과 position embedding을 더한다.

```text
H_0 = TokenEmbedding(input_ids) + PositionEmbedding(positions)
```

원 논문은 sinusoidal positional encoding을 사용했다. GPT 계열 구현에서는 learned positional embedding을 사용하는 경우가 많다. 이 레포의 `InputEmbedding`도 `nn.Embedding(context_length, emb_dim)` 형태의 learned position embedding을 구현하도록 되어 있다.

## Attention의 직관

Self-attention은 각 토큰이 "지금 내 표현을 만들 때 어느 다른 토큰을 얼마나 참고할 것인가"를 계산하는 장치다.

예를 들어 "그 영화는 길었지만 지루하지 않았다"에서 "지루하지"는 "않았다"와 강하게 연결되어야 한다. 단어 자체만 보면 부정적인 단어처럼 보일 수 있지만, 문맥을 보면 긍정 또는 중립에 가까워진다. Attention은 이런 문맥 연결을 학습한다.

Query, key, value는 정보 검색에 비유할 수 있다.

- Query: 내가 지금 찾고 싶은 정보의 모양
- Key: 각 토큰이 가진 색인
- Value: 실제로 가져올 내용

각 위치의 query가 모든 위치의 key와 점수를 계산하고, 그 점수로 value를 가중합한다.

## Scaled dot-product attention

입력 hidden states `X`가 있다.

```text
X shape: (B, T, d_model)
```

각 head마다 선형 변환을 통해 `Q`, `K`, `Value`를 만든다.

```text
Q = X W_Q
K = X W_K
Value = X W_V
```

한 head에서 shape는 보통 다음과 같다.

```text
Q, K, Value shape: (B, T, d_k)
```

각 토큰 위치가 다른 토큰 위치를 얼마나 볼지 점수 행렬을 만든다.

```text
Score = Q K^T / sqrt(d_k)
Score shape: (B, T, T)
```

`Score[b, i, j]`는 batch `b`에서 i번째 토큰이 j번째 토큰을 얼마나 참고할지에 대한 logit이다. 여기에 softmax를 적용하면 attention weight가 된다.

```text
A = softmax(Score)
HeadOutput = A Value
```

최종 수식은 다음과 같다.

```text
Attention(Q, K, Value) = softmax(Q K^T / sqrt(d_k)) Value
```

## 왜 `sqrt(d_k)`로 나누는가

이 부분은 단순한 트릭처럼 보이지만, softmax의 안정성과 직접 연결된다.

가정:

- query 벡터 `q`와 key 벡터 `k`의 각 성분은 평균 0, 분산 1이다.
- 성분들은 서로 독립이라고 단순화한다.

점곱은 다음과 같다.

```text
q · k = q_1 k_1 + q_2 k_2 + ... + q_d k_d
```

각 항 `q_i k_i`의 평균은 0이고 분산은 1이라고 볼 수 있다. 독립인 항의 합의 분산은 분산의 합이므로,

```text
Var(q · k) = d_k
```

즉 `d_k`가 커질수록 점곱 값의 스케일이 커진다. softmax에 너무 큰 양수와 음수가 들어가면 한 위치가 거의 1, 나머지가 거의 0이 된다. 이 상태에서는 gradient가 작아지고 학습이 느려진다.

`sqrt(d_k)`로 나누면 분산이 다음처럼 정규화된다.

```text
Var((q · k) / sqrt(d_k)) = Var(q · k) / d_k = 1
```

그래서 attention logit이 너무 커지는 것을 막고, softmax가 덜 포화되도록 만든다. 이게 scaled dot-product attention의 핵심이다.

## Causal mask

GPT는 다음 토큰을 예측해야 한다. t번째 위치에서 t+1 이후 토큰을 보면 정답을 훔쳐보는 셈이다. 그래서 causal mask를 적용한다.

예를 들어 길이 4 시퀀스라면 허용되는 attention은 아래와 같다.

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

미래 위치는 softmax 전에 매우 작은 값, 보통 `-inf`로 바꾼다.

```text
Score[i, j] = -inf if j > i
```

softmax를 통과하면 `j > i`인 위치의 weight는 0이 된다.

### Causal mask가 미래 정보 누수를 막는다는 증명

명제: layer `l`의 위치 `t` hidden state는 입력 위치 `0..t`에만 의존한다.

기저:

입력 embedding `H_0[t]`는 token `x_t`와 position `t`에만 의존한다. 따라서 미래 토큰 `x_{t+1..}`에는 의존하지 않는다.

귀납 가정:

layer `l`의 모든 위치 `i <= t` hidden state는 입력 `0..i`에만 의존한다고 하자.

귀납 단계:

layer `l+1`에서 위치 `t`의 attention은 causal mask 때문에 key/value 위치 `0..t`만 볼 수 있다. 각 value 위치 `i <= t`는 귀납 가정에 의해 입력 `0..i`에만 의존하고, 이는 모두 `0..t` 안에 포함된다. 따라서 attention 출력은 미래 입력에 의존하지 않는다.

Feed-forward network는 각 위치에 독립적으로 적용된다. residual connection과 layer normalization도 같은 위치의 벡터를 변환할 뿐 미래 위치를 섞지 않는다. 그러므로 layer `l+1`의 위치 `t`도 입력 `0..t`에만 의존한다.

결론:

모든 layer에서 위치 `t`는 미래 토큰을 볼 수 없다. 따라서 GPT의 다음 토큰 예측은 자기회귀 조건을 만족한다.

## Multi-head attention

하나의 attention head만 있으면 한 종류의 관계에 치우치기 쉽다. Multi-head attention은 여러 head가 각자 다른 projection을 사용해 서로 다른 관점의 관계를 본다.

```text
head_i = Attention(X W_Q_i, X W_K_i, X W_V_i)
MultiHead(X) = Concat(head_1, ..., head_h) W_O
```

head 수가 `h`이고 `d_model`이 `h`로 나누어 떨어지면 보통 `d_k = d_model / h`가 된다.

예를 들어 `d_model = 768`, `h = 12`이면 head 하나의 차원은 `64`다. 각 head는 작지만 여러 head를 병렬로 계산하고 concat하므로 최종 차원은 다시 `768`이 된다.

Multi-head가 유용한 이유는 관계의 종류가 하나가 아니기 때문이다.

- 바로 앞 토큰과의 지역적 연결
- 문장 전체 주제와의 연결
- 부정어와 수식어의 연결
- 대명사와 선행사의 연결
- 코드에서 괄호, 들여쓰기, 변수 이름의 연결

각 head가 반드시 사람이 해석 가능한 역할을 갖는 것은 아니지만, 여러 subspace에서 관계를 볼 수 있게 만든다는 점이 중요하다.

## Feed-forward network

Attention은 위치 사이의 정보를 섞는다. Feed-forward network는 각 위치의 벡터를 더 풍부하게 비선형 변환한다.

원 논문의 FFN은 다음 형태다.

```text
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
```

GPT 계열은 ReLU 대신 GELU를 많이 사용한다.

```text
FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
```

보통 내부 차원은 `4 * d_model`이다.

```text
d_model -> 4 * d_model -> d_model
```

Attention이 "어디를 볼까"라면 FFN은 "가져온 정보를 어떻게 해석하고 변환할까"에 가깝다.

## Residual connection과 LayerNorm

Transformer block은 보통 attention과 FFN 주변에 residual connection을 둔다.

GPT 계열 구현에서 많이 쓰는 pre-LN 형태는 다음과 같다.

```text
y = x + Attention(LayerNorm(x))
z = y + FFN(LayerNorm(y))
```

Residual connection은 깊은 모델에서 gradient가 흐르는 길을 만든다. LayerNorm은 각 토큰 벡터의 분포를 안정화한다. 이 둘이 없으면 layer를 깊게 쌓을수록 학습이 불안정해진다.

이 레포의 `TransformerBlock` 주석도 `LayerNorm -> Causal Self-Attention -> residual`, `LayerNorm -> FeedForward -> residual` 흐름을 요구한다. 즉 pre-LN GPT 스타일로 구현하면 된다.

## GPT의 학습 목표

GPT는 문장 전체 확률을 왼쪽에서 오른쪽으로 분해한다.

```text
P(x_1, x_2, ..., x_T) = product_t P(x_t | x_1, ..., x_{t-1})
```

훈련할 때는 입력과 정답을 한 칸 밀어 만든다.

```text
input:  [나는, 오늘, 밥을]
target: [오늘, 밥을, 먹었다]
```

모델은 각 위치에서 vocabulary 전체에 대한 logits를 낸다.

```text
logits shape: (B, T, vocab_size)
```

loss는 정답 토큰에 대한 cross entropy다.

```text
loss = -log P(target_token | previous_tokens)
```

이 단순한 목표가 강력한 이유는 텍스트의 거의 모든 능력이 다음 토큰 예측 안에 압축될 수 있기 때문이다. 문법, 사실, 형식, 추론 패턴, 코드 구조, 대화 관습이 모두 다음에 올 말의 확률에 영향을 준다.

## GPT-3 기준으로 본 decoder-only Transformer

GPT-3는 GPT-2와 같은 계열의 decoder-only Transformer를 매우 크게 확장한 모델이다. 논문에서 가장 큰 GPT-3 모델은 175B parameter 모델이며, 문맥 길이는 2048 token이다.

| 항목 | GPT-3 175B 기준 |
| --- | --- |
| 구조 | decoder-only Transformer |
| parameter 수 | 약 175B |
| layer 수 | 96 |
| hidden dimension | 12288 |
| attention heads | 96 |
| head dimension | 128 |
| context window | 2048 tokens |
| 학습 목표 | next-token prediction |
| few-shot 방식 | prompt 안에 예시를 넣고 gradient update 없이 답을 생성 |

여기서 중요한 점은 GPT-3의 few-shot 능력이 "그 자리에서 weight를 바꾸는 학습"이 아니라는 것이다. 예시를 prompt 안에 넣으면, 모델은 그 예시를 문맥으로 읽고 다음 토큰을 예측한다. 이것을 in-context learning이라고 부른다.

## GPT와 원 논문 Transformer의 가장 중요한 차이

원 논문 Transformer는 encoder와 decoder가 모두 있다. GPT는 decoder-only다.

원 논문 Transformer decoder에는 cross-attention이 있다. GPT에는 없다.

원 논문 Transformer는 번역처럼 source 문장과 target 문장을 나누어 다룬다. GPT는 하나의 긴 token sequence를 보고 다음 token을 예측한다.

그래서 GPT 구현에서 가장 중요한 attention은 masked self-attention이다.

```text
encoder self-attention: 모든 입력 위치를 볼 수 있음
decoder masked self-attention: 과거와 현재만 볼 수 있음
GPT masked self-attention: 과거와 현재만 볼 수 있음
```

## 추론과 디코딩

훈련된 GPT는 다음 토큰 확률 분포를 낸다.

```text
P(next token | prompt)
```

텍스트를 생성하려면 이 분포에서 토큰을 선택해야 한다.

| 방법 | 설명 | 특징 |
| --- | --- | --- |
| greedy decoding | 가장 확률 높은 토큰 하나를 선택 | 안정적이지만 반복적일 수 있음 |
| temperature sampling | logits를 온도로 나누어 확률 분포를 조절 | 낮으면 보수적, 높으면 다양함 |
| top-k sampling | 확률 상위 k개 토큰에서만 샘플링 | 이상한 저확률 토큰을 줄임 |
| top-p sampling | 누적 확률 p까지의 후보에서 샘플링 | 문맥마다 후보 수가 달라짐 |

이 레포의 `generate_text_simple`은 greedy decoding을 구현하는 과제다. 먼저 greedy를 정확히 구현하고, 이후에 temperature나 top-k를 추가하면 구조를 이해하기 쉽다.

## 왜 Transformer가 RNN보다 병렬화에 유리한가

RNN은 `h_t`를 계산하려면 `h_{t-1}`이 필요하다. 그래서 sequence length가 `T`이면 시간 방향으로 순차성이 생긴다.

```text
h_1 -> h_2 -> h_3 -> ... -> h_T
```

Self-attention은 모든 위치의 `Q`, `K`, `Value`를 행렬 곱으로 한 번에 계산한다.

```text
Q = X W_Q
K = X W_K
Value = X W_V
```

그리고 `Q K^T`도 행렬 곱으로 병렬 계산된다. 따라서 학습 시 GPU를 훨씬 잘 활용할 수 있다. 대신 attention score가 `(T, T)`라서 sequence length가 길어질수록 메모리와 계산량이 `O(T^2)`로 증가한다.

## Self-attention의 계산 복잡도

한 layer에서 attention의 큰 비용은 두 가지다.

1. `Q K^T`: 대략 `O(T^2 d_model)`
2. `A Value`: 대략 `O(T^2 d_model)`

따라서 sequence length `T`가 길어지면 비용이 제곱으로 커진다. GPT-3처럼 context length가 2048이면 attention matrix 하나가 이미 `2048 x 2048`이다. layer와 head가 많아질수록 메모리 부담이 커진다.

그래서 긴 문맥 모델들은 sparse attention, sliding window attention, linear attention, memory mechanism 같은 변형을 사용하기도 한다. 하지만 기본 원리는 여전히 "각 위치가 다른 위치를 얼마나 참고할지 학습한다"는 것이다.

## 이 레포의 구현 파일과 연결

| 파일 | 연결되는 개념 |
| --- | --- |
| `src/bpe.py` | 텍스트를 token ID로 변환 |
| `src/embeddings.py` | token embedding + position embedding |
| `src/attention.py` | causal multi-head self-attention |
| `src/model.py` | LayerNorm, GELU, FFN, TransformerBlock, GPTModel |
| `src/train.py` | next-token prediction loss와 generation |
| `src/finetune.py` | GPT backbone 위에 classification head 추가 |

구현할 때 shape를 계속 추적하면 디버깅이 쉬워진다.

```text
input_ids: (B, T)
embedding output: (B, T, d_model)
attention output: (B, T, d_model)
block output: (B, T, d_model)
logits: (B, T, vocab_size)
```

## 이해 점검 질문

1. GPT에서 causal mask가 없으면 왜 next-token prediction이 무너지는가.
2. `Q K^T`의 shape가 왜 `(T, T)`가 되는가.
3. attention은 위치 사이를 섞고, FFN은 같은 위치의 벡터를 바꾼다는 말은 무슨 뜻인가.
4. GPT-3의 few-shot은 fine-tuning인가, in-context learning인가.
5. sequence length를 2배로 늘리면 attention memory는 대략 몇 배가 되는가.

## 참고

- Vaswani et al., 2017, [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Brown et al., 2020, [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

