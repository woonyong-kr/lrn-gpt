# GPT 블록 구현과 실험 가능한 구조

이 문서는 지금까지 직접 그린 흐름을 코드로 옮길 때 무엇을 파라미터로 두고, 어떤 값들이 학습되는 가중치인지 정리한다. 핵심은 GPT가 단순히 `Linear -> activation` 하나로 끝나는 신경망이 아니라, 토크나이저, 임베딩, 여러 Transformer block, 출력 head가 연결된 큰 신경망이라는 점이다.

## 전체 흐름

GPT의 순전파는 크게 다음 순서로 진행된다.

```text
텍스트
-> tokenizer.encode()
-> token IDs
-> token embedding + position embedding
-> input embedding X
-> Transformer Block 1
-> Transformer Block 2
-> ...
-> Transformer Block N
-> vocab head
-> logits
-> softmax
-> 다음 토큰 확률
```

학습할 때는 마지막 확률과 실제 다음 토큰을 비교해 loss를 만들고, 그 뒤에 역전파가 시작된다. 즉 우리가 그림으로 그린 attention, residual, layernorm, feed forward는 모두 loss 이전의 순전파 계산이다.

## Transformer block은 은닉층인가

일반적인 신경망에서 은닉층은 보통 다음처럼 이해한다.

```text
입력 -> Linear -> activation -> 출력
```

GPT에서는 이 자리에 Transformer block이 여러 개 쌓인다.

```text
입력 임베딩 X
-> Transformer Block 1
-> Transformer Block 2
-> ...
-> Transformer Block N
```

그래서 감각적으로는 `Transformer block = GPT에서 쓰는 큰 은닉층 단위`라고 봐도 된다. 다만 block 하나 안에는 여러 부품이 들어 있다.

```text
Transformer Block
1. Multi-Head Causal Self-Attention
2. Residual Add
3. LayerNorm
4. Feed Forward Network
5. Residual Add
6. LayerNorm
```

즉 일반 은닉층 하나보다 훨씬 큰 계산 묶음이라서 보통 layer 또는 block이라고 부른다.

## block마다 가중치는 별개다

Block 1과 Block 2는 같은 모양의 계산을 하지만 같은 가중치를 공유하지 않는다.

```text
Block 1:
W_Q, W_K, W_V, W_O, LN1_gamma, LN1_beta, W1, b1, W2, b2, LN2_gamma, LN2_beta

Block 2:
W_Q, W_K, W_V, W_O, LN1_gamma, LN1_beta, W1, b1, W2, b2, LN2_gamma, LN2_beta
```

이름은 비슷하지만 실제 파라미터는 block마다 다르다. 시작할 때는 보통 랜덤하게 초기화되고, 다음 토큰 예측 loss를 줄이는 방향으로 역전파와 optimizer가 조금씩 조정한다.

## 초기화와 시드

개발할 때는 다음 두 방식을 모두 지원하면 실험하기 좋다.

```text
1. 완전 랜덤 초기화
   실행할 때마다 초기 가중치가 달라진다.

2. 시드 고정 초기화
   seed=123처럼 고정하면 같은 랜덤 값이 반복되어 디버깅하기 좋다.
```

시드가 필요한 이유는 내가 만든 모델 구조가 맞는지 확인할 때 매번 다른 값이 나오면 비교가 어렵기 때문이다. 반대로 실제 실험에서는 여러 seed를 바꿔 보며 모델이 우연한 초기값에 너무 의존하지 않는지도 볼 수 있다.

## 폭과 깊이

LLM 크기를 키우는 축은 크게 두 가지다.

```text
폭(width):
embedding dimension, FFN hidden dimension, head 수처럼 한 번에 들고 있는 표현의 크기

깊이(depth):
Transformer block을 몇 개 쌓는지
```

폭이 커지면 한 토큰 벡터가 더 많은 정보를 담고, 한 block 안에서 더 많은 특징을 계산할 수 있다.

```text
d_model = 3
d_model = 768
d_model = 4096
```

깊이가 커지면 문장을 여러 번 다시 해석할 수 있다.

```text
Transformer Block 1
-> Transformer Block 2
-> Transformer Block 3
```

정리하면 다음과 같다.

```text
폭이 커진다
= 한 단계에서 더 많은 특징을 표현한다.

깊이가 커진다
= 여러 단계에 걸쳐 표현을 다시 해석한다.
```

둘 다 표현력을 키우지만 파라미터 수, 메모리, 연산량, 필요한 데이터가 함께 늘어난다.

## 토크나이저 학습과 모델 학습은 다르다

BPE 토크나이저 학습은 딥러닝 역전파가 아니다.

```text
대량 텍스트
-> byte sequence
-> 자주 등장하는 인접 pair 계산
-> 가장 자주 등장한 pair를 merge rule로 추가
-> vocab과 merge rule 완성
```

여기서 만들어지는 것은 `vocab + merge rules`다. 이건 텍스트를 token ID sequence로 바꾸기 위한 규칙이다.

모델 학습은 그 다음이다.

```text
텍스트
-> tokenizer.encode()
-> token IDs
-> input/target pair
-> GPT 순전파
-> 다음 토큰 예측 loss
-> 역전파
-> 모델 파라미터 업데이트
```

토크나이저가 만든 ID는 정답 벡터가 아니다. ID는 단지 식별자다. 모델은 그 ID로 embedding table에서 벡터를 조회하고, 다음 토큰 ID를 맞히도록 학습한다.

## Dataset과 DataLoader

다음 토큰 예측 학습에서는 input과 target을 한 칸씩 민다.

```text
token ids = [40, 827, 1500, 91]

input  = [40, 827, 1500]
target = [827, 1500, 91]
```

이 구조 덕분에 모델은 매 위치에서 다음 토큰을 맞히는 문제를 동시에 푼다.

```text
40을 보고 827을 맞힌다.
40, 827을 보고 1500을 맞힌다.
40, 827, 1500을 보고 91을 맞힌다.
```

`stride`는 학습 샘플을 얼마나 겹치게 자를지 정한다.

```text
stride = 1:
문맥이 많이 겹친다. 샘플 수가 많다.

stride = context_length:
문맥이 겹치지 않는다. 중복이 줄어든다.
```

이건 과적합을 직접 해결하는 마법은 아니지만, 같은 문맥을 과하게 반복 학습하는 정도를 조절할 수 있다.

## Input embedding

토큰 ID 자체는 모델이 계산하기 좋은 값이 아니다. 그래서 embedding table에서 벡터를 조회한다.

```text
token ID -> token embedding vector
position ID -> position embedding vector
```

그 다음 두 벡터를 더해 self-attention 입력을 만든다.

```text
x_i = E[token_id_i] + P[position_i]
```

같은 token ID는 같은 token embedding을 조회하지만, 위치가 다르면 position embedding이 달라져서 최종 input embedding도 달라진다.

## Attention 가중치

Self-attention에서는 각 입력 벡터 `x_i`에서 Q, K, V를 만든다.

```text
q_i = x_i @ W_Q
k_i = x_i @ W_K
v_i = x_i @ W_V
```

`W_Q`, `W_K`, `W_V`는 모두 학습되는 가중치다. 처음에는 랜덤이고, 다음 토큰 예측 학습 중 역전파로 조정된다.

attention score는 query와 key를 비교해 만든다.

```text
score_ij = (q_i · k_j) / sqrt(d_k)
```

GPT는 미래 토큰을 보면 안 되므로 causal mask를 적용한다.

```text
j > i 위치는 mask
```

그 다음 softmax로 attention weight를 만든다.

```text
A = softmax(masked_score)
```

마지막으로 value를 가중합해 context vector를 만든다.

```text
Z = A @ V
```

여기서 `Z`는 하나의 값이 아니라 sequence 전체에 대한 context vector matrix다.

## Multi-head attention

Multi-head attention은 같은 입력 `X`를 여러 attention head에 병렬로 넣는 구조다.

```text
head 0 -> Z0
head 1 -> Z1
head 2 -> Z2
head 3 -> Z3
```

각 head는 자기만의 `W_Q`, `W_K`, `W_V`를 가진다. 사람이 `head 0은 문법`, `head 1은 반복 단어`처럼 의미를 직접 지정하지 않는다. 다만 서로 다른 가중치로 시작하고 학습 중 서로 다른 패턴에 민감해질 수 있다.

각 head 출력은 concat한 뒤 `W_O`로 다시 model dimension에 맞춘다.

```text
MultiHeadOutput = concat(Z0, Z1, Z2, Z3) @ W_O
```

`W_O`도 학습되는 별도 가중치다.

## Residual과 LayerNorm

Attention을 통과하면 원래 입력 정보가 변형된다. 그래서 원래 입력을 다시 더한다.

```text
R = X + MultiHeadAttentionOutput
```

이게 residual connection이다. 의미는 다음과 같다.

```text
새로 계산한 문맥 정보는 반영하되,
원래 입력 정보도 잃지 않게 보존한다.
```

그 다음 LayerNorm을 적용한다.

```text
LN(r_i) = gamma * (r_i - mean(r_i)) / sqrt(var(r_i) + eps) + beta
```

LayerNorm은 토큰끼리 섞는 게 아니다. 한 토큰 벡터 내부의 dimension 값들을 평균 0, 분산 1 근처로 맞춰 다음 계산이 안정적으로 진행되게 한다.

`gamma`, `beta`도 학습되는 파라미터다. 우리가 손계산할 때는 이해를 위해 `gamma=1`, `beta=0`으로 둔 것이다.

LayerNorm이 두 번 나오면 gamma/beta도 두 벌이다.

```text
LN1_gamma, LN1_beta
LN2_gamma, LN2_beta
```

## Feed Forward Network

FFN은 attention처럼 토큰끼리 비교하지 않는다. 각 token vector를 독립적으로 같은 MLP에 통과시킨다.

```text
u = x @ W1 + b1
h = GELU(u)
out = h @ W2 + b2
```

일반 딥러닝에서 말하는 `W`, `b`와 같은 개념이다.

```text
Linear1: 6x3 @ 3x8 = 6x8
GELU
Linear2: 6x8 @ 8x3 = 6x3
```

왜 `3 -> 8 -> 3`처럼 넓혔다가 줄일까?

```text
1. 좁은 벡터를 넓은 특징 공간으로 펼친다.
2. GELU로 비선형성을 넣어 중요한 특징을 살리고 일부를 누른다.
3. residual을 위해 다시 원래 model dimension으로 되돌린다.
```

FFN의 중간 폭이 커질수록 각 토큰 위치에서 더 많은 특징을 표현할 수 있지만, 파라미터와 연산량도 늘어난다.

## 두 번째 residual과 block 출력

FFN 뒤에도 residual과 LayerNorm을 한 번 더 한다.

```text
S = LNOutput + FFNOutput
BlockOutput = LayerNorm(S)
```

이 값이 Transformer block 하나의 최종 출력이다.

다음 block이 있으면:

```text
BlockOutput -> 다음 Transformer Block 입력
```

마지막 block이면:

```text
BlockOutput -> vocab head -> logits -> softmax -> loss
```

## 구현에서 실험 가능하게 둘 설정

코드에서는 다음 값을 config로 바꾸기 쉽게 두는 것이 좋다.

```text
vocab_size
context_length
emb_dim 또는 d_model
n_heads
n_layers
drop_rate
qkv_bias
ffn_mult
activation_name
ffn_dropout_position
attention_impl
norm_eps
tie_embeddings
norm_first 또는 post_norm
seed
init_std
```

이렇게 하면 같은 코드로 다양한 실험을 할 수 있다.

```text
작은 디버그 모델:
emb_dim=32, n_heads=4, n_layers=2

조금 큰 실험 모델:
emb_dim=256, n_heads=8, n_layers=8

재현 가능한 디버그:
seed=123

완전 랜덤 실험:
seed=None
```

핵심은 구조를 하드코딩하지 않고, 폭과 깊이와 초기화 방식을 config로 조절하게 만드는 것이다.

## 교체 실험 지점

전체 Transformer 순서를 바꾸지 않고도 테스트할 수 있는 지점이 있다.

```text
Token/Position Embedding
-> TransformerBlock x N
   -> Attention
   -> Residual + LayerNorm
   -> FeedForward
   -> Residual + LayerNorm
-> final norm
-> lm head
```

여기서 구조를 흔들지 않고 바꿔볼 수 있는 값은 다음과 같다.

activation_name
: FFN 안의 `Linear1 -> activation -> Linear2`에서 activation만 바꾼다.
기본값은 `gelu`이고, `gelu_exact`, `quick_gelu`, `relu`, `silu`, `swish`, `mish`, `squared_relu`, `identity`, `swiglu`, `geglu`를 실험할 수 있다.
`swiglu`, `geglu`는 LLM에서 자주 쓰이는 gated FFN 계열이라 내부에서 `value * activation(gate)`를 계산하지만, 입력과 출력 shape는 그대로 유지한다.

ffn_mult
: FFN 내부 폭을 몇 배로 키울지 정한다.
예를 들어 `emb_dim=256`, `ffn_mult=4`이면 FFN 내부 차원은 `1024`가 된다.
표현력은 좋아질 수 있지만 파라미터 수와 연산량도 늘어난다.

ffn_dropout_position
: dropout을 어디에 둘지 정한다.
`after_output`은 기존 GPT식 기본 흐름이고, `after_activation`은 activation 직후에 dropout을 둔다.
`none`은 dropout module을 통과하지 않는다.

attention_impl
: attention 계산 방식을 바꾼다.
`manual`은 우리가 그린 `QK^T -> mask -> softmax -> V` 흐름을 그대로 보여준다.
`sdpa`는 PyTorch의 `scaled_dot_product_attention`을 사용한다.
학습 확인용으로 attention weight를 직접 보고 싶으면 `manual`이 더 좋고, 실제 속도 비교는 `sdpa`가 유리할 수 있다.

norm_eps
: LayerNorm에서 분모가 0에 가까워지는 것을 막는 작은 값이다.
보통 `1e-5`나 `1e-6`을 둔다.

tie_embeddings
: token embedding weight와 lm head weight를 공유할지 정한다.
켜면 입력 토큰을 벡터로 보는 표와, 마지막에 vocab 점수를 내는 표가 같은 파라미터를 바라본다.
작은 모델에서는 파라미터를 줄이고 GPT 계열의 weight tying 실험을 해볼 수 있다.

예시:

```python
config = {
    "vocab_size": 3000,
    "context_length": 128,
    "emb_dim": 256,
    "n_heads": 8,
    "n_layers": 6,
    "activation_name": "swiglu",
    "attention_impl": "sdpa",
    "ffn_mult": 4,
    "norm_first": False,
}
```
