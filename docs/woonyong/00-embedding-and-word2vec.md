# 벡터, 임베딩, word2vec

이 문서는 텍스트, 오디오, 이미지, 비디오가 어떻게 벡터로 바뀌는지 설명한다. Transformer의 token embedding을 이해하기 전에 보면 좋은 입문 문서다.

## 벡터로 바꾼다는 말의 의미

"텍스트, 오디오, 비디오를 벡터로 바꾼다"는 말은 원본 데이터를 신경망이 계산할 수 있는 숫자 표현으로 바꾼다는 뜻이다.

여기서 말하는 벡터는 보통 3차원 `Vector3`가 아니다. 실제 AI 모델에서 쓰는 embedding vector는 128차원, 768차원, 1024차원, 4096차원처럼 훨씬 큰 고차원 벡터인 경우가 많다.

```text
텍스트 샘플 -> 텍스트 encoder -> [0.12, -0.44, 0.91, ...]
오디오 샘플 -> 오디오 encoder -> [0.09, -0.38, 0.87, ...]
비디오 샘플 -> 비디오 encoder -> [0.15, -0.41, 0.89, ...]
```

원본을 벡터로 바꾼다고 해서 원본을 그대로 압축 파일처럼 보관한다는 뜻은 아니다. 모델이 task를 풀기에 중요한 특징을 숫자 좌표로 표현한다는 뜻이다.

## 원본 데이터도 이미 숫자다

겉으로는 텍스트, 오디오, 이미지, 비디오가 완전히 달라 보이지만 컴퓨터 안에서는 모두 숫자다.

텍스트는 tokenizer를 거쳐 token ID가 된다.

```text
"강아지가 달린다"
  -> tokenizer
  -> [3142, 872, 91, ...]
```

오디오는 시간에 따른 진폭 숫자다. 예를 들어 16kHz 오디오 1초는 16000개의 샘플 값을 가진다.

```text
audio waveform
  -> [0.01, 0.03, -0.02, ...]
```

이미지는 픽셀의 RGB 값이다.

```text
image
  -> height x width x 3
  -> each pixel = [R, G, B]
```

비디오는 이미지 프레임이 시간 순서로 쌓인 것이다.

```text
video
  -> frame_1, frame_2, frame_3, ...
  -> each frame = image
```

즉 모든 modality는 처음부터 숫자로 다룰 수 있다. 문제는 원본 숫자 자체가 너무 크고, 의미가 직접 드러나지 않는다는 점이다.

## Encoder가 의미 벡터를 만든다

원본 숫자를 그대로 쓰면 비교와 검색이 어렵다.

이미지 픽셀 하나의 의미는 이런 수준이다.

```text
이 픽셀은 R=120, G=80, B=60이다.
```

하지만 우리가 알고 싶은 것은 보통 이런 것이다.

```text
사람이 걷고 있음
강아지가 공을 물고 있음
실내 장면
웃는 얼굴
자동차 엔진 소리
질문에 대한 답변
```

Encoder는 원본 숫자 배열을 읽고 task에 필요한 추상 특징을 뽑아 embedding vector로 만든다.

```text
raw data
  -> encoder neural network
  -> embedding vector
```

이 벡터는 의미를 담은 좌표처럼 사용할 수 있다.

```text
"고양이 사진" embedding
실제 고양이 이미지 embedding
```

두 embedding이 같은 의미를 담고 있다면 벡터 공간에서 가깝게 만들 수 있다.

## 서로 다른 데이터가 같은 의미 공간에 놓이는 방식

멀티모달 모델에서는 텍스트, 이미지, 오디오, 비디오를 같은 embedding space에 놓고 비교할 수 있게 학습한다.

예를 들어 텍스트와 비디오가 있다.

```text
텍스트: "강아지가 공을 물고 달린다"
비디오: 강아지가 공을 물고 달리는 장면
```

각각 다른 encoder를 통과한다.

```text
text -> text encoder -> text embedding
video -> video encoder -> video embedding
```

학습 목표는 같은 의미의 쌍은 가깝게, 다른 의미의 쌍은 멀게 만드는 것이다.

```text
좋은 쌍:
이미지: 강아지 사진
텍스트: "강아지가 잔디밭에 있다"
-> 벡터 거리 가깝게

나쁜 쌍:
이미지: 강아지 사진
텍스트: "우주선이 발사된다"
-> 벡터 거리 멀게
```

이런 학습을 contrastive learning 방식으로 하는 경우가 많다. 같은 batch 안에서 맞는 쌍은 positive pair, 맞지 않는 조합은 negative pair로 둔다.

간단히 쓰면 다음과 같은 압력을 준다.

```text
similarity(image_embedding, matching_text_embedding) 높이기
similarity(image_embedding, wrong_text_embedding) 낮추기
```

이 과정을 많이 반복하면 모델은 픽셀, 소리, 토큰, 프레임 변화에서 의미 있는 패턴을 뽑는 법을 배운다.

## 벡터가 가까운지 어떻게 재는가

Embedding 사이의 유사도는 보통 cosine similarity나 dot product로 잰다.

Cosine similarity는 두 벡터의 방향이 얼마나 비슷한지 보는 값이다.

```text
cosine_similarity(a, b) = (a · b) / (||a|| ||b||)
```

방향이 비슷하면 1에 가까워지고, 관계가 약하면 0 근처, 반대 방향이면 -1에 가까워진다.

검색 시스템에서는 query를 embedding으로 바꾸고, 데이터베이스의 embedding들과 비교한다.

```text
query: "웃고 있는 사람"
  -> text embedding
  -> image/video embedding DB와 비교
  -> 가장 가까운 항목 반환
```

이게 벡터 검색의 기본 원리다.

## Word2vec란 무엇인가

word2vec는 단어를 벡터로 바꾸는 고전적인 embedding 방법이다. 2010년대 초반 NLP에서 매우 중요한 전환점이었다.

핵심 아이디어는 다음 문장으로 요약할 수 있다.

```text
비슷한 문맥에서 자주 등장하는 단어는 비슷한 의미를 가진다.
```

예를 들어 이런 문장들이 많다고 하자.

```text
나는 사과를 먹었다
나는 바나나를 먹었다
나는 포도를 먹었다
```

`사과`, `바나나`, `포도`는 비슷한 위치와 문맥에 자주 등장한다.

```text
나는 ___ 를 먹었다
```

그래서 word2vec는 이 단어들을 가까운 벡터로 배치한다.

```text
사과   -> [0.21, -0.13, 0.77, ...]
바나나 -> [0.24, -0.10, 0.75, ...]
포도   -> [0.20, -0.15, 0.79, ...]
자동차 -> [-0.61, 0.44, -0.08, ...]
```

과일 단어들은 서로 가까워지고, 자동차처럼 다른 문맥에 등장하는 단어는 멀어진다.

## CBOW와 Skip-gram

word2vec에는 대표적으로 두 가지 학습 방식이 있다.

CBOW는 주변 단어를 보고 가운데 단어를 맞힌다.

```text
입력: 나는 ___ 를 먹었다
정답: 사과
```

Skip-gram은 가운데 단어를 보고 주변 단어를 맞힌다.

```text
입력: 사과
정답: 나는, 를, 먹었다
```

두 방식 모두 "단어의 의미는 그 단어가 나타나는 주변 문맥에 들어 있다"는 distributional hypothesis에 기대고 있다.

학습이 끝나면 단어마다 하나의 고정된 벡터가 생긴다.

```text
word -> vector
```

## word2vec가 보여준 중요한 직관

word2vec의 유명한 예시는 다음과 같다.

```text
왕 - 남자 + 여자 ≈ 여왕
```

이 예시는 단어 벡터 공간에서 일부 의미 관계가 방향으로 표현될 수 있음을 보여준다.

다른 식으로 생각하면 다음과 같다.

```text
왕 -> 여왕 방향
남자 -> 여자 방향
```

두 방향이 비슷하게 정렬되어 있다면 벡터 연산으로 유추 관계가 어느 정도 드러난다.

물론 이것은 완벽한 논리 추론이 아니다. 대규모 문맥 통계에서 생긴 구조가 벡터 공간에 나타나는 것이다.

## word2vec의 한계

word2vec는 단어 하나에 보통 하나의 벡터를 준다. 그래서 문맥에 따라 뜻이 달라지는 단어를 잘 처리하지 못한다.

예를 들어 한국어 단어 `배`를 보자.

```text
"배를 먹었다"
"배를 탔다"
"배가 아프다"
```

여기서 `배`는 과일, 선박, 신체 부위일 수 있다. 하지만 word2vec에서는 `배`라는 단어가 하나의 고정 벡터를 갖는다.

```text
배 -> [0.31, -0.08, 0.44, ...]
```

이 벡터 하나에 여러 의미가 섞인다. 그래서 문맥별 의미 구분에 약하다.

## Transformer embedding은 무엇이 다른가

Transformer는 token embedding에서 시작하지만, layer를 통과하면서 문맥화된 contextual embedding을 만든다.

처음 token embedding:

```text
"배" token -> fixed token vector
```

Transformer layer 이후:

```text
"배를 먹었다"의 배 -> 과일 문맥을 반영한 vector
"배를 탔다"의 배 -> 선박 문맥을 반영한 vector
"배가 아프다"의 배 -> 신체 문맥을 반영한 vector
```

즉 같은 token이라도 주변 문맥에 따라 hidden state가 달라진다.

이 차이가 중요하다.

| 구분 | word2vec | Transformer |
| --- | --- | --- |
| 기본 단위 | 단어 | token/subword |
| 벡터 성격 | 단어별 고정 벡터 | 문맥에 따라 달라지는 벡터 |
| 문맥 처리 | 주변 단어 통계로 학습하지만 결과 벡터는 고정 | 입력 문장마다 self-attention으로 문맥 반영 |
| 다의어 처리 | 약함 | 강함 |
| 대표 사용 | 단어 유사도, 고전 NLP feature | LLM, BERT, GPT, embedding search |

## PyTorch Embedding 예제

토큰 임베딩을 가장 단순하게 보면 "토큰 ID로 embedding table의 한 행을 꺼내는 일"이다.

```python
import torch

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids))
```

여기서 `vocab_size = 6`은 토큰 ID가 `0`부터 `5`까지 있다는 뜻이다. `output_dim = 3`은 각 토큰을 3차원 벡터로 바꾼다는 뜻이다.

따라서 embedding table의 크기는 다음과 같다.

```text
vocab_size x output_dim = 6 x 3
```

개념적으로는 이런 표다.

```text
ID 0 -> [ ..., ..., ... ]
ID 1 -> [ ..., ..., ... ]
ID 2 -> [ ..., ..., ... ]
ID 3 -> [ ..., ..., ... ]
ID 4 -> [ ..., ..., ... ]
ID 5 -> [ ..., ..., ... ]
```

실행하면 `embedding_layer.weight`는 6행 3열짜리 matrix다.

```text
Parameter containing:
tensor([[ 0.3374, -0.1778, -0.1690],
        [ 0.9178,  1.5810,  1.3010],
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096]], requires_grad=True)
```

`embedding_layer(torch.tensor([3]))`은 3번 ID에 해당하는 행을 꺼낸다.

```text
tensor([[-0.4015, 0.9666, -1.1481]])
```

즉 다음과 같다.

```text
token ID 3 -> embedding table의 3번 행
```

`input_ids = [2, 3, 5, 1]`을 넣으면 각 ID의 행을 순서대로 꺼낸다.

```text
input IDs:
[2, 3, 5, 1]

embedding output:
[
  embedding[2],
  embedding[3],
  embedding[5],
  embedding[1],
]
```

이것이 token embedding이다.

```text
문자열 -> tokenizer -> token ID -> embedding table lookup -> vector
```

중요한 점은 토큰 ID 숫자 자체에는 의미가 없다는 것이다. `3`이라는 숫자가 의미를 담는 게 아니다. `3`은 embedding table의 3번 행을 가리키는 주소다. 의미는 학습이 진행되면서 그 행의 벡터 값에 들어간다.

처음 embedding weight는 보통 무작위로 시작한다.

```text
학습 전:
ID 3 -> 무작위 벡터

학습 후:
ID 3 -> 문맥과 task에 맞게 조정된 벡터
```

## output_dim은 왜 3일 수도 있고 더 클 수도 있는가

예제에서 `output_dim = 3`을 쓴 이유는 사람이 눈으로 보기 쉽게 하기 위해서다. 실제 LLM에서는 3차원으로는 너무 좁다. 토큰 하나가 문법, 의미, 어조, 주제, 위치적 역할, 다른 단어와의 관계 같은 많은 정보를 담아야 하기 때문이다.

```text
toy example:
token ID -> 3차원 vector

실제 GPT/BERT 계열:
token ID -> 768차원, 1024차원, 4096차원, 12288차원 ...
```

차원이 늘어난다는 말은 "모델이 단어를 설명할 수 있는 좌표축이 많아진다"는 뜻에 가깝다. 다만 이 좌표축은 사람이 직접 이름 붙인 축이 아니다.

예를 들어 사람이 설명할 때는 다음처럼 말할 수 있다.

```text
사과:
- 과일
- 음식
- 달다
- 빨갛다
- 먹다와 자주 등장

바나나:
- 과일
- 음식
- 달다
- 노랗다
- 먹다와 자주 등장

왕:
- 사람
- 권력
- 왕국
- 남성형 문맥
- 역사 문맥
```

하지만 실제 embedding vector의 각 차원이 반드시 이런 식으로 깔끔하게 대응되지는 않는다.

```text
사과 -> [0.21, -0.77, 1.02, 0.13, ...]
바나나 -> [0.19, -0.70, 0.95, 0.18, ...]
왕 -> [-0.31, 1.44, -0.12, 0.88, ...]
```

사람이 `0번 차원은 과일성`, `1번 차원은 권력성`처럼 직접 정하는 것이 아니다. 모델은 다음 토큰 예측, 분류, contrastive learning 같은 학습 목표를 잘 맞히도록 loss를 줄이는 과정에서 벡터 값을 조정한다. 그 결과 비슷한 문맥에서 쓰이는 토큰들은 벡터 공간에서 비슷한 방향이나 가까운 위치를 갖게 된다.

정리하면 다음과 같다.

```text
사람이 의미 축을 직접 설계하지 않는다.
학습 데이터의 문맥과 loss가 embedding weight를 움직인다.
그 결과 의미적으로 비슷한 토큰들이 가까워지는 경향이 생긴다.
```

## embedding weight도 역전파로 학습되는 파라미터다

`torch.nn.Embedding(vocab_size, output_dim)`이 만드는 `weight`는 그냥 lookup table처럼 보이지만, 실제로는 학습 가능한 파라미터다.

```text
embedding weight shape:
(vocab_size, output_dim)
```

예를 들어:

```text
vocab_size = 50000
output_dim = 768
```

이면 token embedding만 해도:

```text
50000 x 768 = 38,400,000
```

개의 학습 가능한 숫자를 가진다.

처음에는 보통 랜덤 값으로 시작한다.

```text
초기:
ID 391 -> 랜덤 벡터
```

훈련 중 모델이 어떤 입력에서 loss를 계산하면, 역전파는 embedding table에도 gradient를 보낸다.

```text
loss
  -> output layer
  -> Transformer blocks
  -> embedding output
  -> embedding weight의 특정 행
```

입력에 token ID `391`이 있었다면, `embedding.weight[391]`에 대한 gradient가 생긴다.

```text
embedding.weight[391] <- embedding.weight[391] - learning_rate * gradient
```

이것은 일반 신경망에서 `W`가 `dW`로 업데이트되는 것과 같은 개념이다. 차이는 embedding layer가 matrix multiplication처럼 보이지 않고, "ID로 행을 꺼내는 lookup"처럼 보인다는 점뿐이다.

## output_dim이 커진다는 것은 무엇을 얻는다는 뜻인가

`output_dim`이 커지면 토큰 하나를 표현하는 벡터의 길이가 길어진다.

```text
output_dim = 3:
사과 -> [0.2, -0.1, 0.8]

output_dim = 768:
사과 -> [0.2, -0.1, 0.8, ..., 0.03]
```

차원이 커지면 모델은 더 많은 패턴을 분리해서 표현할 여지를 얻는다. 단어의 의미, 문법적 성질, 자주 함께 나오는 단어, 문장 안 역할, domain, 말투 같은 여러 신호가 고차원 공간에 더 풍부하게 배치될 수 있다.

하지만 무조건 크게 하면 좋은 것은 아니다.

```text
장점:
- 더 복잡한 관계를 표현할 수 있음
- 큰 모델에서 더 풍부한 hidden state를 만들 수 있음
- attention과 FFN이 다룰 정보량이 커짐

비용:
- 파라미터 수 증가
- 메모리 사용량 증가
- 계산량 증가
- 데이터가 부족하면 과적합 위험 증가
```

그래서 `output_dim`은 "신경망이 더 정교해질 수 있는 용량"을 늘리는 설계값이다. 다만 정교함은 차원만으로 생기지 않는다. 충분한 데이터, 적절한 모델 크기, 안정적인 학습, 좋은 목적 함수가 함께 필요하다.

Transformer에서는 보통 이 embedding dimension이 곧 `d_model`이다.

```text
token embedding output: (B, T, d_model)
Transformer hidden state: (B, T, d_model)
```

즉 토큰이 처음 벡터가 되는 차원과 Transformer 내부에서 계속 들고 다니는 은닉 상태 차원이 같은 경우가 많다.

## MNIST 픽셀과 텍스트 토큰의 차이

MNIST 이미지는 처음부터 숫자 배열이다.

```text
28 x 28 grayscale image
-> 784 pixel values
```

각 픽셀 값은 어느 정도 직접적인 의미를 가진다.

```text
0.0 = 어두움
1.0 = 밝음
```

반면 텍스트의 token ID는 숫자이지만, 숫자 크기 자체에 의미가 없다.

```text
"apple" -> 391
"banana" -> 12482
```

여기서 `12482`가 `391`보다 더 크다고 해서 바나나가 사과보다 더 크거나 더 중요하다는 뜻은 아니다. token ID는 embedding table의 행 번호일 뿐이다.

```text
MNIST:
픽셀값 자체가 입력 특징

텍스트:
token ID는 주소
embedding vector가 입력 특징
```

그래서 텍스트 모델은 보통 다음 단계를 거친다.

```text
문자열
  -> tokenizer
  -> token IDs
  -> embedding lookup
  -> vectors
  -> Transformer
```

이미지 모델도 결국 내부에서는 특징 벡터를 학습하지만, 텍스트는 특히 token ID가 범주형 값이기 때문에 embedding layer가 거의 필수적으로 들어간다.

## 같은 단어가 여러 번 나오면 벡터를 복사하는가

문장 안에 같은 토큰이 여러 번 나오면 token ID는 같다.

```text
apple apple apple
-> [391, 391, 391]
```

embedding lookup을 하면 같은 embedding row를 여러 위치에서 가져온다.

```text
[
  E[391],
  E[391],
  E[391],
]
```

이때 새로운 단어를 vocabulary에 등록하는 것이 아니다. `E[391]`이라는 하나의 학습 파라미터 행을 이번 batch의 각 위치에 펼쳐 놓는 것에 가깝다.

계산 그래프 안에서는 각 위치에 벡터가 따로 있는 것처럼 보인다.

```text
position 0 -> E[391]
position 1 -> E[391]
position 2 -> E[391]
```

하지만 근원은 같은 embedding table row다. 그래서 역전파 때 같은 토큰이 여러 위치에서 쓰였다면, 그 위치들에서 온 gradient가 같은 row 업데이트에 합쳐진다.

```text
loss from position 0
loss from position 1
loss from position 2
  -> embedding.weight[391] update
```

이것을 "복사해서 새로 등록한다"라고 이해하면 조금 틀린다. 더 정확히는 다음과 같다.

```text
같은 token ID는 같은 embedding row를 조회한다.
문장 계산을 위해 각 위치에 같은 값을 펼쳐 놓는다.
vocab에 새 항목을 만들지는 않는다.
```

## token embedding과 position embedding은 왜 더하는가

토큰 임베딩만 있으면 같은 토큰은 언제나 같은 벡터로 시작한다.

```text
apple apple
-> [391, 391]
-> [E[391], E[391]]
```

이 상태만 보면 첫 번째 `apple`과 두 번째 `apple`은 구분되지 않는다. 그러나 Transformer는 문장 안 순서가 중요하다.

```text
개가 사람을 물었다
사람이 개를 물었다
```

등장하는 단어가 비슷해도 위치가 바뀌면 의미가 바뀐다. 그래서 각 위치에 position embedding을 더한다.

```text
첫 번째 apple = E[391] + P[0]
두 번째 apple = E[391] + P[1]
```

여기서 `E`는 token embedding table이고, `P`는 position embedding table이다.

```text
E[token_id] = 이 토큰이 무엇인가
P[position] = 이 토큰이 몇 번째 자리에 있는가
```

최종 입력은 두 정보를 합친 것이다.

```text
input_vector[t] = token_embedding[token_id[t]] + position_embedding[t]
```

다르게 말하면, 입력 벡터 자체는 두 종류의 벡터를 더해서 만든다.

```text
토큰 ID
  -> token embedding
  -> "이 토큰이 무엇인가" 벡터

위치 번호
  -> position embedding
  -> "이 토큰이 몇 번째 자리에 있는가" 벡터

token embedding + position embedding
  -> input embedding
  -> Transformer가 실제로 받는 입력 벡터
```

사용자가 말한 "알파 벡터"에 해당하는 값은 보통 `input embedding`이라고 부르면 된다. 새 token을 만든 것이 아니라, 이미 얻은 token vector와 position vector를 같은 차원에서 더해 **학습에 사용할 최종 입력 벡터**를 만든 것이다.

예를 들어 40번 token이 0번 위치에 있다면 다음과 같다.

```text
token ID 40 -> E[40]
position 0 -> P[0]

input embedding at position 0 = E[40] + P[0]
```

이 `E[40] + P[0]`가 Transformer block으로 들어간다. 학습 중 역전파가 일어나면 token embedding table의 `E[40]`도 조정될 수 있고, learned absolute position embedding을 쓰는 경우 position embedding table의 `P[0]`도 조정될 수 있다.

따라서 같은 토큰이라도 위치가 다르면 Transformer에 들어가는 첫 입력 벡터가 달라진다.

```text
같은 단어 + 다른 위치
-> 다른 입력 벡터
-> attention에서 다른 역할 가능
```

중요한 점은 position embedding이 token embedding을 "복사해서 새 단어로 등록"하는 것이 아니라는 점이다. 위치 벡터는 token vocabulary와 별개의 table에서 나온다.

```text
token embedding table:
vocab_size x d_model

position embedding table:
context_length x d_model
```

예를 들어 context length가 1024이고 `d_model = 768`이면 position embedding table은 다음 크기다.

```text
1024 x 768
```

0번째 위치, 1번째 위치, 2번째 위치마다 별도의 위치 벡터가 있고, 이 벡터를 해당 자리의 token vector에 더한다. 이것이 "이 토큰이 무엇인가"와 "어디에 있는가"를 동시에 알려주는 방식이다.

## Token embedding과 contextual embedding 구분

Transformer를 공부할 때는 두 embedding을 구분해야 한다.

Token embedding은 embedding table에서 token ID로 꺼낸 초기 벡터다.

```text
token_id -> embedding table lookup -> token vector
```

Contextual embedding은 Transformer block을 통과한 뒤의 hidden state다.

```text
token vector + position vector
  -> self-attention layers
  -> contextual hidden state
```

GPT에서 다음 토큰을 예측할 때 실제로 사용하는 것은 마지막 layer의 contextual hidden state다.

```text
hidden state -> LM head -> next-token logits
```

BERT에서 문장 분류를 할 때도 `[CLS]`의 contextual hidden state를 사용한다.

```text
[CLS] hidden state -> classifier -> label
```

## 멀티모달 embedding과 word2vec의 연결

word2vec는 단어를 의미 벡터 공간에 놓았다. 멀티모달 embedding은 이 생각을 더 넓힌 것이다.

```text
word2vec:
단어 -> 의미 벡터

multimodal embedding:
텍스트, 이미지, 오디오, 비디오 -> 의미 벡터
```

둘의 공통점은 비슷한 의미를 가까운 벡터로 만들고 싶다는 것이다.

차이는 입력의 종류와 encoder 구조다.

| 구분 | word2vec | 멀티모달 embedding |
| --- | --- | --- |
| 입력 | 단어와 주변 단어 | 텍스트, 이미지, 오디오, 비디오 |
| 학습 신호 | 주변 문맥 예측 | matching pair 가깝게, non-matching pair 멀게 |
| 결과 | 단어 벡터 | 샘플 또는 문장/이미지/오디오/비디오 벡터 |
| 활용 | 단어 유사도 | 검색, 추천, captioning, cross-modal retrieval |

## 이 레포와 연결

이 레포의 GPT 구현에서 embedding은 `src/embeddings.py`와 연결된다.

```text
input token ids
  -> token embedding
  -> position embedding
  -> Transformer blocks
  -> contextual hidden states
```

`src/bpe.py`는 문자열을 token ID로 바꾸는 역할을 한다. `src/embeddings.py`는 token ID를 vector로 바꾼다. `src/attention.py`와 `src/model.py`는 그 vector들이 서로 문맥을 주고받게 만든다.

즉 이 과제의 흐름은 다음 한 줄로 볼 수 있다.

```text
문자열 -> token ID -> token vector -> contextual vector -> next-token logits
```

## 이해 점검 질문

1. AI에서 말하는 embedding vector가 보통 3차원 벡터가 아닌 이유는 무엇인가.
2. 텍스트와 이미지를 같은 벡터 공간에 놓으려면 어떤 학습 신호가 필요한가.
3. word2vec의 "비슷한 문맥에 나오는 단어는 비슷하다"는 말은 어떤 뜻인가.
4. `배를 먹었다`와 `배를 탔다`에서 word2vec가 약한 이유는 무엇인가.
5. Transformer의 token embedding과 contextual embedding은 무엇이 다른가.
6. token ID 숫자 자체가 의미를 담는 것이 아니라면, 의미는 어디에 저장되는가.
7. 같은 token ID가 여러 번 등장할 때 embedding table에는 어떤 일이 일어나는가.
8. token embedding과 position embedding을 더하는 이유는 무엇인가.

## 참고

- Mikolov et al., 2013, [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- Vaswani et al., 2017, [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
