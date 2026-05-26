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

## 참고

- Mikolov et al., 2013, [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- Vaswani et al., 2017, [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

