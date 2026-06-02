# Byte Pair Encoding, BPE

이 문서는 지금까지 질문한 내용을 기준으로 BPE를 정리한다. 핵심은 "BPE는 학습 단계에서 자주 붙는 조각을 세고 병합 규칙을 만들며, 사용 단계에서는 이미 만들어진 규칙을 적용한다"는 점이다.

## 먼저 토큰 번호와 토큰 순서를 구분하기

단순 tokenizer 예제에서는 먼저 텍스트를 토큰 목록으로 쪼갰다.

```python
preprocessed = ["I", "HAD", "always", "thought", "Jack"]
```

이 리스트의 인덱스는 문장 안 위치다.

```text
위치 0: "I"
위치 1: "HAD"
위치 2: "always"
위치 3: "thought"
위치 4: "Jack"
```

하지만 이것은 토큰 ID가 아니다. 토큰 ID는 vocabulary에서 찾는 고유 번호다.

```python
vocab = {
    "HAD": 44,
    "I": 56,
    "Jack": 57,
    "always": 128,
    "thought": 999,
}
```

인코딩은 토큰 문자열을 vocab ID로 바꾸는 과정이다.

```text
["I", "HAD", "always", "thought", "Jack"]
-> [56, 44, 128, 999, 57]
```

정리하면 다음과 같다.

```text
리스트 인덱스 = 문장 안에서 몇 번째 토큰인가
토큰 ID = vocab에서 그 토큰에 붙은 고유 번호
```

같은 토큰은 문장 안 위치가 달라도 같은 ID를 가진다.

```text
I thought I knew
위치: 0 1 2 3
ID:   56 ... 56 ...
```

## 단순 vocab tokenizer의 번호 부여 방식

현재 예제의 단순 tokenizer는 BPE가 아니다. 다음 순서로 토큰 번호를 붙인다.

```python
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
```

즉 규칙은 다음과 같다.

```text
1. 원문을 토큰으로 쪼갠다.
2. 중복을 제거한다.
3. 문자열 기준으로 정렬한다.
4. 정렬된 순서대로 0, 1, 2, 3... 번호를 붙인다.
5. 특수 토큰은 마지막에 추가된다.
```

그래서 출력이 이런 식으로 나온다.

```text
('!', 0)
('"', 1)
("'", 2)
('(', 3)
...
('A', 11)
('Ah', 12)
```

이 번호 자체에는 의미가 들어 있지 않다. 번호는 embedding table을 찾기 위한 주소에 가깝다.

```text
token string -> token ID -> embedding table row -> vector
```

## 단순 tokenizer의 한계

`SimpleTokenizerV1`은 vocab에 없는 단어를 만나면 실패한다.

```text
vocab에 "Hello"가 없음
-> KeyError
```

`SimpleTokenizerV2`는 모르는 토큰을 `<|unk|>`로 바꾼다.

```text
Hello -> <|unk|>
palace -> <|unk|>
```

그래서 예제에서 다음처럼 출력될 수 있다.

```text
<|unk|>, do you like tea? <|unk|> the sunlit terraces of the <|unk|>.
```

이 현상은 vocabulary가 `the-verdict.txt`에서 본 토큰만 담고 있기 때문에 생긴다. 원문에 없는 단어는 `<|unk|>`로 치환된다.

BPE는 이 문제를 줄이기 위해 등장한 방식이다.

## BPE의 핵심 질문

BPE가 해결하려는 문제는 다음이다.

```text
단어 단위 tokenizer:
모르는 단어가 나오면 <|unk|>가 되기 쉽다.

문자 단위 tokenizer:
모든 단어를 표현할 수 있지만 sequence가 너무 길어진다.
```

BPE는 둘 사이의 절충이다.

```text
자주 나오는 조각은 크게 묶고,
낯선 단어는 작게 쪼개서 표현한다.
```

예를 들어 다음과 같이 표현할 수 있다.

```text
lower -> lower
lowest -> low + est
unbelievable -> un + believ + able
아주 낯선 문자열 -> 더 작은 문자 또는 byte 조각
```

## BPE는 언제 합치는가

BPE는 두 단계를 구분해야 한다.

1. tokenizer 학습 단계
2. tokenizer 사용 단계

학습 단계에서는 실제로 corpus를 잘게 쪼개고, 자주 붙어 나오는 인접 pair를 세고, 병합 규칙을 만든다.

사용 단계에서는 새로 세지 않는다. 이미 학습된 merge rule을 입력 텍스트에 적용한다.

이 차이가 가장 중요하다.

```text
BPE 학습:
빈도 세기 + 병합 규칙 만들기

BPE 사용:
이미 만든 병합 규칙 적용하기
```

## BPE 학습 단계

예를 들어 corpus가 다음과 같다고 하자.

```text
low
lower
lowest
newer
wider
```

처음에는 작은 단위에서 시작한다. 단순화를 위해 문자 단위로 보자.

```text
l o w
l o w e r
l o w e s t
n e w e r
w i d e r
```

이제 인접한 쌍을 센다.

```text
(l, o)
(o, w)
(w, e)
(e, r)
(e, s)
(s, t)
...
```

가장 자주 등장한 pair가 `(l, o)`라면 병합한다.

```text
l + o -> lo
```

corpus 표현은 바뀐다.

```text
lo w
lo w e r
lo w e s t
n e w e r
w i d e r
```

이제 `lo`는 하나의 새 토큰이다. 다음 단계에서는 `lo`도 다른 토큰과 pair를 이룬다.

```text
(lo, w)
(w, e)
(e, r)
...
```

만약 `(lo, w)`가 가장 자주 나오면 다시 병합한다.

```text
lo + w -> low
```

corpus는 이렇게 된다.

```text
low
low e r
low e s t
n e w e r
w i d e r
```

이 과정을 반복한다.

## 병합 기준은 무엇인가

기본 BPE의 병합 기준은 다음이다.

```text
현재 corpus 표현에서 가장 자주 등장하는 인접 pair를 병합한다.
```

사용자가 보통 정하는 것은 다음 중 하나다.

```text
vocab_size = 최종 vocabulary 크기
merge 횟수 = 병합을 몇 번 할지
min_frequency = 최소 몇 번 이상 등장해야 병합할지
```

실제로는 `vocab_size`를 정하는 경우가 많다.

```text
vocab_size = 50000
```

그러면 tokenizer 학습기는 목표 vocabulary 크기에 도달할 때까지 가장 자주 등장한 pair를 계속 병합한다.

`12회 이상 나오면 병합한다` 같은 규칙을 직접 기준으로 삼는 것이 핵심은 아니다. 그런 `min_frequency` 옵션을 둘 수는 있지만, 기본 흐름은 "가장 빈도 높은 pair부터 greedy하게 병합"이다.

## merge 횟수란 무엇인가

merge 1회는 새 토큰 1개를 만드는 일이다.

```text
l + o -> lo
```

이 병합으로 vocab에 `lo`가 추가된다.

```text
merge 1회 = 새 토큰 1개 추가
```

초기 vocab size가 1000개이고 목표 vocab size가 10000개라면, 대략 9000번의 merge가 필요하다.

```text
필요 merge 수 = 목표 vocab size - 초기 vocab size
```

특수 토큰도 고려하면 다음처럼 볼 수 있다.

```text
초기 byte vocab: 256개
special token: 2개
목표 vocab: 10000개
필요 merge 수: 10000 - 256 - 2 = 9742회
```

즉 사용자가 vocab size를 10000으로 잡으면, tokenizer는 새 토큰을 추가하며 10000개에 도달할 때까지 병합한다. 다만 corpus가 너무 작으면 의미 있는 pair가 부족할 수 있으므로 corpus 규모에 맞는 vocab size를 잡아야 한다.

## 병합된 토큰은 다시 카운트되는가

그렇다. 병합된 토큰은 그 순간부터 새 토큰으로 취급된다.

예를 들어:

```text
l + o -> lo
```

이후에는 `lo`가 단일 토큰이다.

```text
lo w e r
```

다음 pair count에서는 이런 pair가 등장한다.

```text
(lo, w)
(w, e)
(e, r)
```

그래서 `lo + w -> low`처럼 또 병합될 수 있다.

```text
lo + w -> low
low + er -> lower
```

이것이 BPE가 작은 조각에서 시작해 점점 큰 subword와 단어를 만드는 원리다.

## 카운트는 0부터 다시 세는가

개념적으로는 매 병합 후 현재 corpus 표현을 기준으로 pair 빈도를 다시 평가한다.

병합 전:

```text
l o w e r
pair: (l, o), (o, w), (w, e), (e, r)
```

`l + o -> lo` 병합 후:

```text
lo w e r
pair: (lo, w), (w, e), (e, r)
```

pair 목록 자체가 바뀌므로 다시 평가해야 한다.

하지만 실제 구현은 매번 전체 corpus를 처음부터 다시 세지 않는다. 너무 비싸기 때문이다. 보통은 병합이 일어난 주변 pair만 갱신한다.

예를 들어:

```text
a b c
```

에서 `a + b -> ab`를 병합하면 영향을 받는 pair는 주변이다.

```text
기존: (a, b), (b, c)
새것: (ab, c)
```

멀리 떨어진 pair들은 변하지 않는다. 그래서 효율적인 구현은 count table, heap, priority queue, 주변 갱신을 사용한다.

정리하면:

```text
개념적으로는 다시 센다고 보면 된다.
실제 구현은 바뀐 주변만 갱신해서 최적화한다.
```

## 병합 전에 체크한 항목은 다시 검사하는가

개념적으로는 다시 검사해야 한다. 병합 때문에 예전 pair count가 더 이상 맞지 않을 수 있기 때문이다.

예를 들어 `(o, w)`가 3회였더라도 `l + o -> lo`를 병합하면 어떤 위치의 `(o, w)`는 사라지고 `(lo, w)`가 생긴다.

최적화된 구현에서는 다음 방식 중 하나를 쓴다.

```text
1. 영향을 받은 pair count를 즉시 수정한다.
2. 우선순위 큐에서 꺼낼 때 현재 count와 맞는지 확인한다.
3. 오래된 stale entry라면 버리고 다음 후보를 본다.
```

그래서 "예전에 많이 나왔던 pair"가 현재도 많이 나오는지 계속 유효성 검사를 한다.

## BPE 사용 단계

학습이 끝나면 다음 두 가지가 저장된다.

```text
vocab
merge rules
```

새 문장이 들어왔을 때는 새로 빈도를 세지 않는다. 이미 저장된 merge rule을 적용한다.

예를 들어 merge rule이 다음과 같다고 하자.

```text
l + o -> lo
lo + w -> low
e + r -> er
low + er -> lower
```

입력:

```text
lower
```

처리:

```text
l o w e r
-> lo w e r
-> low e r
-> low er
-> lower
```

만약 전체 단어를 하나로 만드는 규칙이 없으면 가능한 만큼만 합친다.

```text
lowest
-> low est
```

아주 낯선 단어는 더 작은 조각으로 남는다.

```text
xqzv
-> x q z v
```

byte-level BPE라면 최악의 경우 byte 단위로 표현할 수 있다. 그래서 `<|unk|>`가 거의 필요 없다.

## 모르는 단어가 나오면 그때 학습하는가

아니다. 사용 단계에서는 새 단어가 나와도 tokenizer를 다시 학습하지 않는다.

모르는 단어가 들어오면 이미 학습된 merge rule로 합칠 수 있는 만큼 합치고, 나머지는 더 작은 조각으로 둔다.

```text
새 단어 입력
-> 기존 merge rule 적용
-> 가능한 subword/byte 조각으로 표현
```

즉 기준은 현재 입력 텍스트의 빈도가 아니다. 기준은 tokenizer를 학습할 때 사용한 corpus에서 얻은 병합 규칙이다.

## 메모리와 계산 비용

BPE 학습은 pair count가 필요하므로 메모리와 시간이 든다.

naive하게 하면 매우 비싸다.

```text
1. 전체 corpus pair 다시 세기
2. 가장 빈도 높은 pair 찾기
3. 전체 corpus에서 pair 치환
4. 목표 vocab size까지 반복
```

실제 구현은 여러 최적화를 사용한다.

```text
1. 같은 단어는 한 번만 저장하고 빈도만 기록한다.
2. pair count는 단어 빈도를 곱해서 계산한다.
3. 병합이 일어난 주변 pair만 갱신한다.
4. 가장 빈도 높은 pair를 빠르게 찾기 위해 heap/priority queue를 쓴다.
5. 큰 corpus는 streaming, shard, chunk 단위로 처리한다.
```

예를 들어 원문에 `lower`가 100000번 나온다면 100000번 저장하지 않고 다음처럼 저장할 수 있다.

```text
lower: 100000
lowest: 42000
newer: 18000
```

pair count는 빈도를 반영해 계산한다.

```text
lower = l o w e r, 빈도 100000

(l, o) += 100000
(o, w) += 100000
(w, e) += 100000
(e, r) += 100000
```

그래도 BPE tokenizer 학습은 공짜가 아니다. 다만 LLM 본체 pretraining에 비하면 훨씬 작고, 한 번 학습한 뒤 계속 재사용한다.

## byte-level BPE

GPT 계열 tokenizer는 byte-level BPE를 사용한다. 문자보다 더 낮은 UTF-8 byte 단위에서 시작한다.

```text
텍스트
-> UTF-8 byte
-> byte pair merge
-> token ID
```

byte-level BPE의 장점은 거의 모든 문자열을 표현할 수 있다는 점이다.

```text
영어
한국어
이모지
특수문자
깨진 문자열 일부
```

모든 입력은 byte로 표현 가능하므로, vocabulary에 완전한 단어가 없어도 byte 조각으로 분해할 수 있다.

## tiktoken과 BPE

`tiktoken`은 OpenAI GPT 계열 tokenizer를 빠르게 실행하기 위한 라이브러리다. Python에서 사용하지만 내부 핵심 구현은 Rust 기반으로 매우 빠르게 동작한다.

예를 들어:

```python
import tiktoken

enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("Hello, world!")
print(ids)
```

출력 예:

```text
[15496, 11, 995, 0]
```

여기서 숫자들은 단순 정렬 vocab의 번호가 아니다. 이미 학습된 BPE vocabulary와 merge rule에 따라 정해진 token ID다.

## 전체 흐름 요약

```text
BPE 학습 단계:
작은 단위로 시작
-> 인접 pair 빈도 계산
-> 가장 자주 나온 pair 병합
-> 새 토큰을 vocab에 추가
-> 목표 vocab size까지 반복
-> vocab과 merge rules 저장

BPE 사용 단계:
새 텍스트 입력
-> 작은 단위로 분해
-> 저장된 merge rules 적용
-> token ID sequence 출력
```

## 질문별 짧은 답

### 애초에 단어를 전부 쪼개서 토큰화하면서 합치는가

학습 단계에서는 그렇다. 작은 단위에서 시작해 자주 나오는 pair를 합친다.

### 모르는 단어가 나오면 그때 나눠서 토큰화하는가

사용 단계에서는 기존 merge rule로 가능한 만큼 합치고, 안 되는 부분은 작은 단위로 남긴다. 새로 학습하지는 않는다.

### 병합 기준은 사용자가 정하는가

사용자는 보통 `vocab_size`, `merge 횟수`, 선택적으로 `min_frequency`를 정한다. 실제 병합 후보는 현재 corpus 표현에서 가장 빈도 높은 pair가 선택된다.

### 병합된 단어는 또 카운트되는가

그렇다. 병합된 새 토큰도 다른 토큰과 pair를 이루며 다음 count 대상이 된다.

### 병합 전 항목은 다시 검사하는가

개념적으로는 다시 검사한다. 실제 구현은 바뀐 주변 count만 갱신하거나 stale count를 버리는 방식으로 최적화한다.

### vocab size를 10000으로 잡으면 10000개가 될 때까지 병합하는가

대체로 그렇다. 초기 vocab과 특수 토큰을 제외하고, 목표 크기에 도달할 때까지 merge로 새 토큰을 추가한다.

## 이해 점검 질문

1. BPE 학습 단계와 사용 단계의 가장 큰 차이는 무엇인가.
2. `merge 횟수 = 새 토큰을 추가한 횟수`라는 말은 무슨 뜻인가.
3. 병합된 `lo`가 다시 `w`와 병합될 수 있는 이유는 무엇인가.
4. byte-level BPE가 `<|unk|>`를 줄일 수 있는 이유는 무엇인가.
5. 단순 vocab tokenizer와 BPE tokenizer의 가장 큰 차이는 무엇인가.

## 참고

- Sennrich et al., 2015, [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- OpenAI, [tiktoken](https://github.com/openai/tiktoken)
- Hugging Face, [Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5)

