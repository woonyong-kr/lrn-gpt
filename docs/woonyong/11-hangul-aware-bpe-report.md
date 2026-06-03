# 한글 인식 BPE 토크나이저 보고서

이 문서는 지금까지의 BPE 질문과 프로젝트 실험 기록을 바탕으로, 한글을 UTF-8 byte 단위로 그대로 두는 방식, 260개 기본 byte 토큰만 쓰는 방식, 제한 없이 BPE merge를 계속하는 방식, 그리고 한글 한 글자까지는 강제 병합하고 그 이상은 빈도 기준으로만 병합하는 방식을 비교한다.

핵심 결론은 다음이다.

```text
한글 UTF-8 byte 3개 -> 한글 한 글자 token:
무조건 병합하는 편이 효율적일 가능성이 높다.

한글 한 글자 이상 -> 더 긴 한글 조각:
최소 등장 횟수 n 이상일 때만 병합하는 편이 안전하다.

260개 byte token만 사용하는 방식:
각 byte token은 골고루 학습되지만, 문장 표현이 너무 길어지고 의미 단위 학습이 비효율적이다.

중복 제한 없이 merge하는 방식:
자주 쓰이는 조각에는 유리하지만, rare long token이 많아지면 embedding/LM head row가 제대로 학습되지 않고 과적합 또는 불안정한 일반화를 만들 수 있다.
```

이 문서의 주장은 아직 최종 증명이 아니다. 기존 실험은 tokenizer만 완전히 통제한 한글 tokenizer 비교 실험이 아니라, vocab size, tokenization scale, model size, context, dataset이 섞인 기록이다. 따라서 아래 내용은 "기존 증거로부터 가장 그럴듯하게 추론되는 설계 방향"으로 읽어야 한다.

## 현재 데이터셋 상황

프로젝트에는 여러 데이터셋이 함께 있다.

| 데이터셋 | 성격 | 한글 비율 |
| --- | --- | ---: |
| `data/nsmc_lm_train.txt` | NAVER 영화 리뷰 NSMC | 한글 중심 |
| `data/obsidian_os_pintos_ai_lm_train.txt` | Obsidian OS/PintOS/AI 문서 | 한글 + 영어/코드 혼합 |
| `data/obsidian_llm_10x_lm_train.txt` | Obsidian 기반 10x LLM/AI corpus | 영어/코드/LLM 자료 중심 + 일부 한글 |
| `src/learning/the-verdict.txt` | 영어 소설 샘플 | 영어 |

## 이 보고서가 기준으로 삼는 데이터셋

이 보고서의 주 기준 데이터셋은 `data/obsidian_llm_10x_lm_train.txt`와 `data/obsidian_llm_10x_lm_val.txt`다.

```text
primary train:
data/obsidian_llm_10x_lm_train.txt

primary validation:
data/obsidian_llm_10x_lm_val.txt
```

즉 이 보고서는 NSMC 영화 리뷰 모델을 만들기 위한 tokenizer 보고서가 아니다. 목표는 사용자가 만든 Obsidian 기반 OS/AI/LLM 10x corpus를 더 잘 학습하기 위한 tokenizer 설계다.

각 데이터셋의 역할은 다음처럼 구분한다.

| 데이터셋 | 이 보고서에서의 역할 | 해석 기준 |
| --- | --- | --- |
| `data/obsidian_llm_10x_lm_train.txt` | 주 학습 corpus | tokenizer 학습과 모델 학습의 기본 기준 |
| `data/obsidian_llm_10x_lm_val.txt` | 주 검증 corpus | 전체 목표 corpus 성능을 판단하는 1차 기준 |
| `data/obsidian_os_pintos_ai_lm_train.txt` | 이전/부분 Obsidian corpus | 10x corpus의 출발점 또는 비교 맥락 |
| `data/nsmc_lm_val.txt` | 외부 한글 stress test | 한글 일반화 확인용이며 목표 corpus 성능을 대체하지 않음 |
| `src/learning/the-verdict.txt` | 영어 smoke sample | 과제 학습용 fallback이며 이 보고서의 주 기준이 아님 |

따라서 한글 인식 BPE를 검증할 때 dataset을 NSMC로 바꿔서 학습하면 안 된다. 그러면 tokenizer 효과와 dataset 효과가 섞인다. 올바른 비교는 `obsidian_llm_10x` train/val을 고정하고 tokenizer만 바꾸는 것이다.

```text
좋은 비교:
obsidian_llm_10x 고정 + standard BPE
obsidian_llm_10x 고정 + Hangul-aware BPE

나쁜 비교:
obsidian_llm_10x 학습 결과
NSMC 학습 결과

이 경우 dataset 자체가 달라져 tokenizer 효과를 분리할 수 없다.
```

NSMC는 한글 비율이 매우 높은 외부 평가셋으로 남겨두는 편이 좋다. NSMC에서 좋아지는 것은 "순수 한글 영화 리뷰에 강하다"는 신호이고, `obsidian_llm_10x` validation에서 좋아지는 것은 "실제 목표 corpus에 도움이 된다"는 신호다. 이 둘은 서로 다른 주장이다.

전체 파일 기준으로 측정한 비율은 다음과 같다.

| 파일 | 전체 문자 | 한글 문자 | 한글/공백 제외 | 한글/문자숫자 | ASCII 알파벳/문자숫자 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `obsidian_llm_10x_lm_train.txt` | 14,898,445 | 829,056 | 6.28% | 7.15% | 85.09% |
| `obsidian_llm_10x_lm_val.txt` | 1,321,972 | 73,956 | 6.36% | 7.27% | 85.36% |
| `obsidian_os_pintos_ai_lm_train.txt` | 4,474,450 | 862,295 | 23.75% | 30.19% | 64.76% |
| `nsmc_lm_train.txt` | 1,379,486 | 977,036 | 89.56% | 97.86% | 1.02% |

따라서 현재 목표가 `obsidian_llm_10x`라면, 이 corpus는 한글 중심 corpus가 아니라 혼합 corpus다. 한글 tokenizer 설계를 적용하더라도 "한글 전용 모드"가 아니라 "한글은 글자 단위를 보호하고, 영어/코드/기호는 기존 BPE처럼 처리하는 하이브리드 모드"가 더 적절하다.

## 현재 BPE 구현의 출발점

현재 `src/bpe.py`의 token ID 구조는 다음과 같다.

```text
0~3: special tokens
4~259: raw UTF-8 byte tokens
260~: BPE merge tokens
```

즉 `vocab_size=260`이면 special token 4개와 byte token 256개만 존재한다. 이 경우 BPE merge token은 하나도 없다.

이 방식의 장점은 모든 UTF-8 텍스트를 표현할 수 있다는 점이다. 한국어, 영어, 코드, 이모지, 특수문자 모두 byte sequence로 표현할 수 있다. 하지만 한글의 경우 일반적인 완성형 한글 음절은 UTF-8에서 3 byte로 표현된다. 예를 들어 개념적으로는 다음과 같은 문제가 생긴다.

```text
한글 한 글자:
실제 언어 단위로는 1글자

byte-level tokenizer:
모델 입력에서는 3개 token
```

따라서 `260` no-merge tokenizer는 "모든 것을 표현한다"는 면에서는 강하지만, "언어 단위에 가까운 표현을 효율적으로 학습한다"는 면에서는 손해가 크다.

## token-level loss가 비교를 어렵게 만드는 이유

언어 모델의 token-level loss는 다음 토큰 하나를 맞히는 평균 난이도다.

```text
token-level loss = - average log P(next token | previous tokens)
```

문제는 tokenizer가 바뀌면 "토큰 하나"의 크기가 달라진다는 점이다.

```text
byte tokenizer:
한글 한 글자 = 보통 3 token

한글 글자 tokenizer:
한글 한 글자 = 1 token

BPE tokenizer:
자주 나오는 단어 또는 조각 = 1 token일 수 있음
```

따라서 서로 다른 tokenizer의 token-level loss는 직접 비교하면 안 된다. byte token 하나를 맞히는 loss와, 두 글자짜리 단어 조각 하나를 맞히는 loss는 같은 단위의 문제가 아니다.

비교 가능한 단위로 바꾸려면 문자 기준으로 환산해야 한다.

```text
nats_per_char = token_loss * tokens_per_char
bits_per_char = token_loss * tokens_per_char / log(2)
```

기존 `docs/train/tokenizer_profile.md`도 같은 규칙을 둔다.

```text
1. 같은 tokenizer/vocab이면 final_val_loss, gap, overfit_score를 직접 비교한다.
2. tokenizer/vocab이 다르면 final_val_loss 단독 비교를 금지한다.
3. vocab/BPE 실험의 primary metric은 final_val_nats_per_char 또는 생성 샘플 품질이다.
```

## 기존 실험에서 보이는 tokenizer 상관

수동 HY 실험의 vocab size 관련 결과는 다음과 같다.

| 실험 | vocab_size | 검증 token 수 | 검증 token/문자 | 검증 문자/token | token 기준 검증 loss | 검증 nats/문자 | 검증 bits/문자 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E04 | 2000 | 78,686 | 0.652671 | 1.532166 | 4.811352 | 3.140229 | 4.530393 |
| E05 | 4000 | 65,634 | 0.544409 | 1.836853 | 5.638393 | 3.069594 | 4.428488 |
| E06 | 5000 | 62,493 | 0.518356 | 1.929176 | 5.882533 | 3.049246 | 4.399133 |

이 표는 매우 중요하다. token 기준 검증 loss만 보면 `vocab_size=2000`이 가장 좋아 보인다.

```text
token loss:
2000 vocab: 4.811
4000 vocab: 5.638
5000 vocab: 5.883
```

하지만 문자당 loss로 보면 방향이 달라진다.

```text
nats per char:
2000 vocab: 3.140
4000 vocab: 3.070
5000 vocab: 3.049
```

즉 vocab이 커질수록 token 하나는 더 큰 조각을 담당하므로 token loss는 커져 보인다. 그러나 같은 문자를 표현하는 token 수가 줄어들기 때문에 문자 기준 성능은 좋아질 수 있다.

상관 보고서의 관찰 상관도 이 해석을 지지한다.

| 관계 | 피어슨 r | 해석 |
| --- | ---: | --- |
| `val_chars_per_token -> final_val_loss` | 0.9419 | 토큰이 길수록 token-level loss가 커지는 경향 |
| `val_tokens_per_char -> final_val_loss` | -0.9394 | 문자당 token 수가 줄수록 token-level loss가 커지는 경향 |
| `vocab_size -> final_val_loss` | 0.9178 | vocab이 클수록 token-level loss가 커지는 경향 |
| `vocab_size -> overfit_score` | 0.5832 | vocab 증가가 과적합 신호와도 어느 정도 연결될 수 있음 |

여기서 중요한 것은 마지막 행이다. vocab이 커지는 것은 항상 나쁜 것이 아니다. 그러나 작은 corpus 또는 낮은 min_frequency에서 vocab을 너무 키우면 tail token이 늘고, 그 token들이 충분히 학습되지 않을 위험이 있다.

## 260개 token만 쓰는 방식의 이득

`vocab_size=260`은 merge가 없는 byte tokenizer다. 이 방식에는 분명한 장점이 있다.

첫째, 모든 기본 byte token이 자주 등장한다. 영어, 한글, 코드, 공백, 기호를 모두 byte로 쪼개므로 256개 byte row 대부분은 반복적으로 update를 받는다. embedding table과 LM head에서 아예 한두 번만 등장하는 token row가 생길 위험이 작다.

둘째, vocab이 작기 때문에 softmax 분류 문제가 작다. 무작위 초기 상태의 cross entropy scale도 vocab 크기에 의해 달라진다.

```text
무작위 예측 baseline:
vocab 260: log(260) ~= 5.56
vocab 600: log(600) ~= 6.40
vocab 4000: log(4000) ~= 8.29
```

따라서 260 tokenizer의 token-level loss가 낮아 보이는 것은 상당 부분 자연스러운 일이다. 출력 class 수가 작기 때문이다.

셋째, OOV 문제가 사실상 없다. 어떤 문자가 들어와도 UTF-8 byte sequence로 표현할 수 있다.

넷째, rare long token이 없으므로 "한 번 본 특이한 긴 조각을 token 하나로 외우는 문제"는 줄어든다. 이 점은 작은 dataset에서는 regularization처럼 보일 수 있다.

## 260개 token만 쓰는 방식의 비용

하지만 비용도 크다.

첫째, 문장 길이가 크게 늘어난다. 한글 한 글자는 보통 3 byte이므로, 한글 문장은 글자 단위 tokenizer보다 token 수가 약 3배 길어진다. 영어도 자주 나오는 단어를 merge하지 못하므로 단어 단위보다 훨씬 길다.

둘째, 같은 `context_length`가 담는 실제 문맥이 짧아진다.

```text
context_length = 128 token

byte-level 한글:
대략 40여 글자 분량

한글 글자 단위:
대략 128글자 분량

BPE 단어 조각 단위:
그보다 더 긴 실제 문맥 가능
```

GPT의 attention은 token 위치를 기준으로 동작한다. byte tokenizer에서는 모델이 긴 의미 문맥을 보기 전에 UTF-8 byte 조합을 복원하는 데 많은 위치를 소비한다.

셋째, 학습 신호가 언어 단위가 아니라 encoding 단위에 묶인다. 모델은 `영`, `화`, `학`, `습` 같은 단위를 바로 보지 못하고, byte prefix와 continuation byte의 패턴을 먼저 배워야 한다. 이 패턴은 텍스트 복원에는 필요하지만 의미 학습에는 우회로다.

넷째, 같은 token budget에서 처리하는 문자 수가 줄어든다.

```text
고정 token step으로 학습:
byte tokenizer는 더 적은 실제 문자와 더 짧은 문맥을 본다.

고정 character corpus를 epoch 기준으로 학습:
byte tokenizer는 token 수가 많아져 step 수와 계산량이 늘어난다.
```

따라서 260 tokenizer는 "각 token row가 골고루 학습된다"는 장점이 있지만, 그 대가로 sequence가 길어지고 언어적 추상화가 늦어진다.

## 제한 없이 merge하는 방식의 이득

반대로 BPE merge를 많이 하면 자주 등장하는 조각이 하나의 token이 된다.

예를 들어 한글 corpus에서는 다음과 같은 조각이 자주 나올 수 있다.

```text
영화
재미
정말
합니다
습니다
ㅋㅋ
별로
감동
```

이런 조각을 하나의 token으로 만들면 장점이 있다.

첫째, sequence가 짧아진다. 같은 `context_length`에서 더 많은 실제 문자를 볼 수 있다.

둘째, 자주 함께 등장하는 의미 단위를 embedding row 하나로 학습할 수 있다. `영`과 `화`를 항상 따로 조합하지 않고 `영화`라는 조각을 바로 사용할 수 있다.

셋째, attention 비용이 줄 수 있다. 같은 문자 길이를 표현하는 token 수가 줄어들면 attention의 길이 부담이 줄어든다.

넷째, 문자당 loss가 좋아질 수 있다. HY 실험에서 vocab이 커질수록 token loss는 커졌지만 nats/char는 개선되는 흐름이 있었다.

## 제한 없이 merge하는 방식의 비용

문제는 BPE가 greedy하게 vocab을 채울 때, 충분한 제한이 없으면 드문 조각까지 token으로 만들 수 있다는 점이다.

```text
vocab_size가 큼
min_frequency가 낮음
corpus가 작음

=> 아주 드문 pair도 merge token이 됨
```

이때 생기는 문제는 네가 짚은 직관과 거의 같다.

첫째, tail token이 생긴다. 어떤 token은 train corpus에서 몇 번밖에 등장하지 않는다. 그 token의 embedding row와 LM head row는 gradient update를 충분히 받지 못한다.

둘째, token row의 품질 차이가 커진다.

```text
빈번한 token:
많은 update를 받아 안정적인 embedding을 가짐

희소 token:
거의 random initialization에 가까운 상태로 남거나, 몇 번의 예시에 과하게 맞춰짐
```

셋째, 출력 분류 문제가 커진다. vocab이 커지면 softmax class 수가 늘어난다. 모델은 더 많은 후보 중 하나를 맞혀야 하고, rare class의 확률 보정도 더 어려워진다.

넷째, parameter 수가 늘어난다.

```text
embedding parameter 증가:
vocab_size * emb_dim

LM head parameter 증가:
tie_embeddings=False이면 vocab_size * emb_dim이 추가로 증가
tie_embeddings=True여도 출력 class 수는 여전히 증가
```

다섯째, 과적합 위험이 커질 수 있다. rare long token은 일종의 고유 ID처럼 작동할 수 있다. 모델이 어떤 드문 phrase를 일반적인 문자 조합으로 배우는 대신, 그 phrase token ID 자체를 train 문맥과 함께 외워버릴 수 있다.

이 경우 train loss는 내려가지만 validation loss는 충분히 내려가지 않는다. 프로젝트의 `overfit_score` 관점에서는 다음 신호가 나타날 수 있다.

```text
final_train_loss 크게 감소
final_val_loss는 덜 감소
final_generalization_gap 증가
train_val_improvement_gap 증가
overfit_score 증가
```

따라서 merge를 많이 하는 것이 항상 나쁜 것은 아니지만, 빈도 제한 없이 vocab을 채우는 것은 위험하다.

## 한글 한 글자까지 강제 merge하는 이유

한글의 경우 "byte 3개를 한 글자로 합치는 것"과 "한 글자 이상을 단어 조각으로 합치는 것"은 성격이 다르다.

```text
byte -> 한글 한 글자:
encoding 복원 단계

한글 한 글자 -> 여러 글자 조각:
언어적 빈도 압축 단계
```

첫 단계는 빈도 기반으로 볼 필요가 적다. 한글 완성형 음절은 UTF-8에서 3 byte로 표현되지만, 사용자는 그것을 하나의 글자로 인식한다. 모델도 최소한 그 단위는 바로 볼 수 있게 해주는 편이 자연스럽다.

따라서 한글 byte를 한 글자로 만드는 과정은 다음처럼 보는 것이 좋다.

```text
빈도 기반 merge:
자주 나오는 조각만 합침

한글 글자 보호 merge:
UTF-8 encoding 단위를 Unicode 글자 단위로 복원
```

이 규칙은 260 tokenizer의 장점과 BPE의 장점을 절충한다.

```text
260 tokenizer의 장점:
희소 long token을 만들지 않음

한글 글자 보호의 장점:
한글 1글자를 byte 3개로 낭비하지 않음
```

## 한글 한 글자 이상은 왜 빈도 제한이 필요한가

한글 한 글자 이후부터는 이야기가 달라진다.

```text
영 + 화 -> 영화
재 + 미 -> 재미
별 + 로 -> 별로
```

이런 merge는 corpus에서 자주 나온다면 매우 유용하다. 그러나 모든 2글자 조합을 무조건 token으로 만들면 문제가 생긴다.

한글 완성형 음절의 가능한 조합 수는 매우 크다. 실제 corpus에는 우연히 한두 번 등장한 2글자, 3글자 조합이 많다. 이들을 모두 token으로 만들면 rare token이 폭증한다.

따라서 한글 글자 이상 merge에는 최소 빈도 `n`이 필요하다.

추천 규칙은 다음이다.

```text
1. 한글 UTF-8 byte sequence는 한글 한 글자까지 무조건 묶는다.
2. merge 결과가 한글 2글자 이상이면 pair 빈도가 n 이상일 때만 병합한다.
3. n의 기본값은 corpus의 한글 문자 수에 따라 정한다.
```

기본값은 다음 정도가 적절하다.

```text
n = max(5, min(50, round(total_hangul_chars / 50_000)))
```

이 공식을 적용하면 다음과 같다.

| corpus | 한글 문자 수 | 추천 n |
| --- | ---: | ---: |
| NSMC train | 977,036 | 20 |
| Obsidian LLM 10x train | 829,056 | 17, 실험 기본값으로는 20 |
| 300k char sample, 한글 6.3% 가정 | 약 18,900 | 5 |

실험 기본값으로는 `n=20`을 추천한다. 다만 `obsidian_llm_10x`를 20k 또는 300k character limit으로 잘라서 빠른 실험을 할 때는 한글 문자 수가 적어지므로 `n=5`도 함께 비교하는 것이 좋다.

## 제안하는 하이브리드 토크나이저 규칙

현재 목표 corpus가 `obsidian_llm_10x`라면 한글 중심이 아니라 영어/코드 중심의 혼합 corpus다. 따라서 다음 규칙이 좋다.

```text
mode: hybrid

1. 전체 입력은 Unicode scalar 기준으로 안전하게 읽는다.
2. 한글 완성형 음절과 한글 자모는 최소 1글자 token으로 보호한다.
3. 한글 글자 내부를 UTF-8 byte token 3개로 남기지 않는다.
4. 영어, 숫자, 코드, 기호는 기존 byte-level BPE 흐름을 유지한다.
5. 한글 2글자 이상 merge는 hangul_pair_min_frequency 이상일 때만 허용한다.
6. 너무 긴 한글 token을 막기 위해 max_hangul_chars_per_token을 둔다.
```

초기 실험값은 다음을 추천한다.

```text
force_hangul_char_token = true
hangul_pair_min_frequency = 20
max_hangul_chars_per_token = 4
base_min_frequency = 2 또는 5
vocab_size = 2000 또는 4000
```

한글 전용 NSMC 실험이라면 다음도 비교할 수 있다.

```text
hangul_pair_min_frequency = 5
hangul_pair_min_frequency = 20
hangul_pair_min_frequency = 50
max_hangul_chars_per_token = 2
max_hangul_chars_per_token = 4
```

`max_hangul_chars_per_token=2`는 사용자가 제안한 "최대 2글자 위주" 가설을 검증하기 좋다. 하지만 기본값으로는 `4`가 더 안전하다. 한국어에는 `합니다`, `습니다`, `재밌다`, `영화가`, `아니라`처럼 3~4글자 빈번 조각도 많기 때문이다.

## 문장 표현 길이의 trade-off

토크나이저는 모델이 보는 문장의 길이를 바꾼다.

### 짧은 token 단위의 이득

byte 또는 글자 단위처럼 작은 token은 OOV에 강하다. 어떤 문자열이 들어와도 표현할 수 있다. rare token row가 적고, 각 기본 token이 자주 학습된다.

또한 형태를 잘게 쪼개므로 새로운 단어를 조합해서 표현할 수 있다.

```text
낯선 단어:
작은 조각들의 조합으로 표현 가능
```

### 짧은 token 단위의 비용

sequence가 길어진다. 같은 context length에서 실제 문맥 범위가 줄어든다. 한글 byte tokenizer는 특히 손해가 크다.

```text
영화가 정말 재미있다

byte-level:
한글마다 대략 3 token

한글 글자-level:
각 글자 1 token

BPE:
영화, 정말, 재미 같은 조각이 1 token일 수 있음
```

길어진 sequence는 두 가지 비용을 만든다.

```text
1. 계산 비용 증가
2. 의미 문맥 범위 감소
```

GPT는 token 위치 단위로 attention을 하므로, byte tokenizer는 "언어적 문맥"보다 "문자 encoding 복원"에 많은 위치를 사용한다.

### 긴 token 단위의 이득

자주 나오는 조각을 긴 token으로 만들면 sequence가 짧아진다. context가 더 넓은 실제 텍스트를 담고, attention이 의미 단위에 더 가까운 조각을 본다.

빈번한 단어 또는 형태소 조각은 embedding 하나로 안정적으로 학습된다.

```text
영화
정말
재미
attention
tokenizer
gradient
```

혼합 corpus에서는 영어/코드/기호 조각도 중요하다.

```text
def
return
torch
attention
<|chunk_start|>
source_path
```

이런 조각들은 `obsidian_llm_10x`에서 자주 나오므로 BPE merge 대상이 될 가치가 있다.

### 긴 token 단위의 비용

긴 token이 모두 좋은 것은 아니다. 빈번한 긴 token은 유용하지만, 드문 긴 token은 문제다.

드문 긴 token은 다음 문제를 만든다.

```text
1. embedding row가 충분히 update되지 않음
2. LM head row가 충분히 학습되지 않음
3. train 문장에는 맞지만 validation에는 약함
4. token-level loss scale이 커져 비교가 어려움
5. vocab parameter가 증가함
```

따라서 좋은 tokenizer는 "긴 token을 만드는 tokenizer"가 아니라 "자주 나오는 조각만 길게 만들고, 드문 조각은 작게 남기는 tokenizer"다.

## 과적합과의 관계

토크나이저는 과적합과 직접적으로 연결된다. 다만 방향은 단순하지 않다.

### 260 byte tokenizer와 과적합

260 tokenizer는 vocab이 작고 rare long token이 없으므로 embedding row 차원에서는 과적합 위험이 작다. 모든 byte row가 반복적으로 update되고, 특정 phrase를 token ID 하나로 외우는 경로도 없다.

하지만 이것이 일반화가 좋다는 뜻은 아니다.

byte tokenizer는 sequence가 길어지고 의미 단위가 늦게 형성된다. 작은 모델과 짧은 학습에서는 train loss도 충분히 내려가지 않는 underfit이 될 수 있다.

이 경우 gap이 작아 보여도 좋은 신호가 아닐 수 있다.

```text
train loss도 높음
val loss도 높음
gap은 작음

=> 과적합이 적은 것이 아니라 과소학습일 수 있음
```

따라서 260 tokenizer를 평가할 때는 gap만 보면 안 된다. 문자당 validation loss와 생성 품질을 함께 봐야 한다.

### 제한 없는 큰 vocab과 과적합

제한 없이 merge하면 rare long token이 생길 수 있다. 이 token들은 train corpus의 특정 문맥에만 묶인 고유 feature처럼 작동한다.

모델은 다음 방식으로 train data에 과하게 맞을 수 있다.

```text
드문 긴 token ID를 embedding으로 외움
그 token이 나오는 주변 문맥을 외움
train loss는 낮아짐
validation에서는 같은 token 또는 같은 문맥이 적어 성능이 낮음
```

이때 나타나는 지표는 다음과 같다.

```text
final_train_loss 감소
final_val_loss 개선 둔화 또는 악화
final_generalization_gap 증가
train_val_improvement_gap 증가
overfit_score 증가
```

상관 보고서에서 `vocab_size -> overfit_score`가 중간 정도 양의 상관을 보인 것도 이 위험과 맞닿아 있다. 다만 이것은 관찰 상관이며, tokenizer만 통제한 인과 증명은 아니다.

### 빈도 제한 BPE와 과적합

`min_frequency`는 tokenizer 차원의 regularization으로 볼 수 있다.

```text
min_frequency가 낮음:
드문 조각도 token이 됨

min_frequency가 높음:
자주 나오는 조각만 token이 됨
```

한글에 대해 `hangul_pair_min_frequency=20`을 두는 것은 rare Hangul bigram/trigram을 줄이고, 자주 나오는 조각만 token화하겠다는 뜻이다.

이 규칙은 과적합을 줄이는 방향으로 작동할 가능성이 있다. 이유는 다음과 같다.

```text
1. rare token row를 줄인다.
2. train-only phrase token을 줄인다.
3. validation에서도 반복될 만한 조각만 merge한다.
4. 한글 한 글자 단위는 보호하므로 byte-level underfit 비용은 줄인다.
```

즉 한글 글자 보호 + 빈도 제한 merge는 다음 두 극단의 중간이다.

```text
극단 A: 260 byte tokenizer
과적합은 덜할 수 있으나 sequence가 너무 길고 의미 단위 학습이 느림

극단 B: 제한 없는 큰 vocab tokenizer
sequence는 짧지만 rare token이 많아져 과적합과 undertrained row 위험

절충안: 한글 글자 보호 + 빈도 제한 BPE
encoding 낭비는 줄이고 rare long token은 제한
```

## 기존 실험으로부터의 추론 과정

기존 실험에서 이 설계를 추론하는 과정은 다음 순서다.

### 1단계: token-level loss는 tokenizer scale에 강하게 묶여 있다

HY 실험에서 vocab이 커질수록 token-level loss는 커졌다. 그러나 nats/char는 오히려 개선되는 흐름이 있었다.

따라서 "token loss가 낮다"는 이유만으로 작은 vocab이 좋다고 말할 수 없다.

### 2단계: token 길이가 길어지면 token 하나의 난이도는 커진다

상관 보고서에서 `val_chars_per_token -> final_val_loss`의 피어슨 r은 0.9419였다. 이는 토큰 하나가 더 많은 문자를 담당할수록 token-level loss가 커지는 강한 관찰 상관이다.

따라서 큰 vocab의 token loss 상승은 자연스러운 scale 변화다.

### 3단계: 문자당 loss는 큰 vocab의 이득을 보여줄 수 있다

E04~E06에서 vocab 2000보다 4000/5000의 token loss는 높았지만, nats/char와 bits/char는 개선되었다. 이는 BPE merge가 실제 텍스트 압축과 문맥 효율에 도움이 될 수 있음을 보여준다.

### 4단계: 그러나 vocab 증가는 과적합 위험도 동반할 수 있다

상관 보고서에서 `vocab_size -> overfit_score`는 0.5832의 중간 양의 상관을 보였다. 이는 큰 vocab이 작은 corpus에서 rare token 또는 parameter 증가를 통해 과적합 신호와 연결될 수 있음을 시사한다.

### 5단계: 한글 byte-level은 언어 단위 이전의 encoding 문제를 강제로 학습시킨다

260 byte tokenizer는 row update는 균등하지만 한글 한 글자를 3 token으로 표현한다. 이는 모델이 의미를 배우기 전에 UTF-8 byte 구성을 학습하게 만든다.

따라서 한글은 최소한 한 글자 단위까지 보호하는 것이 합리적이다.

### 6단계: 한 글자 이상은 의미 압축이므로 빈도 제한이 필요하다

`영화`, `정말`, `합니다` 같은 조각은 유용하지만, 한두 번 나온 조합을 token화하는 것은 rare token 문제를 만든다.

따라서 한글 한 글자 이상 merge에는 `n`회 이상 등장 조건을 둔다.

### 7단계: 결론은 하이브리드 설계다

최종 추론은 다음이다.

```text
한글 내부 byte는 무조건 한 글자까지 합친다.
그 이상의 한글 조각은 최소 등장 횟수 n 이상일 때만 합친다.
영어/코드/기호는 기존 BPE 빈도 기반 압축을 유지한다.
평가는 token loss가 아니라 nats/char, bits/char, gap, overfit_score로 한다.
```

## 검증해야 할 실험 설계

이 주장을 실제로 검증하려면 dataset을 바꾸지 않고 tokenizer만 바꿔야 한다.

현재 목표 corpus가 `obsidian_llm_10x`라면 다음처럼 고정한다.

```text
train:
data/obsidian_llm_10x_lm_train.txt

main validation:
data/obsidian_llm_10x_lm_val.txt

korean validation:
data/obsidian_llm_10x_korean_val.txt

external korean stress test:
data/nsmc_lm_val.txt
```

비교할 tokenizer는 다음과 같다.

| 이름 | 설명 | 목적 |
| --- | --- | --- |
| `byte_260` | merge 없는 260 byte tokenizer | 균등 update와 긴 sequence의 기준점 |
| `standard_bpe` | 기존 byte-level BPE | 현재 기준선 |
| `hangul_char_bpe_n5` | 한글 1글자 보호, 한글 pair n=5 | 작은 샘플에서 한글 merge 허용 |
| `hangul_char_bpe_n20` | 한글 1글자 보호, 한글 pair n=20 | 추천 기본값 |
| `hangul_char_bpe_n50` | 한글 1글자 보호, 한글 pair n=50 | 보수적 merge |
| `hangul_char_bpe_max2` | 한글 token 최대 2글자 | 사용자의 2글자 가설 검증 |
| `hangul_char_bpe_max4` | 한글 token 최대 4글자 | 추천 기본값 |

고정해야 할 조건은 다음이다.

```text
dataset
train/val split
model size
optimizer
learning_rate
context_length
stride
batch_size
max_steps 또는 epochs
seed set
```

최소 3개 seed를 사용한다.

```text
seed = 134, 151, 202
```

주 지표는 다음이다.

```text
final_val_nats_per_char
final_val_bits_per_char
```

보호 지표는 다음이다.

```text
final_generalization_gap
train_val_improvement_gap
overfit_score
train_tokens_per_char
val_tokens_per_char
rare_token_ratio
token_frequency_histogram
generation sample quality
```

## 성공과 실패의 해석 규칙

### 성공 신호

한글 인식 BPE가 성공하려면 다음 조건을 만족해야 한다.

```text
main validation nats/char가 standard_bpe보다 같거나 낮음
korean validation nats/char가 개선됨
overfit_score가 증가하지 않음
rare token 비율이 과도하게 늘지 않음
생성 샘플에서 한글 깨짐 또는 이상한 byte 패턴이 줄어듦
```

### 부분 성공

다음 경우는 부분 성공이다.

```text
main validation은 비슷함
korean validation만 개선됨
overfit_score는 유지됨
```

이 경우 한글 처리에는 이득이 있지만, 전체 `obsidian_llm_10x` corpus에서는 한글 비중이 낮아서 main metric 개선이 작을 수 있다.

### 실패 신호

다음 경우는 실패다.

```text
main validation nats/char가 악화됨
overfit_score가 증가함
rare Hangul token이 많아짐
train loss만 내려가고 validation은 정체됨
NSMC에서는 좋아지지만 Obsidian Korean subset에서는 좋아지지 않음
```

특히 마지막 경우가 중요하다. NSMC는 영화 리뷰 corpus이므로 네 Obsidian OS/AI 문장과 도메인이 다르다. NSMC 성능은 외부 한글 stress test일 뿐, 목표 corpus 성능을 대체할 수 없다.

## 최종 제안

현재 상황에서 가장 합리적인 설계는 다음이다.

```text
학습 dataset:
obsidian_llm_10x 고정

평가 dataset:
obsidian_llm_10x val
obsidian_llm_10x Korean subset val
NSMC val

tokenizer:
hybrid Hangul-aware BPE

한글 규칙:
한글 byte sequence는 한 글자까지 무조건 보호
한글 2글자 이상 merge는 n회 이상 등장할 때만 허용

기본 n:
20

기본 max_hangul_chars_per_token:
4

비교 실험:
standard_bpe vs hangul_char_bpe_n20
```

이 설계는 다음 세 목표를 동시에 만족하려는 절충안이다.

```text
1. 260 byte tokenizer처럼 token row가 골고루 학습되는 안정성 일부 유지
2. byte-level 한글 표현의 3배 sequence 낭비 제거
3. 제한 없는 BPE의 rare long token 과적합 위험 완화
```

따라서 이 보고서의 중심 결론은 다음 한 문장으로 요약할 수 있다.

```text
한글은 최소 한 글자까지는 deterministic하게 보호하고, 그 이상의 단어 조각은 빈도 기반으로만 merge하는 하이브리드 BPE가 260 byte tokenizer와 무제한 BPE 사이의 가장 합리적인 절충점이다.
```
