# LLM 학습 예제 실행 순서

이 폴더는 한 파일에 누적하던 실험 코드를 단계별로 나눈 학습용 예제 모음이다.
처음 보면 token ID, embedding vector, target ID, position embedding이 한 덩어리처럼 섞여 보이기 때문에 파일을 일부러 잘게 나누었다.

읽을 때 붙잡을 기준은 이것이다.

```text
tokenizer는 문자열을 token ID로 바꾼다.
embedding layer는 token ID를 vector로 바꾼다.
target은 vector가 아니라 다음 token ID다.
position embedding은 같은 token이라도 몇 번째 자리인지 알려준다.
```

## 실행 순서

| 순서 | 파일 | 설명 |
| --- | --- | --- |
| 0 | `00_download_and_read.py` | 먼저 원문이 뭔지 본다. 아직 token도 vector도 아니다. |
| 1 | `01_regex_preprocess.py` | toy tokenizer를 위해 정규식으로 token 후보를 만든다. 이건 BPE가 아니라 감 잡기용이다. |
| 2 | `02_vocab_and_simple_tokenizer.py` | `"문자열 조각 -> token ID"` 사전을 만든다. 여기까지도 vector는 아니다. |
| 3 | `03_tiktoken_bpe.py` | GPT-2 BPE tokenizer로 실제 token ID sequence를 만든다. 사용 단계에서는 새 병합을 만들지 않는다. |
| 4 | `04_dataset_and_dataloader.py` | `input=[현재 token들]`, `target=[다음 token들]` 구조를 만든다. target은 다음 token ID다. |
| 5 | `05_token_embedding.py` | token ID가 embedding table의 한 행, 즉 vector로 바뀌는 과정을 본다. |
| 6 | `06_token_and_position_embedding.py` | token vector에 position vector를 더해 Transformer 입력을 만든다. |

전체를 한 번에 실행하려면 다음 명령을 사용한다.

```bash
python src/learning/download_the_verdict.py
```

개별 파일도 직접 실행할 수 있다.

```bash
python src/learning/06_token_and_position_embedding.py
```

## 큰 흐름

```text
문자열
  -> tokenizer
  -> token IDs
  -> token embedding
  -> position embedding 더하기
  -> Transformer 입력
```

Tokenizer의 vocab은 `"lower" -> 1500`처럼 문자열 조각에 정수 ID를 붙인 것이다.
Embedding은 `1500 -> vector`처럼 GPT 모델 내부에서 token ID를 계산 가능한 벡터로 바꾸는 layer다.

## MNIST처럼 보기

MNIST는 이미지 하나를 보고 0부터 9까지의 클래스 중 하나를 맞힌다.
GPT의 다음 token 예측도 비슷하다. 다만 클래스 개수가 10개가 아니라 `vocab_size`개다.

```text
MNIST:
이미지 픽셀 -> 신경망 -> 0~9 점수 -> 정답 숫자 class와 비교

GPT:
token IDs -> embedding -> Transformer -> vocab 전체 점수 -> 다음 token ID와 비교
```

그래서 이 예제들을 볼 때는 "정답 벡터를 맞히는 것인가?"가 아니라
"출력층의 정답 token ID 칸을 높이는 것인가?"라고 생각하면 된다.
