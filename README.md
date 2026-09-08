# lrn-gpt

작은 언어 모델의 학습·재개·생성 프로그램. 핵심 엔진을 실제 입력으로 실행하고 결과와 내부 동작을 확인하는 독립 프로그램이다.

## 실행

Python 3.12와 uv가 필요하다. `make setup`은 이 저장소의 `.venv`만 준비하며 의존성을 `requirements.lock`으로 고정한다.

```sh
make setup
make test
make demo
# 50 step에서 저장하고 100 step까지 재개
.venv/bin/python -m src.runnable train --steps 50 --checkpoint .artifacts/run.pt
.venv/bin/python -m src.runnable train --resume .artifacts/run.pt --steps 100 --checkpoint .artifacts/run.pt
.venv/bin/python -m src.runnable generate --model .artifacts/run.pt --prompt 'A small model' --temperature 0.8 --top-k 20 --length 60
# NSMC를 직접 준비·학습하는 별도 경로
make data
make train-nsmc
```

## 입력에서 출력까지

UTF-8 byte BPE → shifted input/target → embedding·causal attention·decoder Transformer → loss/backward → checkpoint → sampling

byte-level BPE와 핵심 Transformer 구조는 기존 직접 구현을 사용한다. PyTorch tensor/autograd를 사용하며 외부 완제품 언어 모델 API를 호출하지 않는다. CPU 기본 설정은 context 32, embedding 48, head 4, layer 2, dropout 0.1이다.

checkpoint 하나에 model·optimizer·global step·설정·tokenizer·Python/NumPy/PyTorch 난수 상태와 train/validation corpus hash를 저장한다. corpus가 바뀌면 재개를 거절한다. 같은 CPU 환경에서 dropout을 포함해 중간 저장 후 재개한 학습과 연속 학습의 가중치가 정확히 일치한다. 이번 실행 경로는 seeded random contiguous token windows를 복원하므로 예전 epoch DataLoader 실험과 sampling schedule이 다르다.

`--steps`는 추가 step 수가 아니라 최종 global step이다. 학습 step이 시작된 뒤 Ctrl-C에서는 현재 checkpoint를 저장한다. 초기 BPE 준비 중 중단하면 checkpoint가 없으므로 다음 실행에서 준비를 다시 시작한다. 강제 종료는 마지막 저장 지점까지 복구된다. 생성은 빈 prompt·비정상 temperature/top-k/길이를 거절한다.

`models/untrained.pt`, `models/reference.pt`를 포함하므로 demo는 학습 없이 실행된다. 예제 corpus는 이 프로젝트를 위한 짧은 직접 작성 텍스트다. 기본 모델은 0 → 100 step 학습 → checkpoint 재개 → 200 step까지 진행했다. validation loss는 5.7103 → 3.0901(100 step) → 3.2903(200 step)로 다시 증가했다. 과적합을 관찰하는 예제이며 좋은 문장을 생성하는 모델로 소개하지 않는다.

`configs/nsmc.json`은 기존 NSMC baseline의 vocab 3000, context 128, embedding 192, head 4, layer 4, dropout 0.1, lr 4e-4를 유지한다. 감성 분류·sweep·agent 실험은 Git 이력에 보존하고 실행 경로에서 제거했다.

구현을 읽는 순서: `src/runnable.py`, `src/bpe.py`, `src/model.py`, `src/attention.py`.

## 검증과 관찰

8개 필수 테스트로 byte 왕복, causal 미래 차단, 독립 loss 계산, gradient 흐름, dropout 포함 정확한 resume, 변경 corpus 거절을 검증한다. 문장 출력과 구현 정확성·생성 품질을 구분한다.

실행 환경·명령·exit code·원본 백업과 전체 결과는 이번 전환의 별도 작업 폴더에 기록한다. 새 기계에서는 같은 명령으로 직접 재검증한다. 수치가 기록되어 있다는 사실과 현재 실행 성공을 구분한다.

## 지원 범위와 한계

범용 챗봇 품질·대규모 사전학습·분산 학습·RLHF·RAG·상용 모델 대체는 제외한다. 이 경로는 CPU용이며 CUDA/MPS의 난수·kernel 재현성은 검증하지 않는다. torch 버전·하드웨어가 달라지면 bitwise 재현성을 보장하지 않는다.

`make data`는 [NSMC 원본](https://github.com/e9t/nsmc)의 `ratings_train.txt`를 받아 seed 42로 섞은 뒤 최대 약 150만 문자를 train/validation(8%)으로 나눈다. 분류용 파일은 만들지 않는다. 데이터 다운로드와 모델 학습은 기본 demo의 필수 과정이 아니다.

2026-09-08 macOS arm64 / Apple M4 / Python 3.12 / PyTorch 2.14.0 / CPU 2 threads에서 전체 train 3,335,336 bytes를 사용했다. BPE 학습·인코딩 준비 313.91초, 1,000-step 학습·검증·저장 314.43초였다. validation loss는 8.0456 → 5.3064로 감소했다. validation 70,386 tokens 중 완전한 context window 70,272 tokens를 평가했다. 시간은 이 환경의 측정값이며 설치·다운로드 비용은 제외한다.

BPE의 pair 빈도 집계와 병합은 NumPy 배열로 처리한다. 빈도가 같으면 첫 등장 pair를 채택하고 겹치는 반복 pair는 왼쪽부터 병합한다. 기존 86개 corpus/config에서 vocab·merge·encode 결과가 같음을 확인했다. 외부 tokenizer로 대체하지 않았다.

전체 checkpoint는 `.artifacts/nsmc.pt`에 저장된다. 같은 데이터로 1,001번째 step을 재개하려면 다음 명령을 사용한다.

```sh
.venv/bin/python -m src.runnable train --resume .artifacts/nsmc.pt --train data/nsmc_lm_train.txt --validation data/nsmc_lm_val.txt --steps 1001 --checkpoint .artifacts/nsmc-resumed.pt
.venv/bin/python -m src.runnable generate --model .artifacts/nsmc-resumed.pt --prompt '이 영화는' --length 20
```

생성 품질과 loss 감소는 별도로 평가한다. byte-level 생성이 잘못된 UTF-8 조합을 만들면 `�`가 나올 수 있다. checkpoint는 `weights_only=True`로 읽으며 옛 노트북 checkpoint와 자동 호환하지 않는다.

## 출처와 기여

[woonyong-kr/SW_AI-W13-gpt](https://github.com/woonyong-kr/SW_AI-W13-gpt), [Jungle-12-303/wk13_6_gpt](https://github.com/Jungle-12-303/wk13_6_gpt), [krafton-jungle/gpt-lab](https://github.com/krafton-jungle/gpt-lab)에서 이어 받은 학습용 파생본이다. 기준 원본 revision은 `cf423731112aaa992001f04be7c15f2c83b104a0`이다. 원본 과제·팀 코드와 이후 개인 확장을 구분하며, 개별 기여는 Git author와 diff로 확인한다. 기존 저작권 표시는 소스에 유지한다.

과거 문서·실험·기여 기록은 [정리 전 이력](https://github.com/woonyong-kr/lrn-gpt/tree/7875b64f85f19959d66dcaf06c1dcbeb129d71eb)에서 확인할 수 있다. 실행법과 지원 계약은 이 README에 모았다. 개념·설계·실험 해석 자료는 개인 WIKI inbox에서 검토한 뒤 기존 정본에 흡수한다.
