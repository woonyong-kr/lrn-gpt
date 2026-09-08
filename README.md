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

대화형 서버는 해당 터미널에서 Ctrl-C로 종료한다. demo/test의 자식 프로세스는 실행기가 보유한 PID 또는 컨테이너 ID로만 종료한다. 다른 서버를 포트 번호로 찾아 일괄 종료하지 않는다. 준비된 Python 환경이 없으면 먼저 `make setup`을 실행한다.

## 입력에서 출력까지

UTF-8 byte BPE → shifted input/target → embedding·causal attention·decoder Transformer → loss/backward → checkpoint → sampling

byte-level BPE와 핵심 Transformer 구조는 기존 직접 구현을 사용한다. PyTorch tensor/autograd를 사용하며 외부 완제품 언어 모델 API를 호출하지 않는다. CPU 기본 설정은 context 32, embedding 48, head 4, layer 2, dropout 0.1이다.

checkpoint 하나에 model·optimizer·global step·설정·tokenizer·Python/NumPy/PyTorch 난수 상태와 train/validation corpus hash를 저장한다. corpus가 바뀌면 재개를 거절한다. 같은 CPU 환경에서 dropout을 포함해 중간 저장 후 재개한 학습과 연속 학습의 가중치가 정확히 일치한다. 이번 실행 경로는 seeded random contiguous token windows를 복원하므로 예전 epoch DataLoader 실험과 sampling schedule이 다르다.

`--steps`는 추가 step 수가 아니라 최종 global step이다. Ctrl-C에서는 현재 checkpoint를 저장한다. 강제 종료는 마지막 저장 지점까지 복구된다. 생성은 빈 prompt·비정상 temperature/top-k/길이를 거절한다.

`models/untrained.pt`, `models/reference.pt`를 포함하므로 demo는 학습 없이 실행된다. 예제 corpus는 이 프로젝트를 위한 짧은 직접 작성 텍스트다. 기본 모델은 0 → 100 step 학습 → checkpoint 재개 → 200 step까지 진행했다. validation loss는 5.7103 → 3.0901(100 step) → 3.2903(200 step)로 다시 증가했다. 과적합을 관찰하는 예제이며 좋은 문장을 생성하는 모델로 소개하지 않는다.

`configs/nsmc.json`은 기존 NSMC baseline의 vocab 3000, context 128, embedding 192, head 4, layer 4, dropout 0.1, lr 4e-4를 유지한다. 원본 자료·감성 분류·sweep·agent 코드는 [아카이브 안내](archive/README.md)에서 구분한다.

구현을 읽는 순서: `src/runnable.py`, `src/bpe.py`, `src/model.py`, `src/attention.py`.

## 검증과 관찰

기존 테스트와 byte 왕복, causal 미래 차단, 독립 loss 계산, gradient 흐름, dropout 포함 정확한 resume, 변경 corpus 거절을 검증한다. 문장 출력과 구현 정확성·생성 품질을 구분한다.

실행 환경·명령·exit code·원본 백업과 전체 결과는 이번 전환의 별도 작업 폴더에 기록한다. 새 기계에서는 같은 명령으로 직접 재검증한다. 수치가 기록되어 있다는 사실과 현재 실행 성공을 구분한다.

## 지원 범위와 한계

범용 챗봇 품질·대규모 사전학습·분산 학습·RLHF·RAG·상용 모델 대체는 제외한다. 이 경로는 CPU용이며 CUDA/MPS의 난수·kernel 재현성은 검증하지 않는다. torch 버전·하드웨어가 달라지면 bitwise 재현성을 보장하지 않는다.

`make data`는 NSMC 공개 원본을 다운로드한다. baseline의 순수 Python BPE 학습은 작은 CPU demo보다 훨씬 비싸고, tokenization 이후 1000 step 학습 비용이 추가된다. 준비된 짧은 corpus 모델과 NSMC 실행 결과는 구분해서 기록한다. 체크포인트는 `weights_only=True`로 읽으며 예전 노트북의 불완전 checkpoint와 자동 호환하지 않는다.

## 원본·학습 문서의 경계

[원본 아카이브와 기여 구분](archive/README.md)을 확인한다. 이 저장소는 실행 코드·테스트·사용법·설계 근거를 소유한다. WIKI는 개념 정본을 소유하며 기존 정본·공통 색인·배포 파일을 이 작업에서 수정하지 않는다. SQL·PintOS와 RepoLM/음성 서비스는 이 프로그램의 실행 의존성이 아니다.
