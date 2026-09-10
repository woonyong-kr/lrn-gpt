# 🧠 lrn-gpt

직접 구현한 byte-level BPE와 작은 decoder Transformer로 텍스트를 학습하고 생성하는 프로그램입니다. PyTorch는 텐서 연산과 자동 미분에 사용합니다.

[Transformer Wiki](https://docs.woonyong.com/wiki/ai-machine-learning-transformer-924ea08dd69a/) · [모델과 측정 결과](models/manifest.json)

## 실행

Python 3.12와 [uv](https://docs.astral.sh/uv/getting-started/installation/)를 설치한 뒤 실행합니다. 의존성은 `requirements.lock`으로 고정합니다.

```sh
make setup
make demo
make test
```

데모는 포함된 학습 전·후 모델에 같은 prompt를 넣고 sampling 설정별 결과를 출력합니다. 데이터 다운로드나 새 학습은 필요 없습니다.

```sh
# 50 step에서 저장하고 최종 100 step까지 재개
.venv/bin/python -m src.runnable train --steps 50 --checkpoint .artifacts/run.pt
.venv/bin/python -m src.runnable train --resume .artifacts/run.pt --steps 100 --checkpoint .artifacts/run.pt
.venv/bin/python -m src.runnable generate --model .artifacts/run.pt --prompt 'A small model' --temperature 0.8 --top-k 20 --length 60
# NSMC 데이터 준비와 학습은 별도 실행
make data
make train-nsmc
```

## 구현과 설계

UTF-8 텍스트 → BPE → 입력·정답 토큰 쌍 → causal attention·Transformer → loss/backward → checkpoint → sampling으로 이어집니다.

- [bpe.py](src/bpe.py): byte-level encode/decode. 빈도 동률은 처음 등장한 pair, 겹치는 pair는 왼쪽부터 병합합니다.
- [model.py](src/model.py)·[attention.py](src/attention.py): embedding, 미래 토큰을 가리는 attention, residual·normalization·feed-forward.
- [runnable.py](src/runnable.py): 학습·생성 CLI. checkpoint에 model·optimizer·step·설정·tokenizer·난수 상태·corpus hash를 함께 저장하며, 데이터가 바뀐 재개는 거절합니다.

`make test`는 토큰화 왕복, causal mask, 독립 loss 계산, gradient, dropout을 포함한 재개를 검사합니다. `--steps`는 추가 횟수가 아닌 최종 step입니다. 학습 중 Ctrl-C는 현재 checkpoint를 저장하고, 강제 종료는 마지막 저장 지점까지 복구합니다. 초기 BPE 준비 중에는 checkpoint가 아직 없습니다.

## 현재 범위

CPU용 학습·생성 예제입니다. 제공 모델의 loss·설정·준비 비용은 [manifest](models/manifest.json)에 있으며, 생성 품질과 구현 정확성은 별도로 봅니다. byte-level 생성에서 잘못된 UTF-8 조합은 `�`로 보일 수 있습니다. checkpoint는 `weights_only=True`로 읽으며 이전 노트북 형식과 자동 호환하지 않습니다.

`make data`는 [NSMC 원본](https://github.com/e9t/nsmc)을 내려받습니다. 다운로드·BPE 준비·학습 비용은 기본 데모와 별개입니다. 같은 CPU 환경의 재개를 검증했으며 버전·하드웨어가 다른 실행의 bitwise 일치는 보장하지 않습니다. 범용 챗봇, 분산 학습, RLHF와 RAG는 범위에 포함하지 않습니다.

## 출처와 기여

`krafton-jungle/gpt-lab` 과제와 `Jungle-12-303/wk13_6_gpt` 팀 구현을 거친 `woonyong-kr/SW_AI-W13-gpt`에서 이어 받은 학습용 파생본이다. 기준 원본 revision은 `cf423731112aaa992001f04be7c15f2c83b104a0`이다. 과제 제공물·팀 구현·이후 확장은 Git author와 diff로 구분하며, 기존 저작권 표시는 유지한다. 원본 주소의 공개 접근이 제한돼 있어 자료는 아래 이력 링크로 확인할 수 있다.

기존 BPE·Transformer 구현을 CPU 학습·재개·생성 경로로 묶었다. 감성 분류와 대량 탐색, 자동 학습 agent를 포함한 이전 실험은 [정리 전 이력](https://github.com/woonyong-kr/lrn-gpt/tree/7875b64f85f19959d66dcaf06c1dcbeb129d71eb)에 남아 있다.
