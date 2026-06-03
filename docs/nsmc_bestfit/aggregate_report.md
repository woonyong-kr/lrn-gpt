# LLM 10x 집계 보고서

## 현재 상태

- 계획된 조건 수: `5`
- 계획된 분석 행 수: `5`
- 계획된 실제 실행 수: `5`
- 완료된 실행 수: `1`
- 완료된 실제 실행 수: `1`
- 화면 검토 가능 조건(`탐색 조건 n >= 3`): `1`
- 주장 근거 가능 조건(`n >= 10`): `0`
- 전체 실행 결과 원장: `/Users/woonyong/workspace/Krafton-Jungle/SW_AI-W13-gpt/docs/nsmc_bestfit/all_run_results.jsonl`
- 원장 행 수: `1`
- 원장 누락 감사: `PASS`
- result.json 읽기 실패: `0`
- 중복 완료 run_number: `0`
- matrix 밖 완료 결과: `0`
- matrix/result 불일치: `0`

탐색 조건은 3회 반복이 끝나면 화면 검토는 가능하지만, 최종 주장은 선별된 후보에 대해 10회 반복이 필요합니다.

원장 누락 감사가 `PASS`가 아니면 완료된 개별 결과가 집계 원장에 빠졌거나, 읽을 수 없는 결과 파일/중복/계획표 불일치가 있다는 뜻입니다.

## 조건별 요약

| phase | 조건 | 단계 | 축 | 값 | n | 화면 검토 가능 | 주장 가능 | 검증 bits/char 중앙값 | 검증 IQR | 과적합 중앙값 | gap 중앙값 | 시간 h 중앙값 | warm tok/s 중앙값 | paired delta 중앙값 | 기준선 승률 |
| --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phase1_tokenizer | TOK_V8000_MF2 | exploratory | tokenizer | vocab=8000,min_frequency=2 | 1 | yes | no | 6.18769 | 0 | 0.0210552 | 0.00851631 | 0.00097263 | 3817.54 |  |  |
| phase1_tokenizer | TOK_V12000_MF2 | exploratory | tokenizer | vocab=12000,min_frequency=2 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase1_tokenizer | TOK_V16000_MF3 | exploratory | tokenizer | vocab=16000,min_frequency=3 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase1_tokenizer | TOK_V24000_MF3 | exploratory | tokenizer | vocab=24000,min_frequency=3 | 0 | no | no |  |  |  |  |  |  |  |  |
| phase1_tokenizer | TOK_V32000_MF5 | exploratory | tokenizer | vocab=32000,min_frequency=5 | 0 | no | no |  |  |  |  |  |  |  |  |
