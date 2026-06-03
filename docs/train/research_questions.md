# 해석 가능한 다음 연구 질문

현재 목표는 새 run을 많이 쌓는 것이 아니라, 변수를 올리면 무엇이 좋아지고 무엇이 나빠지는지 설명 가능한 지도를 만드는 것입니다.

## 현재 기준점

- overfit-aware best: run `102` / val `5.534507` / overfit `0.011694`
- latest: run `112` / val `5.525625` / overfit `0.057627`
- 모든 새 결과는 loss 그래프와 overfit 그래프가 함께 있어야 유효합니다.

## 우선순위 질문

1. BPE/tokenizer 상태와 dataset 상태는 loss scale과 학습 내용에 어떤 상관을 갖는가?
   - 짝비교 단계: dataset 고정 후 vocab/BPE만 변경, 그 다음 tokenizer 고정 후 dataset만 변경
   - 주 지표: `final_val_nats_per_char`, 생성 샘플 품질
   - 보호 지표: `final_generalization_gap`, `overfit_score`, loss/overfit 그래프

2. `epochs` 2.56 -> 2.69는 low-risk seed에서 raw validation을 개선하지만, overfit 비용을 얼마나 반복적으로 키우는가?
   - 짝비교: 같은 seed, 같은 tokenizer, 같은 model, epochs만 변경
   - 주 지표: `final_val_loss`
   - 보호 지표: `final_generalization_gap`, `overfit_score`

3. `stride` 24 -> 20은 high-gap seed rescue인가, 아니면 기본 정책으로도 유리한가?
   - 짝비교: 같은 seed와 실제 업데이트 수, stride만 변경
   - 주 지표: `overfit_score`
   - 보호 지표: `final_val_loss`

4. vocab/BPE merge는 token loss 착시를 얼마나 만드는가?
   - 짝비교 단계: vocab만 변경, model/training 고정
   - 주 지표: `final_val_nats_per_char`, `final_val_bits_per_char`
   - 보호 지표: 생성 샘플, parameter_count, steps_per_epoch

5. capacity를 올리면 무엇이 좋아지는가?
   - 짝비교: `emb_dim`, `n_layers`, `ffn_mult` 중 하나만 변경
   - 주 지표: `final_val_loss`
   - 보호 지표: `parameter_count`, `tokens_per_sec`, `overfit_score`

## 금지할 해석

- vocab_size가 다른 run끼리 token-level `final_val_loss`만 비교하지 않는다.
- loss가 낮아졌는데 overfit 그래프가 없으면 결론으로 쓰지 않는다.
- seed와 hyperparameter가 동시에 바뀐 run은 인과 결론으로 쓰지 않는다.
- 한 번에 여러 변수를 바꾼 run은 rescue 후보로만 보고 effect map의 핵심 근거로 쓰지 않는다.

## 다음 실험 제안 형식

```json
{
  "baseline_run": 112,
  "changed_variable": "seed",
  "expected_intermediate_change": "2.69 epoch 후보의 seed variance를 확인한다.",
  "primary_metric": "low-risk 범위 안의 final_val_loss",
  "guardrail_metric": "final_generalization_gap과 overfit_score",
  "interpretation_rule": "fresh seed에서도 low-risk면 2.69 epochs를 후보로 유지하고, gap이 커지면 2.56 epochs를 기본값으로 둔다."
}
```
