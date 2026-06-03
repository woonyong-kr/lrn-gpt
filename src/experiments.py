# -*- coding: utf-8 -*-
"""재현 가능한 mini GPT 하이퍼파라미터 실험 계획과 실행기."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

try:
    from .bpe import BPETokenizer
    from .config import GPTConfig, set_seed
    from .dataset import create_dataloader
    from .model import GPTModel
    from .train import calc_loss_batch, calc_loss_loader
except ImportError:
    from bpe import BPETokenizer
    from config import GPTConfig, set_seed
    from dataset import create_dataloader
    from model import GPTModel
    from train import calc_loss_batch, calc_loss_loader


HYPOTHESES = {
    "baseline": "기준 모델입니다. 다른 실험은 이 설정 대비 train/val loss와 속도 차이를 봅니다.",
    "vocab_size": "어휘가 커지면 byte sequence가 짧아질 수 있지만 LM head가 커져 작은 데이터에서는 과적합과 속도 저하가 생길 수 있습니다.",
    "context_length": "문맥 길이가 길수록 장거리 의존성을 볼 수 있지만 attention 비용이 늘고 작은 데이터에서는 학습 sample 수가 줄어들 수 있습니다.",
    "stride": "stride를 줄이면 겹치는 학습 sample이 늘어 loss는 안정될 수 있지만 데이터 중복으로 과적합 신호가 커질 수 있습니다.",
    "batch_size": "batch가 커지면 gradient noise가 줄어 안정적이지만 같은 step 수에서는 업데이트 다양성이 줄 수 있습니다.",
    "learning_rate": "학습률이 크면 초반 loss는 빨리 내려가지만 발산/불안정 위험이 커집니다.",
    "weight_decay": "weight decay는 과적합을 줄일 수 있지만 작은 모델/짧은 학습에서는 underfit을 만들 수 있습니다.",
    "emb_dim": "embedding 폭이 커지면 표현력과 파라미터 수가 함께 늘어 train loss는 내려가고 val gap은 커질 수 있습니다.",
    "n_heads": "head 수는 attention 관점을 나누지만 head_dim이 작아지면 한 head의 표현력이 줄 수 있습니다.",
    "n_layers": "layer 수가 늘면 조합적 표현력이 늘지만 작은 데이터와 짧은 학습에서는 최적화가 어려워질 수 있습니다.",
    "drop_rate": "dropout은 regularization을 주지만 데이터/step이 작으면 학습 속도를 늦출 수 있습니다.",
    "qkv_bias": "QKV bias는 attention projection의 자유도를 늘리지만 파라미터 증가 대비 효과가 작을 수 있습니다.",
    "ffn_mult": "FFN 확장 배율이 커지면 token별 비선형 변환 능력이 늘지만 계산량과 과적합 위험도 늘어납니다.",
    "norm_first": "Pre-LN은 깊은 모델에서 gradient 흐름이 안정적일 수 있고, Post-LN은 얕은 모델에서 기준 구현과 비교하기 좋습니다.",
    "norm_eps": "LayerNorm eps는 수치 안정성 축입니다. 너무 크면 정규화 효과가 둔해지고 너무 작으면 극단 입력에 민감해질 수 있습니다.",
    "activation_name": "FFN 활성함수는 비선형성과 gradient 형태를 바꿉니다. GELU 계열은 LLM 기준점이고 SwiGLU/GEGLU는 shape를 유지하는 gated FFN 대안입니다.",
    "ffn_dropout_position": "dropout 위치가 activation 뒤인지 output 뒤인지에 따라 noise가 hidden 확장부 또는 residual 직전부에 들어갑니다.",
    "attention_impl": "manual과 SDPA는 같은 attention 의미를 다른 함수로 계산합니다. 결과보다 속도와 수치 차이를 확인하는 축입니다.",
    "tie_embeddings": "입출력 embedding 공유는 파라미터를 줄이고 작은 데이터에서 regularization처럼 작동할 수 있습니다.",
    "init_std": "초기화 표준편차는 logit/activation scale을 정합니다. 너무 크면 불안정하고 너무 작으면 학습 신호가 약할 수 있습니다.",
    "grad_clip": "gradient clipping은 불안정한 설정의 폭주를 막지만 너무 낮으면 학습 속도를 제한합니다.",
    "overfit_gap": "train loss보다 validation loss가 얼마나 높은지 봅니다. gap이 커지면 외운 정도가 늘었다고 해석합니다.",
    "interaction": "여러 축이 동시에 바뀐 상호작용 실험입니다. 단일 축 가설이 조합에서도 유지되는지 봅니다.",
    "seed_repeat": "같은 하이퍼파라미터를 다른 seed로 반복해 결과가 seed 운인지 확인합니다.",
}


@dataclass(slots=True)
class LMExperimentConfig:
    """작은 LM pretraining smoke test 한 회의 설정."""

    run_id: int
    hypothesis: str
    seed: int = 123
    vocab_size: int = 600
    min_frequency: int = 2
    context_length: int = 64
    stride: int | None = None
    batch_size: int = 8
    max_steps: int = 20
    eval_batches: int = 4
    train_ratio: float = 0.9
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float | None = 1.0
    emb_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    drop_rate: float = 0.1
    qkv_bias: bool = False
    ffn_mult: int = 4
    norm_first: bool = False
    norm_eps: float = 1e-5
    activation_name: str = "gelu"
    ffn_dropout_position: str = "after_output"
    attention_impl: str = "manual"
    tie_embeddings: bool = False
    init_std: float = 0.02

    def to_model_config(self) -> dict[str, Any]:
        return GPTConfig(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            emb_dim=self.emb_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            drop_rate=self.drop_rate,
            qkv_bias=self.qkv_bias,
            ffn_mult=self.ffn_mult,
            norm_first=self.norm_first,
            norm_eps=self.norm_eps,
            activation_name=self.activation_name,
            ffn_dropout_position=self.ffn_dropout_position,
            attention_impl=self.attention_impl,
            tie_embeddings=self.tie_embeddings,
            init_std=self.init_std,
            seed=self.seed,
        ).to_dict()


BASELINE = LMExperimentConfig(run_id=1, hypothesis=HYPOTHESES["baseline"])


ONE_FACTOR_VALUES: dict[str, list[Any]] = {
    "vocab_size": [400, 800],
    "context_length": [32, 128],
    "stride": [32],
    "batch_size": [4, 16],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "weight_decay": [0.0, 0.1],
    "grad_clip": [None, 0.5],
    "emb_dim": [64, 192, 256],
    "n_heads": [2, 8],
    "n_layers": [1, 4, 6],
    "drop_rate": [0.0, 0.2],
    "qkv_bias": [True],
    "ffn_mult": [2, 6],
    "norm_first": [True],
    "norm_eps": [1e-6, 1e-4],
    "activation_name": ["gelu_exact", "quick_gelu", "relu", "silu", "mish", "squared_relu", "identity", "swiglu", "geglu"],
    "ffn_dropout_position": ["after_activation", "none"],
    "attention_impl": ["sdpa"],
    "tie_embeddings": [True],
    "init_std": [0.01, 0.04],
}


RANDOM_SPACE: dict[str, list[Any]] = {
    "vocab_size": [400, 600, 800, 1000],
    "context_length": [32, 64, 128],
    "batch_size": [4, 8, 16],
    "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3],
    "weight_decay": [0.0, 0.01, 0.1],
    "grad_clip": [None, 0.5, 1.0],
    "emb_dim": [64, 128, 192, 256],
    "n_heads": [2, 4, 8],
    "n_layers": [1, 2, 4, 6],
    "drop_rate": [0.0, 0.1, 0.2],
    "qkv_bias": [False, True],
    "ffn_mult": [2, 4, 6],
    "norm_first": [False, True],
    "norm_eps": [1e-6, 1e-5, 1e-4],
    "activation_name": ["gelu", "gelu_exact", "quick_gelu", "relu", "silu", "mish", "squared_relu", "identity", "swiglu", "geglu"],
    "ffn_dropout_position": ["after_output", "after_activation", "none"],
    "attention_impl": ["manual", "sdpa"],
    "tie_embeddings": [False, True],
    "init_std": [0.01, 0.02, 0.04],
}


def validate_experiment_config(config: LMExperimentConfig) -> None:
    """실행 전에 shape와 최소 데이터 조건에 영향을 주는 설정을 확인합니다."""
    if config.emb_dim % config.n_heads != 0:
        raise ValueError(f"emb_dim={config.emb_dim} must be divisible by n_heads={config.n_heads}")
    if config.context_length <= 0:
        raise ValueError("context_length must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not 0.0 < config.train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")


def make_experiment_plan(total_runs: int = 150, seed: int = 123) -> list[LMExperimentConfig]:
    """기준 실험, 단일 축 실험, seed 반복, random coverage를 섞어 계획을 만듭니다."""
    if total_runs <= 0:
        return []

    configs: list[LMExperimentConfig] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()

    def add(config: LMExperimentConfig) -> None:
        if len(configs) >= total_runs:
            return
        validate_experiment_config(config)
        signature = _config_signature(config)
        if signature in seen:
            return
        seen.add(signature)
        configs.append(replace(config, run_id=len(configs) + 1))

    add(BASELINE)

    for field, values in ONE_FACTOR_VALUES.items():
        for value in values:
            hypothesis = f"{field}: {HYPOTHESES[field]}"
            add(replace(BASELINE, **{field: value}, hypothesis=hypothesis))

    for seed_value in [321, 777, 2026, 3407]:
        add(replace(BASELINE, seed=seed_value, hypothesis=HYPOTHESES["seed_repeat"]))

    rng = random.Random(seed)
    while len(configs) < total_runs:
        add(_sample_random_config(rng))

    return configs


def write_plan(plan: list[LMExperimentConfig], path: str | Path) -> None:
    """실험 계획을 CSV로 저장합니다."""
    rows = [asdict(config) for config in plan]
    _write_csv(rows, path)


def run_language_model_experiment(config: LMExperimentConfig, corpus: str, device: torch.device, deterministic: bool = False) -> dict[str, Any]:
    """한 개 설정으로 작은 LM 학습 smoke test를 실행하고 metric을 반환합니다."""
    validate_experiment_config(config)
    set_seed(config.seed, deterministic=deterministic)

    train_text, val_text = _split_corpus_text(corpus, config.train_ratio)
    tokenizer = BPETokenizer(vocab_size=config.vocab_size, min_frequency=config.min_frequency).train(train_text)
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    _validate_split_token_ids(train_ids, val_ids, config)

    train_loader = create_dataloader(train_ids, context_length=config.context_length, batch_size=config.batch_size, stride=config.stride, shuffle=True, drop_last=False, seed=config.seed)
    val_loader = create_dataloader(val_ids, context_length=config.context_length, batch_size=config.batch_size, stride=config.context_length, shuffle=False, drop_last=False)
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Not enough tokens for train/validation loaders")

    model = GPTModel(config.to_model_config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    initial_train_loss = calc_loss_loader(train_loader, model, device, num_batches=config.eval_batches)
    initial_val_loss = calc_loss_loader(val_loader, model, device, num_batches=config.eval_batches)

    start_time = time.perf_counter()
    tokens_seen = 0
    last_loss = 0.0
    train_iter = iter(train_loader)
    model.train()

    for _ in range(config.max_steps):
        try:
            input_batch, target_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_batch, target_batch = next(train_iter)

        optimizer.zero_grad(set_to_none=True)
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        loss.backward()
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config.grad_clip))
        optimizer.step()
        last_loss = float(loss.item())
        tokens_seen += int(input_batch.numel())

    elapsed_sec = time.perf_counter() - start_time
    final_train_loss = calc_loss_loader(train_loader, model, device, num_batches=config.eval_batches)
    final_val_loss = calc_loss_loader(val_loader, model, device, num_batches=config.eval_batches)
    overfit_metrics = compute_overfit_metrics(
        initial_train_loss=initial_train_loss,
        initial_val_loss=initial_val_loss,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
    )

    return {
        **asdict(config),
        "actual_vocab_size": len(tokenizer.id_to_token),
        "train_token_count": len(train_ids),
        "val_token_count": len(val_ids),
        "parameter_count": count_parameters(model),
        "initial_train_loss": initial_train_loss,
        "initial_val_loss": initial_val_loss,
        "last_step_loss": last_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "train_loss_delta": initial_train_loss - final_train_loss,
        "val_loss_delta": initial_val_loss - final_val_loss,
        **overfit_metrics,
        "elapsed_sec": elapsed_sec,
        "tokens_seen": tokens_seen,
        "tokens_per_sec": 0.0 if elapsed_sec == 0 else tokens_seen / elapsed_sec,
        "device": str(device),
    }


def run_experiment_plan(plan: list[LMExperimentConfig], corpus: str, output_dir: str | Path, device: torch.device, deterministic: bool = False) -> list[dict[str, Any]]:
    """계획 전체를 실행하고 결과 CSV/JSONL을 저장합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_path = output_path / "results.csv"
    jsonl_path = output_path / "results.jsonl"

    results: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for config in plan:
            result = run_language_model_experiment(config, corpus=corpus, device=device, deterministic=deterministic)
            results.append(result)
            jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            jsonl_file.flush()
            _write_csv(results, results_path)
            print(
                f"run {config.run_id:03d}: "
                f"val {result['initial_val_loss']:.4f} -> {result['final_val_loss']:.4f}, "
                f"gap {result['final_generalization_gap']:.4f}, "
                f"{result['fit_status']}, {result['tokens_per_sec']:.1f} tok/s"
            )

    return results


def count_parameters(model: torch.nn.Module) -> int:
    """학습 가능한 파라미터 수를 반환합니다. 공유 weight는 중복 계산하지 않습니다."""
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        total += parameter.numel()
    return total


def compute_overfit_metrics(initial_train_loss: float, initial_val_loss: float, final_train_loss: float, final_val_loss: float) -> dict[str, Any]:
    """Train/validation loss 차이로 과적합 위험을 해석하기 쉬운 지표로 바꿉니다."""
    train_loss_delta = initial_train_loss - final_train_loss
    val_loss_delta = initial_val_loss - final_val_loss
    initial_generalization_gap = initial_val_loss - initial_train_loss
    final_generalization_gap = final_val_loss - final_train_loss
    generalization_gap_delta = final_generalization_gap - initial_generalization_gap
    improvement_gap = train_loss_delta - val_loss_delta
    overfit_score = max(0.0, final_generalization_gap) + max(0.0, generalization_gap_delta) + max(0.0, improvement_gap)

    if val_loss_delta < -0.01:
        fit_status = "val_regressed"
    elif train_loss_delta < 0.005 and val_loss_delta < 0.005:
        fit_status = "underfit_or_too_short"
    elif generalization_gap_delta > 0.05 and improvement_gap > 0.02:
        fit_status = "overfit_risk"
    elif train_loss_delta > 0.01 and val_loss_delta > 0.01:
        fit_status = "generalizing"
    else:
        fit_status = "mixed"

    return {
        "initial_generalization_gap": initial_generalization_gap,
        "final_generalization_gap": final_generalization_gap,
        "generalization_gap_delta": generalization_gap_delta,
        "train_val_improvement_gap": improvement_gap,
        "overfit_score": overfit_score,
        "fit_status": fit_status,
    }


def load_corpus(path: str | Path, char_limit: int | None = None) -> str:
    """실험용 corpus를 읽고 선택적으로 앞부분만 사용합니다."""
    text = Path(path).read_text(encoding="utf-8")
    if char_limit is not None:
        text = text[:char_limit]
    if not text.strip():
        raise ValueError("Corpus is empty")
    return text


def resolve_device(device_name: str = "auto") -> torch.device:
    """auto/cpu/cuda/mps 이름을 torch.device로 바꿉니다."""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def _sample_random_config(rng: random.Random) -> LMExperimentConfig:
    values = {field: rng.choice(options) for field, options in RANDOM_SPACE.items()}
    context_length = int(values["context_length"])
    values["stride"] = None if rng.random() < 0.55 else max(1, context_length // 2)

    for _ in range(100):
        emb_dim = int(values["emb_dim"])
        n_heads = int(values["n_heads"])
        if emb_dim % n_heads == 0:
            break
        values["n_heads"] = rng.choice(RANDOM_SPACE["n_heads"])
    else:
        values["n_heads"] = 4

    return replace(BASELINE, **values, hypothesis=HYPOTHESES["interaction"])


def _config_signature(config: LMExperimentConfig) -> tuple[tuple[str, Any], ...]:
    ignored = {"run_id", "hypothesis"}
    return tuple(sorted((key, value) for key, value in asdict(config).items() if key not in ignored))


def _split_corpus_text(corpus: str, train_ratio: float) -> tuple[str, str]:
    """Validation text가 tokenizer 학습에 섞이지 않도록 raw text 단계에서 나눕니다."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    split_idx = int(len(corpus) * train_ratio)
    split_idx = max(1, min(split_idx, len(corpus) - 1))
    return corpus[:split_idx], corpus[split_idx:]


def _validate_split_token_ids(train_ids: list[int], val_ids: list[int], config: LMExperimentConfig) -> None:
    min_len = config.context_length + 2
    if len(train_ids) < min_len:
        raise ValueError(f"Need at least {min_len} train tokens, got {len(train_ids)}")
    if len(val_ids) < min_len:
        raise ValueError(f"Need at least {min_len} validation tokens, got {len(val_ids)}")


def _split_token_ids(token_ids: list[int], config: LMExperimentConfig) -> tuple[list[int], list[int]]:
    min_len = config.context_length + 2
    if len(token_ids) < min_len * 2:
        raise ValueError(f"Need at least {min_len * 2} tokens, got {len(token_ids)}")

    split_idx = int(len(token_ids) * config.train_ratio)
    split_idx = max(min_len, split_idx)
    split_idx = min(split_idx, len(token_ids) - min_len)
    return token_ids[:split_idx], token_ids[split_idx:]


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run reproducible mini GPT hyperparameter experiments.")
    parser.add_argument("--total-runs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--corpus-path", type=Path, default=Path(__file__).resolve().parent / "learning" / "the-verdict.txt")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments") / "lm_hparam")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--char-limit", type=int, default=20_000)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    plan = make_experiment_plan(total_runs=args.total_runs, seed=args.seed)
    if args.max_steps is not None:
        plan = [replace(config, max_steps=args.max_steps) for config in plan]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_plan(plan, args.output_dir / "plan.csv")
    (args.output_dir / "plan.json").write_text(json.dumps([asdict(config) for config in plan], ensure_ascii=False, indent=2), encoding="utf-8")

    if args.plan_only:
        print(f"wrote {len(plan)} planned runs to {args.output_dir}")
        return

    corpus = load_corpus(args.corpus_path, char_limit=args.char_limit)
    device = resolve_device(args.device)
    run_experiment_plan(plan, corpus=corpus, output_dir=args.output_dir, device=device, deterministic=args.deterministic)


if __name__ == "__main__":
    main()
