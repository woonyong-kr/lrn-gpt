# -*- coding: utf-8 -*-
"""실험 옵션과 150회 계획 생성 테스트."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_feedforward_activation_variants_keep_shape():
    """활성함수 교체가 FFN 입출력 shape를 바꾸지 않는지 확인한다."""
    from model import FeedForward

    x = torch.randn(2, 5, 16)
    for activation_name in ["gelu", "gelu_exact", "quick_gelu", "relu", "silu", "swish", "mish", "squared_relu", "identity", "swiglu", "geglu"]:
        ffn = FeedForward(16, dropout=0.0, mult=2, activation_name=activation_name)
        assert ffn(x).shape == x.shape


def test_feedforward_dropout_positions_keep_shape():
    """FFN dropout 위치 옵션이 구조의 바깥 shape를 유지하는지 확인한다."""
    from model import FeedForward

    x = torch.randn(2, 5, 16)
    for dropout_position in ["after_output", "after_activation", "none"]:
        ffn = FeedForward(16, dropout=0.1, mult=2, dropout_position=dropout_position)
        assert ffn(x).shape == x.shape


def test_gpt_model_experiment_options_keep_logits_shape():
    """구조를 유지한 실험 옵션들이 GPT logits shape를 유지하는지 확인한다."""
    from model import GPTModel

    config = {
        "vocab_size": 128,
        "context_length": 16,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "ffn_mult": 2,
        "norm_first": True,
        "activation_name": "silu",
        "ffn_dropout_position": "after_activation",
        "attention_impl": "sdpa",
        "tie_embeddings": True,
    }
    model = GPTModel(config)
    idx = torch.randint(0, config["vocab_size"], (2, 8))
    logits = model(idx)
    assert logits.shape == (2, 8, config["vocab_size"])
    assert model.lm_head.weight is model.embedding.token_embedding.weight


def test_gpt_config_contains_experiment_options():
    """GPTConfig dataclass에 실험 옵션이 기본값과 함께 포함되는지 확인한다."""
    from config import GPTConfig

    config = GPTConfig(vocab_size=128, context_length=16)
    config_dict = config.to_dict()

    assert config_dict["activation_name"] == "gelu"
    assert config_dict["ffn_dropout_position"] == "after_output"
    assert config_dict["attention_impl"] == "manual"
    assert config_dict["norm_eps"] == 1e-5
    assert config_dict["tie_embeddings"] is False


def test_create_dataloader_seed_reproducible_shuffle():
    """같은 seed를 넘긴 shuffle DataLoader가 같은 배치 순서를 만드는지 확인한다."""
    from dataset import create_dataloader

    token_ids = list(range(200))
    loader1 = create_dataloader(token_ids, context_length=8, batch_size=4, shuffle=True, seed=123)
    loader2 = create_dataloader(token_ids, context_length=8, batch_size=4, shuffle=True, seed=123)
    batch1 = next(iter(loader1))[0]
    batch2 = next(iter(loader2))[0]

    torch.testing.assert_close(batch1, batch2)


def test_make_experiment_plan_150_valid_configs():
    """150회 실험 계획이 요청한 개수만큼 만들어지고 head divisibility를 만족하는지 확인한다."""
    from experiments import make_experiment_plan

    plan = make_experiment_plan(total_runs=150, seed=123)
    assert len(plan) == 150
    assert {config.run_id for config in plan} == set(range(1, 151))
    assert all(config.emb_dim % config.n_heads == 0 for config in plan)


def test_experiment_result_contains_required_metrics():
    """실험 실행 결과가 loss, 과적합, 속도, 환경 지표를 기록하는지 확인한다."""
    from experiments import LMExperimentConfig, run_language_model_experiment

    corpus = ("hello world. " * 400).strip()
    config = LMExperimentConfig(
        run_id=1,
        hypothesis="smoke",
        vocab_size=260,
        context_length=8,
        batch_size=2,
        max_steps=1,
        eval_batches=1,
        emb_dim=16,
        n_heads=4,
        n_layers=1,
        drop_rate=0.0,
    )
    result = run_language_model_experiment(config, corpus=corpus, device=torch.device("cpu"))

    for key in [
        "initial_train_loss",
        "initial_val_loss",
        "final_train_loss",
        "final_val_loss",
        "final_generalization_gap",
        "overfit_score",
        "fit_status",
        "epochs",
        "steps_per_epoch",
        "max_steps",
        "actual_vocab_size",
        "bpe_merge_count",
        "corpus_char_count",
        "corpus_sha256",
        "train_char_count",
        "val_char_count",
        "train_token_count",
        "val_token_count",
        "train_tokens_per_char",
        "val_tokens_per_char",
        "train_chars_per_token",
        "val_chars_per_token",
        "final_train_nats_per_char",
        "final_val_nats_per_char",
        "final_train_bits_per_char",
        "final_val_bits_per_char",
        "parameter_count",
        "tokens_per_sec",
        "elapsed_sec",
        "device",
    ]:
        assert key in result
    assert result["val_tokens_per_char"] > 0
    assert result["final_val_nats_per_char"] > 0


def test_epoch_option_resolves_to_loader_length():
    """epochs 옵션이 train DataLoader 길이에 맞춰 실제 update 수로 환산되는지 확인한다."""
    from experiments import LMExperimentConfig, estimate_steps_per_epoch, run_language_model_experiment

    corpus = ("hello world. " * 400).strip()
    config = LMExperimentConfig(
        run_id=1,
        hypothesis="epoch smoke",
        vocab_size=260,
        context_length=8,
        batch_size=2,
        epochs=1.0,
        max_steps=99,
        eval_batches=1,
        emb_dim=16,
        n_heads=4,
        n_layers=1,
        drop_rate=0.0,
    )
    result = run_language_model_experiment(config, corpus=corpus, device=torch.device("cpu"))

    expected_steps = estimate_steps_per_epoch(
        int(result["train_token_count"]),
        context_length=config.context_length,
        batch_size=config.batch_size,
        stride=config.stride,
    )
    assert result["epochs"] == 1.0
    assert result["steps_per_epoch"] == expected_steps
    assert result["max_steps"] == expected_steps


def test_compute_overfit_metrics_flags_overfit_risk():
    """Train loss만 크게 내려간 경우 과적합 위험으로 분류하는지 확인한다."""
    from experiments import compute_overfit_metrics

    metrics = compute_overfit_metrics(
        initial_train_loss=5.0,
        initial_val_loss=5.1,
        final_train_loss=3.0,
        final_val_loss=5.05,
    )

    assert metrics["final_generalization_gap"] > metrics["initial_generalization_gap"]
    assert metrics["train_val_improvement_gap"] > 0
    assert metrics["fit_status"] == "overfit_risk"


def test_train_loop_visual_summary_and_svgs_include_metrics():
    """자동화 대시보드용 요약과 SVG가 loss/과적합 지표를 포함하는지 확인한다."""
    from train_loop_agent import build_metrics_summary, render_latest_run_svg, render_trend_svg

    rows = [
        {
            "run_id": 1,
            "final_train_loss": 5.0,
            "final_val_loss": 5.2,
            "final_generalization_gap": 0.2,
            "generalization_gap_delta": 0.15,
            "train_val_improvement_gap": 0.13,
            "overfit_score": 0.48,
            "fit_status": "overfit_risk",
            "parameter_count": 1234,
            "tokens_per_sec": 100.0,
            "device": "cpu",
            "artifact_dir": "docs/train/runs/run_001_artifacts",
        }
    ]

    summary = build_metrics_summary(rows)
    trend_svg = render_trend_svg(summary)
    latest_svg = render_latest_run_svg({**rows[0], "initial_train_loss": 5.5, "initial_val_loss": 5.6})

    assert summary[0]["risk_level"] == "high"
    assert "학습 / 검증 손실" in trend_svg
    assert "일반화 / 과적합" in trend_svg
    assert "손실 요약" in latest_svg
    assert "과적합 신호" in latest_svg


def test_train_loop_effect_map_requires_loss_and_overfit_deltas():
    """해석 지도는 loss 개선과 과적합 비용을 함께 보여줘야 한다."""
    from train_loop_agent import derive_matched_effects, render_effect_map_svg

    base_config = {
        "seed": 42,
        "vocab_size": 600,
        "context_length": 48,
        "stride": 24,
        "batch_size": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "emb_dim": 96,
        "n_heads": 4,
        "n_layers": 3,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "ffn_mult": 4.0,
        "norm_first": True,
        "norm_eps": 1e-5,
        "activation_name": "gelu",
        "ffn_dropout_position": "post_activation",
        "attention_impl": "sdpa",
        "tie_embeddings": True,
        "init_std": 0.02,
    }
    rows = [
        {
            **base_config,
            "run_id": 1,
            "epochs": 2.5,
            "max_steps": 100,
            "steps_per_epoch": 40,
            "final_val_loss": 5.4,
            "final_generalization_gap": 0.01,
            "overfit_score": 0.05,
            "final_val_nats_per_char": 3.2,
        },
        {
            **base_config,
            "run_id": 2,
            "epochs": 2.75,
            "max_steps": 110,
            "steps_per_epoch": 40,
            "final_val_loss": 5.3,
            "final_generalization_gap": 0.03,
            "overfit_score": 0.08,
            "final_val_nats_per_char": 3.14,
        },
    ]

    effects = derive_matched_effects(rows)
    svg = render_effect_map_svg(effects)

    assert effects[0]["axis"] == "epochs"
    assert effects[0]["delta_val_loss"] == -0.1
    assert effects[0]["delta_overfit"] == 0.03
    assert "검증 손실 변화" in svg
    assert "과적합 점수 변화" in svg


def test_train_loop_correlation_report_keeps_tokenizer_and_overfit_outcomes():
    """BPE/tokenizer 상관은 char-normalized loss와 과적합 outcome을 함께 계산한다."""
    from train_loop_agent import build_correlation_records, compute_correlation_results, render_correlation_svg

    rows = [
        {
            "run_id": 1,
            "scope": "auto_loop",
            "vocab_size": 400,
            "actual_vocab_size": 400,
            "val_tokens_per_char": 0.5,
            "val_chars_per_token": 2.0,
            "final_val_loss": 5.8,
            "final_val_nats_per_char": 2.9,
            "final_generalization_gap": 0.01,
            "overfit_score": 0.03,
            "corpus_sha256": "same",
        },
        {
            "run_id": 2,
            "scope": "auto_loop",
            "vocab_size": 600,
            "actual_vocab_size": 600,
            "val_tokens_per_char": 0.4,
            "val_chars_per_token": 2.5,
            "final_val_loss": 5.5,
            "final_val_nats_per_char": 2.2,
            "final_generalization_gap": 0.02,
            "overfit_score": 0.05,
            "corpus_sha256": "same",
        },
        {
            "run_id": 3,
            "scope": "auto_loop",
            "vocab_size": 800,
            "actual_vocab_size": 800,
            "val_tokens_per_char": 0.3,
            "val_chars_per_token": 3.333,
            "final_val_loss": 5.3,
            "final_val_nats_per_char": 1.59,
            "final_generalization_gap": 0.04,
            "overfit_score": 0.08,
            "corpus_sha256": "same",
        },
    ]

    correlation_rows = build_correlation_records(rows, [])
    correlations = compute_correlation_results(correlation_rows)
    svg = render_correlation_svg(correlations)

    assert any(item["feature"] == "vocab_size" and item["outcome"] == "final_val_nats_per_char" for item in correlations)
    assert any(item["outcome"] == "overfit_score" for item in correlations)
    assert "상관 증거" in svg


def test_split_corpus_text_keeps_validation_out_of_tokenizer_training():
    """Raw text를 먼저 나눠 tokenizer 학습과 validation 평가를 분리할 수 있게 한다."""
    from experiments import _split_corpus_text

    train_text, val_text = _split_corpus_text("abcdefghij", train_ratio=0.6)

    assert train_text == "abcdef"
    assert val_text == "ghij"
