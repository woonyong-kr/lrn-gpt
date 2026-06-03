# -*- coding: utf-8 -*-
"""LLM 10x metric logging and aggregation helpers."""

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def test_per_char_metrics_use_loss_times_tokens_per_char():
    """bits/char는 token loss에 tokens/char를 곱한 뒤 ln(2)로 나눈 값이다."""
    from llm_10x_run_next import per_char_metrics

    metrics = per_char_metrics(loss=5.0, tokens_per_char=0.5)

    assert metrics["nats_per_char"] == 2.5
    assert metrics["bits_per_char"] == 2.5 / math.log(2.0)


def test_tokenizer_profile_contains_required_fields():
    """tokenizer_profile은 vocab sweep 해석에 필요한 필수 필드를 담는다."""
    from llm_10x_prepare_cache import profile_from_manifest

    manifest = {
        "corpus_sha256": "corpus",
        "train_sha256": "train",
        "val_sha256": "val",
        "requested_vocab_size": 8,
        "actual_vocab_size": 8,
        "tokenizer_min_frequency": 2,
        "bpe_merge_count": 2,
        "train_chars": 100,
        "val_chars": 40,
        "train_tokens": 50,
        "val_tokens": 20,
        "train_tokens_per_char": 0.5,
        "val_tokens_per_char": 0.5,
        "train_chars_per_token": 2.0,
        "val_chars_per_token": 2.0,
        "special_token_ids": {"<pad>": 0, "<unk>": 1},
    }
    tokenizer_payload = {
        "model": {
            "vocab": {"<pad>": 0, "<unk>": 1, "a": 2, "ab": 3, "abcd": 4, "abcdefgh": 5, "abcdefghijkl": 6, "abcdefghijklmnopq": 7},
            "merges": ["a b", "ab c"],
        }
    }

    profile = profile_from_manifest(manifest, tokenizer_payload)

    for field in [
        "corpus_sha256",
        "train_sha256",
        "val_sha256",
        "requested_vocab_size",
        "actual_vocab_size",
        "tokenizer_min_frequency",
        "bpe_merge_count",
        "train_tokens_per_char",
        "val_tokens_per_char",
        "train_chars_per_token",
        "val_chars_per_token",
        "token_length_histogram",
        "top_50_merge_rules",
        "special_token_ids",
    ]:
        assert field in profile
    assert profile["token_length_histogram"]["length_17_plus"] == 1
    assert profile["top_50_merge_rules"] == ["a b", "ab c"]


def test_aggregate_enriches_compute_and_rebound_metrics():
    """기존 result.json도 manifest와 loss만 있으면 LLM 지표를 재계산할 수 있어야 한다."""
    from llm_10x_aggregate import enrich_result_metrics

    result = enrich_result_metrics(
        {
            "final_train_loss": 4.0,
            "final_val_loss": 5.0,
            "best_val_loss": 4.8,
            "parameter_count": 100,
            "tokens_seen": 25,
            "batch_size": 5,
            "context_length": 5,
            "best_step": 1,
            "cache_vocab_manifest": {
                "train_tokens_per_char": 0.5,
                "val_tokens_per_char": 0.25,
                "train_chars_per_token": 2.0,
            },
        }
    )

    assert result["final_val_nats_per_char"] == 1.25
    assert result["final_val_bits_per_char"] == 1.25 / math.log(2.0)
    assert math.isclose(result["final_minus_best_val_loss"], 0.2)
    assert result["best_tokens_seen"] == 25
    assert result["compute_proxy"] == 2500
    assert result["estimated_train_flops"] == 15000
    assert result["tokens_per_param"] == 0.25


def test_generation_repetition_metrics_detect_repeated_ngrams():
    """generation audit는 distinct-n과 3gram 반복률을 계산한다."""
    from llm_generation_metrics import sample_metrics

    metrics = sample_metrics({"prompt": "p", "text": "a b c a b c a b c"})

    assert metrics["distinct_1"] < 1.0
    assert metrics["distinct_2"] < 1.0
    assert metrics["repetition_ratio_3gram"] > 0.0
    assert metrics["generated_length_units"] == 9
