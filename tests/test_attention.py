# -*- coding: utf-8 -*-
"""
MultiHeadAttention 단위 테스트.
실행: `pytest tests/test_attention.py -v`
"""

import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# =============================================================================
# MultiHeadAttention
# =============================================================================


class TestMultiHeadAttention:
    """MultiHeadAttention forward 구현 후 실행."""

    def test_mha_output_shape(self):
        """MultiHeadAttention 출력 shape가 입력과 같은 (batch, seq_len, d_model)인지 확인한다."""
        from attention import MultiHeadAttention

        batch_size, seq_len, d_model = 2, 10, 64
        n_heads = 4
        try:
            mha = MultiHeadAttention(d_model, n_heads, drop_rate=0.0)
            x = torch.randn(batch_size, seq_len, d_model)
            out = mha(x, causal_mask=True)
        except (NotImplementedError, TypeError):
            pytest.fail("MultiHeadAttention 미구현")
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape == (batch_size, seq_len, d_model)

    def test_mha_causal_mask_future_zero(self):
        """Causal mask 적용 시 현재 토큰이 미래 위치에 attention을 주지 않는지 확인한다."""
        from attention import MultiHeadAttention

        batch_size, seq_len, d_model = 1, 4, 8
        n_heads = 2
        try:
            mha = MultiHeadAttention(d_model, n_heads, drop_rate=0.0)
            x = torch.randn(batch_size, seq_len, d_model)
            out, attn_weights = mha(x, causal_mask=True, return_attention_weights=True)
        except (NotImplementedError, TypeError):
            pytest.fail("MultiHeadAttention + return_attention_weights 미구현")
        # attn_weights: (B, n_heads, T, T). 상삼각(diagonal 제외)이 0이어야 함
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert torch.allclose(attn_weights[0, :, i, j], torch.zeros(n_heads))

    def test_mha_mask_created_in_inference_mode_can_train_afterward(self):
        """평가 모드에서 만들어진 causal mask가 이후 backward를 막지 않는지 확인한다."""
        from attention import MultiHeadAttention

        mha = MultiHeadAttention(d_model=8, n_heads=2, drop_rate=0.0)
        x = torch.randn(1, 4, 8)
        with torch.inference_mode():
            mha(x, causal_mask=True)

        train_x = torch.randn(1, 4, 8, requires_grad=True)
        out = mha(train_x, causal_mask=True)
        out.sum().backward()
        assert train_x.grad is not None

    def test_sdpa_impl_returns_manual_attention_weights_when_requested(self):
        """SDPA 설정이어도 attention weight 요청 시 manual 경로로 weight를 반환하는지 확인한다."""
        from attention import MultiHeadAttention

        batch_size, seq_len, d_model = 1, 4, 8
        n_heads = 2
        mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, drop_rate=0.0, attention_impl="sdpa")
        x = torch.randn(batch_size, seq_len, d_model)

        out, attn_weights = mha(x, causal_mask=True, return_attention_weights=True)

        assert out.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert torch.allclose(attn_weights[0, :, i, j], torch.zeros(n_heads))
