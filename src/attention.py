# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.W_query = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_key = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_value = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(drop_rate)
        self.resid_dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        if x.ndim != 3:
            raise ValueError("MultiHeadAttention input must have shape (B, T, C)")
        batch_size, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}")

        queries = self._split_heads(self.W_query(x))
        keys = self._split_heads(self.W_key(x))
        values = self._split_heads(self.W_value(x))

        scores = queries @ keys.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        if causal_mask:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context = attn_weights @ values
        context = self._merge_heads(context)
        out = self.resid_dropout(self.out_proj(context))

        if return_attention_weights:
            return out, attn_weights
        return out

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C)를 (B, H, T, head_dim)으로 바꿉니다."""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, head_dim)을 다시 (B, T, C)로 합칩니다."""
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)
