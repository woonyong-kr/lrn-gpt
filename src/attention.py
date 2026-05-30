# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention_score_matrix(queries: torch.Tensor, keys: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    """Q와 K를 비교해 attention score matrix를 만듭니다."""
    scores = queries @ keys.transpose(-2, -1)
    if scale is None:
        return scores
    return scores / scale


def apply_causal_score_mask(scores: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
    """미래 token 위치의 score를 -inf로 바꿉니다."""
    seq_len = scores.size(-1) if seq_len is None else seq_len
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device), diagonal=1)
    return scores.masked_fill(mask, float("-inf"))


def normalize_attention_scores(scores: torch.Tensor, dropout: nn.Module | None = None) -> torch.Tensor:
    """score row마다 softmax를 적용해 attention weight를 만듭니다."""
    weights = F.softmax(scores, dim=-1)
    return weights if dropout is None else dropout(weights)


def weighted_value_context(attn_weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """attention weight로 value를 가중합해 context vector를 만듭니다."""
    return attn_weights @ values


def simplified_self_attention(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Q/K/V 가중치 없이 X끼리 비교하는 간소화된 self-attention.

    Returns:
        (context, attention_weights)
    """
    scores = attention_score_matrix(x, x)
    weights = normalize_attention_scores(scores)
    return weighted_value_context(weights, x), weights


def projected_self_attention(x: torch.Tensor, w_query: torch.Tensor, w_key: torch.Tensor, w_value: torch.Tensor, causal_mask: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    X에 W_Q/W_K/W_V를 곱해 Q/K/V를 만든 뒤 self-attention을 계산합니다.

    이 함수는 학습용 예제에서 "셀프 어텐션"과 "코잘 어텐션" 차이를
    같은 연산 흐름으로 보여주기 위한 작은 wrapper입니다.
    """
    queries = x @ w_query
    keys = x @ w_key
    values = x @ w_value
    scores = attention_score_matrix(queries, keys, scale=keys.size(-1) ** 0.5)
    if causal_mask:
        scores = apply_causal_score_mask(scores)
    weights = normalize_attention_scores(scores)
    return weighted_value_context(weights, values), weights


def causal_self_attention(x: torch.Tensor, w_query: torch.Tensor, w_key: torch.Tensor, w_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """projected_self_attention에 causal mask를 켠 GPT식 self-attention."""
    return projected_self_attention(x, w_query, w_key, w_value, causal_mask=True)


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

    def __init__(self, d_model: int, n_heads: int, drop_rate: float = 0.1, qkv_bias: bool = False):
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

    def forward(self, x: torch.Tensor, causal_mask: bool = True, return_attention_weights: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        if x.ndim != 3:
            raise ValueError("MultiHeadAttention input must have shape (B, T, C)")
        seq_len = self._validate_embedding_dim(x)
        queries, keys, values = self._project_qkv(x)
        out, attn_weights = self._run_multi_head_attention(queries, keys, values, seq_len=seq_len, causal_mask=causal_mask)

        if return_attention_weights:
            return out, attn_weights
        return out

    def _validate_embedding_dim(self, x: torch.Tensor) -> int:
        """입력의 마지막 차원이 모델 차원과 같은지 확인하고 seq_len을 반환합니다."""
        _, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}")
        return seq_len

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """입력 X에서 head별 Q, K, V를 만듭니다."""
        queries = self._split_heads(self.W_query(x))
        keys = self._split_heads(self.W_key(x))
        values = self._split_heads(self.W_value(x))
        return queries, keys, values

    def _run_multi_head_attention(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, seq_len: int, causal_mask: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """score -> mask -> softmax -> context -> output projection 흐름을 실행합니다."""
        scores = self._scaled_attention_scores(queries, keys)
        if causal_mask:
            scores = self._apply_causal_mask(scores, seq_len)

        attn_weights = self._attention_weights(scores)
        context = self._context_from_values(attn_weights, values)
        return self._output_projection(context), attn_weights

    def _scaled_attention_scores(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Q와 K의 내적을 head_dim으로 스케일링합니다."""
        return attention_score_matrix(queries, keys, scale=self.head_dim ** 0.5)

    def _apply_causal_mask(self, scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """미래 token 위치의 score를 -inf로 바꿉니다."""
        return apply_causal_score_mask(scores, seq_len)

    def _attention_weights(self, scores: torch.Tensor) -> torch.Tensor:
        """score row마다 softmax를 적용해 attention weight를 만듭니다."""
        return normalize_attention_scores(scores, dropout=self.attn_dropout)

    def _context_from_values(self, attn_weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """attention weight로 value를 가중합하고 head를 다시 합칩니다."""
        context = weighted_value_context(attn_weights, values)
        return self._merge_heads(context)

    def _output_projection(self, context: torch.Tensor) -> torch.Tensor:
        """합쳐진 head 출력을 다시 d_model 공간으로 보냅니다."""
        return self.resid_dropout(self.out_proj(context))

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
