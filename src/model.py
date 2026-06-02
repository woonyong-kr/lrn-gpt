# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .attention import MultiHeadAttention
    from .config import DEFAULT_DEBUG, DEFAULT_SEED, init_gpt_weights, normalize_config, set_debug_seed
    from .embeddings import InputEmbedding
    from .guards import require
except ImportError:
    from attention import MultiHeadAttention
    from config import DEFAULT_DEBUG, DEFAULT_SEED, init_gpt_weights, normalize_config, set_debug_seed
    from embeddings import InputEmbedding
    from guards import require


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """tanh 근사식으로 GELU를 계산합니다."""
        return 0.5 * x * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device, dtype=x.dtype)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        require(mult > 0, "mult must be positive")
        hidden_dim = mult * d_model
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.activation = GELU()
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.flow_steps = ("linear1", "activation", "linear2", "dropout")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FeedForward 네트워크를 통과시킵니다."""
        return self._run_projection_stack(x)

    def _run_projection_stack(self, x: torch.Tensor) -> torch.Tensor:
        """flow_steps에 적힌 순서대로 FFN sub-layer를 실행합니다."""
        for step_name in self.flow_steps:
            x = self._run_step(step_name, x)
        return x

    def _run_step(self, step_name: str, x: torch.Tensor) -> torch.Tensor:
        """step 이름으로 실제 layer를 찾아 실행합니다."""
        layer = getattr(self, step_name)
        return layer(x)


class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(self, d_model: int, n_heads: int, drop_rate: float = 0.1, qkv_bias: bool = False, ffn_mult: int = 4, norm_first: bool = False):
        super().__init__()
        self.att = MultiHeadAttention(d_model, n_heads, drop_rate, qkv_bias)
        self.ffn = FeedForward(d_model, dropout=drop_rate, mult=ffn_mult)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.norm_first = norm_first

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """attention과 ffn을 residual connection으로 연결합니다."""
        if self.norm_first:
            return self._forward_pre_norm(x, causal_mask)

        return self._forward_post_norm(x, causal_mask)

    def _forward_pre_norm(self, x: torch.Tensor, causal_mask: bool) -> torch.Tensor:
        """Pre-LN block: LayerNorm을 각 sub-layer 앞에서 적용합니다."""
        x = x + self.att(self.ln1(x), causal_mask=causal_mask)
        return x + self.ffn(self.ln2(x))

    def _forward_post_norm(self, x: torch.Tensor, causal_mask: bool) -> torch.Tensor:
        """Post-LN block: residual add 뒤에 LayerNorm을 적용합니다."""
        x = self._attention_residual_post_norm(x, causal_mask)
        return self._ffn_residual_post_norm(x)

    def _attention_residual_post_norm(self, x: torch.Tensor, causal_mask: bool) -> torch.Tensor:
        """X + MultiHeadAttention(X)를 만든 뒤 첫 번째 LayerNorm을 적용합니다."""
        attention_out = self.att(x, causal_mask=causal_mask)
        return self.ln1(x + attention_out)

    def _ffn_residual_post_norm(self, x: torch.Tensor) -> torch.Tensor:
        """LayerNorm 결과 + FFN 결과를 만든 뒤 두 번째 LayerNorm을 적용합니다."""
        ffn_out = self.ffn(x)
        return self.ln2(x + ffn_out)


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = normalize_config(config)
        set_debug_seed(
            debug=self.config.get("debug", DEFAULT_DEBUG),
            seed=self.config.get("seed", DEFAULT_SEED),
        )

        vocab_size = self.config["vocab_size"]
        context_length = self.config["context_length"]
        emb_dim = self.config["emb_dim"]
        n_heads = self.config["n_heads"]
        n_layers = self.config["n_layers"]
        drop_rate = self.config.get("drop_rate", 0.1)
        qkv_bias = self.config.get("qkv_bias", False)
        ffn_mult = self.config.get("ffn_mult", 4)
        norm_first = self.config.get("norm_first", False)

        self.embedding = InputEmbedding(vocab_size, emb_dim, context_length, drop_rate)
        self.blocks = nn.ModuleList([TransformerBlock(emb_dim, n_heads, drop_rate=drop_rate, qkv_bias=qkv_bias, ffn_mult=ffn_mult, norm_first=norm_first) for _ in range(n_layers)])
        self.final_norm = LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.apply(lambda module: init_gpt_weights(module, self.config.get("init_std", 0.02)))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        logits를 만들고, targets가 있으면 cross entropy loss도 함께 반환합니다.

        Returns:
            targets가 None이면 logits
            targets가 있으면 (loss, logits)
        """
        hidden = self.forward_hidden(idx)
        logits = self.lm_head(hidden)

        if targets is None:
            return logits

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss, logits

    def forward_hidden(self, idx: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """LM head 직전 hidden state를 반환합니다. 분류 head 재사용용입니다."""
        x = self.embedding(idx)
        for block in self.blocks:
            x = block(x, causal_mask=causal_mask)
        return self.final_norm(x)


def generate_text_simple(model: GPTModel, idx: torch.Tensor, max_new_tokens: int, context_size: int) -> torch.Tensor:
    """greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)
            if isinstance(logits, tuple):
                logits = logits[1]
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, next_id), dim=1)
    return idx
