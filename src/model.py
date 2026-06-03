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


class ExactGELU(nn.Module):
    """PyTorch의 exact GELU를 사용하는 비교용 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="none")


class QuickGELU(nn.Module):
    """일부 CLIP 계열 구현에서 쓰는 빠른 GELU 근사."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class SquaredReLU(nn.Module):
    """ReLU 출력을 제곱해 양수 영역의 곡률을 키우는 비교용 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.square(F.relu(x))


class IdentityActivation(nn.Module):
    """활성화 함수를 끈 선형 FFN 비교용 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _normalize_activation_name(name: str) -> str:
    return name.lower().replace("-", "_")


def get_activation(name: str) -> nn.Module:
    """실험 이름으로 FFN 활성화 함수를 선택합니다."""
    normalized = _normalize_activation_name(name)
    activations = {
        "gelu": GELU,
        "gelu_tanh": GELU,
        "gelu_exact": ExactGELU,
        "quick_gelu": QuickGELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "swiglu": nn.SiLU,
        "geglu": GELU,
        "mish": nn.Mish,
        "tanh": nn.Tanh,
        "identity": IdentityActivation,
        "linear": IdentityActivation,
        "squared_relu": SquaredReLU,
    }
    require(normalized in activations, f"Unknown activation_name: {name}")
    return activations[normalized]()


def is_gated_activation(name: str) -> bool:
    """SwiGLU/GEGLU처럼 FFN 내부에서 gate와 value를 나누는 활성화인지 확인합니다."""
    return _normalize_activation_name(name) in {"swiglu", "geglu"}


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> activation -> Linear, with configurable dropout order."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4, activation_name: str = "gelu", dropout_position: str = "after_output"):
        super().__init__()
        require(mult > 0, "mult must be positive")
        require(dropout_position in {"after_output", "after_activation", "none"}, "dropout_position must be after_output, after_activation, or none")
        hidden_dim = mult * d_model
        self.activation_name = _normalize_activation_name(activation_name)
        self.is_gated = is_gated_activation(activation_name)
        self.linear1 = nn.Linear(d_model, hidden_dim * 2 if self.is_gated else hidden_dim)
        self.activation = nn.SiLU() if self.activation_name == "swiglu" else GELU() if self.activation_name == "geglu" else get_activation(activation_name)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_position = dropout_position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FeedForward 네트워크를 통과시킵니다."""
        x = self.linear1(x)
        if self.is_gated:
            value, gate = x.chunk(2, dim=-1)
            x = value * self.activation(gate)
        else:
            x = self.activation(x)
        if self.dropout_position == "after_activation":
            x = self.dropout(x)
        x = self.linear2(x)
        if self.dropout_position == "after_output":
            x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(self, d_model: int, n_heads: int, drop_rate: float = 0.1, qkv_bias: bool = False, ffn_mult: int = 4, norm_first: bool = False, norm_eps: float = 1e-5, activation_name: str = "gelu", ffn_dropout_position: str = "after_output", attention_impl: str = "manual"):
        super().__init__()
        self.att = MultiHeadAttention(d_model, n_heads, drop_rate, qkv_bias, attention_impl=attention_impl)
        self.ffn = FeedForward(d_model, dropout=drop_rate, mult=ffn_mult, activation_name=activation_name, dropout_position=ffn_dropout_position)
        self.ln1 = LayerNorm(d_model, eps=norm_eps)
        self.ln2 = LayerNorm(d_model, eps=norm_eps)
        self.norm_first = norm_first

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """attention과 ffn을 residual connection으로 연결합니다."""
        if self.norm_first:
            x = x + self.att(self.ln1(x), causal_mask=causal_mask)
            return x + self.ffn(self.ln2(x))

        attention_out = self.att(x, causal_mask=causal_mask)
        x = self.ln1(x + attention_out)
        ffn_out = self.ffn(x)
        return self.ln2(x + ffn_out)


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = normalize_config(config)
        set_debug_seed(debug=self.config.get("debug", DEFAULT_DEBUG), seed=self.config.get("seed", DEFAULT_SEED))

        vocab_size = self.config["vocab_size"]
        context_length = self.config["context_length"]
        emb_dim = self.config["emb_dim"]
        n_heads = self.config["n_heads"]
        n_layers = self.config["n_layers"]
        drop_rate = self.config.get("drop_rate", 0.1)
        qkv_bias = self.config.get("qkv_bias", False)
        ffn_mult = self.config.get("ffn_mult", 4)
        norm_first = self.config.get("norm_first", False)
        norm_eps = self.config.get("norm_eps", 1e-5)
        activation_name = self.config.get("activation_name", self.config.get("activation", "gelu"))
        ffn_dropout_position = self.config.get("ffn_dropout_position", "after_output")
        attention_impl = self.config.get("attention_impl", "manual")
        tie_embeddings = self.config.get("tie_embeddings", False)

        self.embedding = InputEmbedding(vocab_size, emb_dim, context_length, drop_rate)
        self.blocks = nn.ModuleList([TransformerBlock(emb_dim, n_heads, drop_rate=drop_rate, qkv_bias=qkv_bias, ffn_mult=ffn_mult, norm_first=norm_first, norm_eps=norm_eps, activation_name=activation_name, ffn_dropout_position=ffn_dropout_position, attention_impl=attention_impl) for _ in range(n_layers)])
        self.final_norm = LayerNorm(emb_dim, eps=norm_eps)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.apply(lambda module: init_gpt_weights(module, self.config.get("init_std", 0.02)))
        if tie_embeddings:
            self.lm_head.weight = self.embedding.token_embedding.weight

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
