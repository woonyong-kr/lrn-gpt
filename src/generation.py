"""Autoregressive sampling for the runnable language model."""

import torch
from .model import GPTModel


def generate(model: GPTModel, idx: torch.Tensor, max_new_tokens: int, context_size: int, temperature: float = 1.0, top_k: int | None = None, eos_id: int | None = None) -> torch.Tensor:
    """temperature와 top-k 샘플링을 지원하는 생성 함수입니다."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    was_training = model.training
    model.eval()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)
            if isinstance(logits, tuple):
                logits = logits[1]
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                top_values, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)

            if eos_id is not None and torch.all(next_id == eos_id):
                break

    if was_training:
        model.train()

    return idx
