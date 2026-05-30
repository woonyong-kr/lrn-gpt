# -*- coding: utf-8 -*-
"""실험 가능한 GPT 설정과 초기화 유틸리티."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random

import numpy as np
import torch
import torch.nn as nn


@dataclass(slots=True)
class GPTConfig:
    """작은 GPT 모델의 폭, 깊이, 초기화 방식을 한곳에서 관리합니다."""

    vocab_size: int
    context_length: int
    emb_dim: int = 256
    n_heads: int = 8
    n_layers: int = 6
    drop_rate: float = 0.1
    qkv_bias: bool = False
    ffn_mult: int = 4
    norm_first: bool = False
    init_std: float = 0.02
    seed: int | None = None

    def to_dict(self) -> dict:
        """기존 dict 기반 코드와 함께 쓰기 위한 변환 함수."""
        return asdict(self)


def normalize_config(config: dict | GPTConfig) -> dict:
    """dict와 dataclass 설정을 모두 GPTModel이 사용할 수 있는 dict로 맞춥니다."""
    if isinstance(config, GPTConfig):
        return config.to_dict()
    return dict(config)


def set_seed(seed: int | None, deterministic: bool = False) -> None:
    """재현 가능한 디버깅을 위해 Python, NumPy, PyTorch seed를 함께 고정합니다."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def init_gpt_weights(module: nn.Module, init_std: float = 0.02) -> None:
    """GPT 계열에서 흔히 쓰는 작은 정규분포 초기화를 적용합니다."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=init_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=init_std)
