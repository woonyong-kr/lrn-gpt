# -*- coding: utf-8 -*-
"""Shared helpers for the step-by-step LLM learning examples.

각 단계 파일에서 같은 코드를 반복하지 않기 위한 공통 모음이다.
여기 있는 함수들은 학습 개념을 숨기기 위한 것이 아니라, 단계 파일이
"지금 무엇을 배우는지"에 집중하게 만들기 위한 받침대다.
"""

from pathlib import Path
import re
import urllib.request

import torch
import tiktoken
from torch.utils.data import DataLoader, Dataset


URL = (
    "https://raw.githubusercontent.com/rickiepark/"
    "llm-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
)
DATA_DIR = Path(__file__).resolve().parent
FILE_PATH = DATA_DIR / "the-verdict.txt"
TOKEN_PATTERN = r'([,.:;?_!"()\']|--|\s)'
SPECIAL_TOKENS = ["<|endoftext|>", "<|unk|>"]


def download_the_verdict(file_path: Path = FILE_PATH) -> Path:
    """Download the sample text only when it is not already present."""
    if not file_path.exists():
        urllib.request.urlretrieve(URL, file_path)
    return file_path


def read_the_verdict() -> str:
    """Return the raw text used throughout the examples."""
    file_path = download_the_verdict()
    return file_path.read_text(encoding="utf-8")


def regex_tokenize(text: str) -> list[str]:
    """Split text into words, punctuation, and separators for the toy tokenizer."""
    preprocessed = re.split(TOKEN_PATTERN, text)
    return [item.strip() for item in preprocessed if item.strip()]


def build_vocab(tokens: list[str]) -> dict[str, int]:
    """
    Build a toy vocabulary.

    This is not BPE. It simply removes duplicates, sorts tokens, then assigns IDs.
    """
    all_tokens = sorted(set(tokens))
    all_tokens.extend(SPECIAL_TOKENS)
    return {token: integer for integer, token in enumerate(all_tokens)}


class SimpleTokenizerV1:
    """A minimal tokenizer that fails when it sees an unknown token."""

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = regex_tokenize(text)
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        return re.sub(r'\s+([,.?!"()\'])', r"\1", text)


class SimpleTokenizerV2:
    """A toy tokenizer that maps unknown tokens to <|unk|>."""

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = regex_tokenize(text)
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        return re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)


class GPTDatasetV1(Dataset):
    """
    Create input/target chunks for next-token prediction.

    If token_ids are [10, 20, 30, 40] and max_length is 3:
    - input  = [10, 20, 30]
    - target = [20, 30, 40]

    The target is the next token ID, not the next token embedding vector.
    """

    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_simple_dataloader_v1(
    txt: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader with the toy tokenizer."""
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader with the GPT-2 BPE tokenizer from tiktoken."""
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
