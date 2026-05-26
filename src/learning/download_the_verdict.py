# -*- coding: utf-8 -*-
"""Download and inspect the text sample from llm-from-scratch."""

from pathlib import Path
import re
import urllib.request

import torch
from torch.utils.data import DataLoader, Dataset


URL = (
    "https://raw.githubusercontent.com/rickiepark/"
    "llm-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
)
FILE_PATH = Path(__file__).with_name("the-verdict.txt")


class SimpleTokenizerV1:
    """A minimal tokenizer that maps text tokens to integer IDs."""

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


class SimpleTokenizerV2:
    """A tokenizer that maps unknown tokens to <|unk|>."""

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
        return text


class GPTDatasetV1(Dataset):
    """Create input/target token chunks for next-token prediction."""

    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

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


def create_dataloader_v1(
    txt: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def main() -> None:
    urllib.request.urlretrieve(URL, FILE_PATH)

    with FILE_PATH.open("r", encoding="utf-8") as f:
        raw_text = f.read()

    print("총 문자 개수:", len(raw_text))
    print(raw_text[:99])

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    print(len(preprocessed))
    print(len(preprocessed[:30]))

    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab_size = len(all_tokens)
    print(vocab_size)

    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    for i, item in enumerate(vocab.items()):
        print(item)
        if i > 50:
            break

    tokenizer = SimpleTokenizerV1(vocab)
    text = """It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride. """
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))

    gpt_dataset = GPTDatasetV1(text, tokenizer, max_length=4, stride=1)
    input_ids, target_ids = gpt_dataset[0]
    print(input_ids.tolist())
    print(target_ids.tolist())
    print(tokenizer.decode(input_ids.tolist()))
    print(tokenizer.decode(target_ids.tolist()))

    dataloader = create_dataloader_v1(
        text,
        tokenizer,
        batch_size=2,
        max_length=4,
        stride=1,
        shuffle=False,
        drop_last=True,
    )
    batch_inputs, batch_targets = next(iter(dataloader))
    print(batch_inputs)
    print(batch_targets)
    print(batch_inputs.shape, batch_targets.shape)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = "<|endoftext|>".join([text1, text2])
    print(text)

    tokenizer_v2 = SimpleTokenizerV2(vocab)
    ids = tokenizer_v2.encode(text)
    print(ids)
    print(tokenizer_v2.decode(ids))


if __name__ == "__main__":
    main()
