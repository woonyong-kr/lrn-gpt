# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from .model import GPTModel
    from .guards import require
except ImportError:
    from model import GPTModel
    from guards import require


def make_sentiment_dataset(train_tsv_path: str | Path, test_tsv_path: str | Path | None = None, val_ratio: float = 0.08, seed: int = 42, output_dir: str | Path | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """
    NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    require(0.0 <= val_ratio < 1.0, "val_ratio must be in [0, 1)")
    train_data = _read_nsmc_tsv(train_tsv_path)
    rng = random.Random(seed)
    rng.shuffle(train_data)

    val_size = int(len(train_data) * val_ratio)
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]
    test_data = [] if test_tsv_path is None else _read_nsmc_tsv(test_tsv_path)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_path / "train.jsonl", train_data)
        _write_jsonl(output_path / "validation.jsonl", val_data)
        _write_jsonl(output_path / "test.jsonl", test_data)

    return train_data, val_data, test_data


def _read_nsmc_tsv(path: str | Path) -> list[dict]:
    """NSMC TSV에서 document/label만 뽑고 빈 리뷰를 제거합니다."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            text = (row.get("document") or "").strip()
            label = (row.get("label") or "").strip()
            if not text or label not in {"0", "1"}:
                continue
            rows.append({"text": text, "label": int(label)})
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """학습/검증/테스트 split을 JSONL로 저장합니다."""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(self, data: list[dict], tokenizer, max_length: int = 128, pad_id: int | None = None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id
        self.samples = [self._encode_sample(row) for row in data]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """미리 tokenization한 input_ids와 label을 반환합니다."""
        return self.samples[idx]

    def _encode_sample(self, row: dict) -> tuple[torch.Tensor, int]:
        """Dataset 생성 시 한 번만 text를 token ID tensor로 변환합니다."""
        ids = self.tokenizer.encode(row["text"], add_bos_eos=True)
        ids = ids[: self.max_length]
        ids = ids + [self.pad_id] * (self.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long), int(row["label"])


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(self, gpt_model: GPTModel, num_labels: int = 2, drop_rate: float = 0.1):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        self.pad_id = 0
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        hidden = self.gpt.forward_hidden(input_ids)
        lengths = (input_ids != self.pad_id).sum(dim=1).clamp(min=1)
        last_token_indices = lengths - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_token_indices]
        logits = self.classifier(self.dropout(pooled))

        if labels is None:
            return logits

        loss = nn.functional.cross_entropy(logits, labels)
        return loss, logits


def train_epoch_sentiment(model: GPTForSequenceClassification, train_loader, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    """감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    model.to(device)
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        loss, logits = model(input_ids, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        return 0.0, 0.0
    return total_loss / total_examples, total_correct / total_examples


def evaluate_sentiment(model: GPTForSequenceClassification, data_loader, device: torch.device) -> tuple[float, float]:
    """감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    was_training = model.training
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            loss, logits = model(input_ids, labels=labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_examples += batch_size

    if was_training:
        model.train()

    if total_examples == 0:
        return 0.0, 0.0
    return total_loss / total_examples, total_correct / total_examples
