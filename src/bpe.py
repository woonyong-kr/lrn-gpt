# -*- coding: utf-8 -*-
"""UTF-8 byte sequence에 BPE merge rule을 적용하는 교육용 tokenizer."""

import json
from pathlib import Path
from typing import Any

try:
    from .guards import fail, require
except ImportError:
    from guards import fail, require

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

"""
Token ID layout:
- 0~3: special tokens
- 4~259: raw UTF-8 byte tokens
- 260~: BPE merge tokens
"""
SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """UTF-8 byte-level BPE tokenizer."""

    def __init__(self, vocab_size: int = 3000, min_frequency: int = 1):
        require(min_frequency > 0, "min_frequency must be positive")
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self._reset_vocab()

    def _init_special_tokens(self):
        """기존 테스트 호환용 wrapper."""
        self._reset_vocab()

    def _reset_vocab(self):
        """기본 special token과 UTF-8 byte token으로 vocabulary를 리셋합니다."""
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

        for token, idx in SPECIAL_IDS.items():
            self.id_to_token[idx] = token
            self.token_to_id[token] = idx

        for byte_value in range(NUM_BYTES):
            token = bytes([byte_value])
            self.id_to_token[BYTE_OFFSET + byte_value] = token
            self.token_to_id[token] = BYTE_OFFSET + byte_value

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        BPE 학습은 역전파가 아니라 빈도 기반 규칙 학습입니다.
        현재 sequence에서 가장 자주 등장한 인접 pair를 새 token으로 만들고,
        그 pair를 새 ID로 치환하는 일을 반복합니다.

        멈추는 조건:
        - 목표 vocab_size에 도달
        - 더 이상 pair가 없음
        - 가장 많이 나온 pair도 min_frequency보다 적게 등장
        """
        self._reset_vocab()
        if self.vocab_size <= len(self.id_to_token):
            return self

        sequence = [BYTE_OFFSET + byte for byte in corpus.encode("utf-8")]

        while len(self.id_to_token) < self.vocab_size:
            pair = self._select_best_pair(sequence, self.min_frequency)
            if pair is None:
                break
            new_id = len(self.id_to_token)
            self.merges.append(pair)
            self.id_to_token[new_id] = pair
            self.token_to_id[pair] = new_id
            sequence = self._replace_pair(sequence, pair, new_id)

        return self

    def save(self, path: str | Path):
        """Vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        path = Path(path)
        payload = {
            "vocab_size": self.vocab_size,
            "id_to_token": [
                {"id": token_id, **self._serialize_token(token)}
                for token_id, token in sorted(self.id_to_token.items())
            ],
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str | Path):
        """save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.vocab_size = int(payload["vocab_size"])
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = [tuple(pair) for pair in payload["merges"]]

        for entry in payload["id_to_token"]:
            token_id = int(entry["id"])
            token = self._deserialize_token(entry)
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        return self

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """문자열을 token ID 리스트로 변환합니다.

        학습된 merge rule을 순서대로 적용해야 train 때 만든 tokenization
        기준과 encode 때 기준이 같아집니다.
        """
        ids = [BYTE_OFFSET + byte for byte in text.encode("utf-8")]
        for pair in self.merges:
            new_id = self.token_to_id.get(pair)
            if new_id is not None:
                ids = self._replace_pair(ids, pair, new_id)

        if add_bos_eos:
            return [self.get_bos_id(), *ids, self.get_eos_id()]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        byte_values: list[int] = []
        text_pieces: list[str] = []

        def flush_bytes() -> None:
            if not byte_values:
                return
            text_pieces.append(bytes(byte_values).decode("utf-8", errors="replace"))
            byte_values.clear()

        for token_id in ids:
            token = self.id_to_token.get(int(token_id))
            if token is None:
                if not skip_special:
                    flush_bytes()
                    text_pieces.append(UNK_TOKEN)
                continue
            if isinstance(token, str):
                if skip_special and token in SPECIAL_TOKENS:
                    continue
                flush_bytes()
                text_pieces.append(token)
                continue
            byte_values.extend(self._expand_to_bytes(int(token_id)))

        flush_bytes()
        return "".join(text_pieces)

    @staticmethod
    def _replace_pair(sequence: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """sequence 안의 pair를 왼쪽부터 겹치지 않게 새 ID로 치환합니다."""
        result: list[int] = []
        i = 0
        while i < len(sequence):
            if i < len(sequence) - 1 and (sequence[i], sequence[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(sequence[i])
                i += 1
        return result

    @staticmethod
    def _select_best_pair(sequence: list[int], min_frequency: int = 1) -> tuple[int, int] | None:
        """빈도, 최초 등장 위치, token ID 순서로 병합할 pair를 선택합니다."""
        if len(sequence) < 2:
            return None

        counts: dict[tuple[int, int], int] = {}
        first_seen: dict[tuple[int, int], int] = {}
        for idx in range(len(sequence) - 1):
            pair = (sequence[idx], sequence[idx + 1])
            counts[pair] = counts.get(pair, 0) + 1
            first_seen.setdefault(pair, idx)

        best_pair = min(counts, key=lambda pair: (-counts[pair], first_seen[pair], pair[0], pair[1]))
        if counts[best_pair] < min_frequency:
            return None
        return best_pair

    def _expand_to_bytes(self, token_id: int) -> list[int]:
        """Merge token을 원본 byte 값 리스트로 재귀적으로 펼칩니다."""
        token = self.id_to_token[token_id]
        if isinstance(token, bytes):
            return list(token)
        if isinstance(token, tuple):
            left, right = token
            return self._expand_to_bytes(left) + self._expand_to_bytes(right)
        if isinstance(token, str):
            return list(token.encode("utf-8"))
        fail(f"Unsupported token type: {type(token)!r}", TypeError)

    @staticmethod
    def _serialize_token(token: str | bytes | tuple[int, int]) -> dict[str, Any]:
        if isinstance(token, str):
            return {"type": "str", "value": token}
        if isinstance(token, bytes):
            return {"type": "bytes", "value": list(token)}
        if isinstance(token, tuple):
            return {"type": "pair", "value": list(token)}
        fail(f"Unsupported token type: {type(token)!r}", TypeError)

    @staticmethod
    def _deserialize_token(entry: dict[str, Any]) -> str | bytes | tuple[int, int]:
        token_type = entry["type"]
        value = entry["value"]
        if token_type == "str":
            return str(value)
        if token_type == "bytes":
            return bytes(value)
        if token_type == "pair":
            return tuple(value)
        fail(f"Unknown token type: {token_type}")
