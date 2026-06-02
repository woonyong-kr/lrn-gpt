# -*- coding: utf-8 -*-
"""Shared helpers for the step-by-step LLM learning examples.

이 파일은 "밑바닥부터 시작하는 LLM" 2장 흐름을 따라가며 반복되는 코드를
한곳에 모은 것이다. 각 단계 파일은 개념을 보여주는 데 집중하고, 실제 다운로드,
toy tokenizer, Dataset, DataLoader 구현은 여기에서 재사용한다.

큰 흐름은 다음과 같다.

1. 원문 텍스트를 가져온다.
2. toy tokenizer용 정규식 전처리로 문자열 조각을 만든다.
3. 문자열 조각에 token ID를 붙여 vocab을 만든다.
4. 텍스트를 token ID sequence로 바꾼다.
5. GPT 학습용 input/target 쌍을 만든다.

여기서 계속 붙잡아야 할 기준:
- vocab은 "문자열 조각 -> token ID" 사전이다.
- token ID는 아직 vector가 아니다.
- embedding vector는 모델 안의 nn.Embedding에서 만들어진다.
- target은 다음 token ID이지, 다음 token의 embedding vector가 아니다.
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
# 모든 예제는 이 폴더 안의 같은 샘플 텍스트를 본다.
# 책에서 사용하는 "the-verdict.txt"를 내려받아 tokenizer와 dataloader 실험에 쓴다.
DATA_DIR = Path(__file__).resolve().parent
FILE_PATH = DATA_DIR / "the-verdict.txt"

# 정규식 괄호 안에 있는 구분자는 결과 리스트에도 남는다.
# 예를 들어 ","나 "." 같은 문장부호도 token 후보로 보려는 의도다.
# 실제 GPT BPE tokenizer는 더 복잡하지만, 이 toy tokenizer는
# "문자열을 조각내고 ID를 붙인다"는 감각을 얻기 위한 예제다.
TOKEN_PATTERN = r'([,.:;?_!"()\']|--|\s)'

# <|endoftext|>는 문서 경계를 표시하는 특수 token,
# <|unk|>는 toy tokenizer가 모르는 token을 만났을 때 쓰는 fallback token이다.
# byte-level BPE에서는 작은 byte 조각으로 쪼갤 수 있어서 <|unk|> 의존이 훨씬 줄어든다.
SPECIAL_TOKENS = ["<|endoftext|>", "<|unk|>"]


def download_the_verdict(file_path: Path = FILE_PATH) -> Path:
    """Download the sample text only when it is not already present.

    이 함수는 학습 데이터를 준비하는 역할만 한다.
    여기서는 tokenizer도 학습하지 않고, GPT 모델도 학습하지 않는다.
    단지 이후 단계에서 모두 같은 원문을 보도록 파일을 확보한다.

    Args:
        file_path: 저장할 위치. 기본값은 src/learning/the-verdict.txt다.

    Returns:
        다운로드되어 있거나 이미 존재하는 텍스트 파일 경로.
    """
    if not file_path.exists():
        urllib.request.urlretrieve(URL, file_path)
    return file_path


def read_the_verdict() -> str:
    """Return the raw text used throughout the examples.

    책의 예제는 raw text에서 시작한다.
    raw text는 아직 token도 아니고 ID도 아니며, 그냥 사람이 읽는 문자열이다.
    이 문자열이 tokenizer를 지나 token ID sequence가 된다.
    """
    file_path = download_the_verdict()
    return file_path.read_text(encoding="utf-8")


def regex_tokenize(text: str) -> list[str]:
    """Split text into words, punctuation, and separators for the toy tokenizer.

    이 함수는 BPE가 아니다. 책 초반부에서 "일단 직접 tokenizer를 만들어 보자"는
    느낌으로 사용하는 단순 전처리다.

    동작:
    - 단어 사이 공백을 기준으로 나눈다.
    - 쉼표, 마침표, 따옴표 같은 문장부호도 token 후보로 남긴다.
    - 공백 자체는 strip 후 빈 문자열로 제거한다.

    예:
        "Hello, world!" -> ["Hello", ",", "world", "!"]

    Returns:
        아직 token ID가 아닌 문자열 token 후보 목록.
    """
    preprocessed = re.split(TOKEN_PATTERN, text)
    return [item.strip() for item in preprocessed if item.strip()]


def build_vocab(tokens: list[str]) -> dict[str, int]:
    """Build a toy vocabulary.

    vocab은 "문자열 token -> 정수 ID" 사전이다.
    여기서 만들어지는 정수 ID는 의미 있는 숫자가 아니다.
    embedding table의 행을 찾기 위한 주소라고 보면 된다.

    이 toy vocab은 BPE처럼 pair 빈도를 세고 병합하지 않는다.
    단순히:
    1. 중복 token을 제거하고
    2. 문자열 순서로 정렬하고
    3. 0부터 차례로 ID를 붙이고
    4. 마지막에 특수 token을 추가한다.

    책에서 이 단계를 먼저 보는 이유는 tokenizer의 가장 기본 형태를 눈으로
    확인하기 위해서다. 실제 GPT 계열은 보통 BPE 계열 tokenizer를 쓴다.
    """
    all_tokens = sorted(set(tokens))
    all_tokens.extend(SPECIAL_TOKENS)
    return {token: integer for integer, token in enumerate(all_tokens)}


class SimpleTokenizerV1:
    """A minimal tokenizer that fails when it sees an unknown token.

    V1은 vocab에 있는 token만 처리한다.
    vocab에 없는 문자열이 들어오면 KeyError가 난다.
    이 불편함을 일부러 확인해야 BPE나 <|unk|>가 왜 필요한지 이해하기 쉽다.
    """

    def __init__(self, vocab: dict[str, int]):
        # str_to_int:
        #   "문자열 token" -> token ID
        # int_to_str:
        #   token ID -> "문자열 token"
        #
        # encode와 decode는 서로 반대 방향 변환이다.
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs using the toy vocab."""
        # 1. 원문 문자열을 token 후보로 자른다.
        # 2. 각 문자열 token을 vocab에서 찾아 정수 ID로 바꾼다.
        # 3. vocab에 없는 token이 있으면 여기서 KeyError가 난다.
        preprocessed = regex_tokenize(text)
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back into readable text."""
        # ID를 문자열 token으로 되돌린 뒤 공백으로 붙인다.
        # 그 다음 문장부호 앞의 불필요한 공백을 제거한다.
        # 이 toy decode는 완벽한 원문 복원이 아니라 개념 확인용이다.
        text = " ".join([self.int_to_str[i] for i in ids])
        return re.sub(r'\s+([,.?!"()\'])', r"\1", text)


class SimpleTokenizerV2:
    """A toy tokenizer that maps unknown tokens to <|unk|>.

    V2의 목적은 "모르는 token을 만나면 어떻게 할까?"를 보여주는 것이다.
    V1은 모르는 token에서 실패하지만, V2는 <|unk|>라는 특수 token ID로 바꾼다.

    다만 <|unk|>는 정보 손실이 크다. "Hello"든 "palace"든 vocab에 없으면
    모두 같은 <|unk|>가 되기 때문이다. BPE가 중요한 이유가 여기서 드러난다.
    BPE는 낯선 단어를 통째로 <|unk|> 처리하기보다 더 작은 조각으로 표현한다.
    """

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Convert unknown string tokens to <|unk|>, then to token IDs."""
        preprocessed = regex_tokenize(text)
        # vocab에 없는 문자열 token은 <|unk|>로 치환한다.
        # 이 치환이 끝난 뒤에는 모든 token이 vocab 안에 있어야 한다.
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back into text, including <|unk|> markers."""
        text = " ".join([self.int_to_str[i] for i in ids])
        return re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)


class GPTDatasetV1(Dataset):
    """
    Create input/target chunks for next-token prediction.

    If token_ids are [10, 20, 30, 40] and max_length is 3:
    - input  = [10, 20, 30]
    - target = [20, 30, 40]

    The target is the next token ID, not the next token embedding vector.

    GPT 사전학습은 "현재까지 본 token들로 다음 token을 맞히기"다.
    원문 token ID가 [10, 20, 30, 40]이면 사람이 label을 따로 붙이지 않아도
    다음 token이 자동으로 정답이 된다.

    MNIST와 비교하면:
    - MNIST target: 0~9 중 정답 class 번호
    - GPT target: vocab 중 다음 token ID 번호
    """

    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        # input_ids와 target_ids는 같은 길이의 tensor 리스트다.
        # 각 input chunk는 모델 입력이고, target chunk는 한 칸 오른쪽으로 밀린 정답이다.
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        # tokenizer.encode가 raw text를 token ID sequence로 바꾼다.
        # 이 순간에도 embedding vector는 아직 생기지 않았다.
        # vector 변환은 이후 nn.Embedding에서 한다.
        token_ids = tokenizer.encode(txt)

        # sliding window:
        # 긴 token sequence에서 max_length만큼 잘라 학습 샘플을 만든다.
        # stride=1이면 한 칸씩 촘촘하게 이동한다.
        # stride=max_length이면 서로 겹치지 않게 이동한다.
        for i in range(0, len(token_ids) - max_length, stride):
            # input_chunk:
            #   모델이 보는 현재 token들
            input_chunk = token_ids[i : i + max_length]

            # target_chunk:
            #   input_chunk보다 한 칸 뒤의 token들
            #   각 위치에서 "다음 token ID"가 무엇인지 알려준다.
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            # PyTorch Dataset은 보통 tensor를 반환한다.
            # token ID는 class index이므로 dtype은 long이 맞다.
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        """Return how many input/target chunks this Dataset contains."""
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one training sample: (input token IDs, target token IDs)."""
        return self.input_ids[idx], self.target_ids[idx]


def create_simple_dataloader_v1(txt: str, tokenizer, batch_size: int = 4, max_length: int = 256, stride: int = 128, shuffle: bool = True, drop_last: bool = True) -> DataLoader:
    """Create a DataLoader with the toy tokenizer.

    DataLoader는 Dataset이 만든 샘플들을 batch로 묶어준다.
    모델 학습에서는 보통 하나의 sample만 처리하지 않고 여러 sample을 batch로 묶어
    병렬 계산한다.
    """
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def create_dataloader_v1(txt: str, batch_size: int = 4, max_length: int = 256, stride: int = 128, shuffle: bool = True, drop_last: bool = True, num_workers: int = 0) -> DataLoader:
    """Create a DataLoader with the GPT-2 BPE tokenizer from tiktoken.

    이 함수는 책의 `create_dataloader_v1` 흐름과 연결된다.
    toy tokenizer 대신 GPT-2 BPE tokenizer를 사용한다는 점만 다르다.

    반환되는 batch의 shape:
    - inputs:  (batch_size, max_length)
    - targets: (batch_size, max_length)

    이후 embedding layer를 통과하면:
    - token_embeddings: (batch_size, max_length, embedding_dim)
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
