# -*- coding: utf-8 -*-
"""Step 03. Use the GPT-2 BPE tokenizer from tiktoken.

BPE tokenizer는 학습 단계에서 vocab + merge rules를 이미 만들어 둔다.
이 파일은 "사용 단계"다. 새 입력이 왔다고 pair 빈도를 다시 세는 것이 아니라,
이미 정해진 규칙으로 token ID sequence를 만든다.
"""

import tiktoken


def main() -> None:
    tokenizer = tiktoken.get_encoding("gpt2")

    # BPE 사용 단계에서는 새 병합 규칙을 학습하지 않는다.
    # "Hello가 많이 나왔으니 지금 새 token을 만들자"가 아니다.
    # 이미 학습된 vocab + merge rule을 입력 문장에 적용한다.
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = "<|endoftext|>".join([text1, text2])

    # 결과 ids는 문장 안 위치마다 놓인 token ID 목록이다.
    # 이 목록이 곧 GPT 모델 학습/사용의 입력 재료가 된다.
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    print("vocab size:", tokenizer.n_vocab)
    print("입력 문장:")
    print(text)
    print("token ids:")
    print(ids)
    print("decode:")
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
