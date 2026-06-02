# -*- coding: utf-8 -*-
"""Step 03. Use the GPT-2 BPE tokenizer from tiktoken.

BPE tokenizer는 학습 단계에서 vocab + merge rules를 이미 만들어 둔다.
이 파일은 "사용 단계"다. 새 입력이 왔다고 pair 빈도를 다시 세는 것이 아니라,
이미 정해진 규칙으로 token ID sequence를 만든다.

책에서 직접 BPE를 구현하기 전에도, tiktoken을 사용하면 실제 GPT 계열 tokenizer가
입력을 어떤 token ID들로 바꾸는지 확인할 수 있다.
"""

import tiktoken


def main() -> None:
    # gpt2 encoding에는 GPT-2에서 쓰는 byte-level BPE vocab과 merge rules가 들어 있다.
    # 이 객체를 가져오는 것은 "학습"이 아니라 이미 학습된 tokenizer를 불러오는 일이다.
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

    # tokenizer.n_vocab은 출력층이 맞혀야 할 class 개수와도 연결된다.
    # GPT가 다음 token을 예측할 때는 이 vocab 전체에 대한 점수를 낸다.
    print("vocab size:", tokenizer.n_vocab)
    print("입력 문장:")
    print(text)
    print("token ids:")
    print(ids)

    # decode는 token ID sequence를 다시 사람이 읽는 문자열로 되돌린다.
    # 생성 단계에서는 모델이 고른 다음 token ID들을 계속 decode해 최종 답변을 만든다.
    print("decode:")
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
