# -*- coding: utf-8 -*-
"""Step 02. Build a toy vocabulary and test SimpleTokenizerV1/V2.

여기서 처음으로 "문자열 조각 -> token ID"가 생긴다.
하지만 이것도 아직 vector가 아니다. 391, 1500 같은 숫자는 embedding table의
행을 찾기 위한 주소라고 생각하면 된다.

책의 이 단계는 tokenizer가 하는 일을 손으로 확인하는 구간이다.
vocab은 신경망 가중치가 아니라, 문자열과 정수 ID를 연결하는 lookup table이다.
"""

from common import (
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    build_vocab,
    read_the_verdict,
    regex_tokenize,
)


def main() -> None:
    raw_text = read_the_verdict()
    preprocessed = regex_tokenize(raw_text)

    # vocab은 "문자열 조각 -> 정수 ID" 사전이다.
    # 이 숫자를 보고 의미를 비교하면 안 된다.
    # MNIST의 class 번호처럼 "몇 번째 칸이 정답인가"를 가리키는 번호다.
    vocab = build_vocab(preprocessed)
    print("vocab size:", len(vocab))

    # 앞쪽 vocab 항목을 출력하면 token ID가 어떻게 부여되었는지 볼 수 있다.
    # 지금 toy vocab은 sorted(set(tokens))라서 문자열 정렬 순서대로 ID가 붙는다.
    # 실제 BPE vocab은 병합 학습 결과와 special token 정책에 따라 ID가 정해진다.
    print("\n처음 52개 vocab 항목:")
    for i, item in enumerate(vocab.items()):
        print(item)
        if i > 50:
            break

    tokenizer = SimpleTokenizerV1(vocab)
    text = """It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride. """

    # encode: 문자열 -> token ID 목록
    # decode: token ID 목록 -> 문자열
    # 여기서 같은 token이 두 번 나오면 같은 ID가 두 번 나올 뿐,
    # vocab에 새 token을 또 등록하는 것은 아니다.
    ids = tokenizer.encode(text)
    print("\nSimpleTokenizerV1 ids:")
    print(ids)
    print("decode:")
    print(tokenizer.decode(ids))

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = "<|endoftext|>".join([text1, text2])

    # V2는 vocab에 없는 token을 <|unk|>로 바꾼다.
    # 이 toy tokenizer의 한계가 여기서 보인다.
    # 실제 GPT 계열 BPE는 모르는 단어를 더 작은 byte/subword 조각으로 쪼개서
    # <|unk|> 문제를 크게 줄인다.
    tokenizer_v2 = SimpleTokenizerV2(vocab)
    ids = tokenizer_v2.encode(text)
    print("\nSimpleTokenizerV2 입력:")
    print(text)
    print("ids:")
    print(ids)

    # decode 결과에 <|unk|>가 보이면 그 위치의 원래 문자열 정보는 사라진 것이다.
    # 이 한계를 보면 BPE가 왜 "낯선 단어를 작은 조각으로 표현"하려 하는지 이해된다.
    print("decode:")
    print(tokenizer_v2.decode(ids))


if __name__ == "__main__":
    main()
