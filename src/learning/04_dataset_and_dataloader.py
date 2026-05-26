# -*- coding: utf-8 -*-
"""Step 04. Create input/target pairs for next-token prediction.

여기가 GPT 학습을 MNIST처럼 이해하기 좋은 지점이다.
MNIST가 0~9 중 정답 class를 맞히듯, GPT는 vocab 전체 중 다음 token ID를
맞힌다. target은 E[20] 같은 벡터가 아니라 정수 token ID 20이다.
"""

from common import GPTDatasetV1, create_dataloader_v1, read_the_verdict
import tiktoken


def main() -> None:
    raw_text = read_the_verdict()
    tokenizer = tiktoken.get_encoding("gpt2")

    # 원문 token이 [10, 20, 30, 40]이면:
    # input  = [10, 20, 30]
    # target = [20, 30, 40]
    #
    # 내가 붙잡아야 할 핵심:
    #   10 자체를 맞히는 게 아니다.
    #   10 다음에 20이 와야 한다는 것을 맞힌다.
    #   정답은 벡터가 아니라 "다음 token ID"다.
    dataset = GPTDatasetV1(raw_text, tokenizer, max_length=4, stride=1)
    input_ids, target_ids = dataset[0]
    print("첫 번째 dataset input:")
    print(input_ids)
    print("첫 번째 dataset target:")
    print(target_ids)

    # stride=1은 한 칸씩 밀면서 샘플을 만든다.
    # 문맥이 많이 겹치므로 촘촘히 배우지만, 같은 구간을 반복해서 보는 느낌이 생긴다.
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=1,
        max_length=4,
        stride=1,
        shuffle=False,
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    second_batch = next(data_iter)
    print("\nstride=1 첫 번째 batch:")
    print(first_batch)
    print("stride=1 두 번째 batch:")
    print(second_batch)

    # stride=max_length는 겹치지 않게 자른다.
    # 과적합을 완전히 막는 장치는 아니지만 중복 샘플은 줄어든다.
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=8,
        max_length=4,
        stride=4,
        shuffle=False,
    )
    inputs, targets = next(iter(dataloader))
    print("\nstride=4 입력:")
    print(inputs)
    print("stride=4 타깃:")
    print(targets)


if __name__ == "__main__":
    main()
