# -*- coding: utf-8 -*-
"""Step 05. Convert token IDs into token embedding vectors.

token ID는 숫자지만 숫자 크기 자체에는 의미가 없다.
이 파일의 목적은 "ID를 신경망이 계산할 수 있는 vector로 바꾸는 순간"을 보는 것이다.
"""

import torch


def main() -> None:
    # input_ids는 token ID다. 숫자 크기 자체에 의미가 있는 것이 아니라
    # embedding table에서 몇 번째 행을 꺼낼지 가리키는 주소다.
    #
    # 예를 들어 ID 3은 "3이라는 값이 의미 있다"가 아니라
    # embedding_layer.weight[3]을 꺼내라는 뜻이다.
    input_ids = torch.tensor([2, 3, 5, 1])
    vocab_size = 6
    output_dim = 3

    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    print("embedding table weight shape:")
    print(embedding_layer.weight.shape)
    print(embedding_layer.weight)

    print("\ntoken ID 3의 embedding vector:")
    print(embedding_layer(torch.tensor([3])))

    # [2, 3, 5, 1]을 넣으면 4개의 token 위치마다 vector가 하나씩 나온다.
    # 즉 결과 shape는 (token 개수, output_dim)이 된다.
    print("\ninput_ids 전체 embedding:")
    print(embedding_layer(input_ids))


if __name__ == "__main__":
    main()
