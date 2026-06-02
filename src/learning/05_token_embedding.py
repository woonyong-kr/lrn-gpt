# -*- coding: utf-8 -*-
"""Step 05. Convert token IDs into token embedding vectors.

token ID는 숫자지만 숫자 크기 자체에는 의미가 없다.
이 파일의 목적은 "ID를 신경망이 계산할 수 있는 vector로 바꾸는 순간"을 보는 것이다.

책의 다음 단계는 token ID를 바로 Transformer에 넣지 않고 nn.Embedding을 통과시키는
이유를 확인하는 것이다. token ID는 category 번호라서, 그대로 숫자 계산하면
"ID 5가 ID 3보다 크다" 같은 잘못된 의미가 생긴다.
"""

import torch


def main() -> None:
    # input_ids는 token ID다. 숫자 크기 자체에 의미가 있는 것이 아니라
    # embedding table에서 몇 번째 행을 꺼낼지 가리키는 주소다.
    #
    # 예를 들어 ID 3은 "3이라는 값이 의미 있다"가 아니라
    # embedding_layer.weight[3]을 꺼내라는 뜻이다.
    input_ids = torch.tensor([2, 3, 5, 1])

    # vocab_size=6이면 가능한 token ID는 0, 1, 2, 3, 4, 5다.
    # output_dim=3이면 각 token ID를 3차원 vector로 표현한다.
    # 실제 GPT 모델은 보통 훨씬 큰 embedding dimension을 쓴다.
    vocab_size = 6
    output_dim = 3

    # seed를 고정하면 예제 출력이 매번 같아진다.
    # embedding weight는 학습 전에는 랜덤으로 시작하고, 학습 중 역전파로 업데이트된다.
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    # embedding_layer.weight shape는 (vocab_size, output_dim)이다.
    # 즉 각 token ID마다 하나의 행 vector가 있다.
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
