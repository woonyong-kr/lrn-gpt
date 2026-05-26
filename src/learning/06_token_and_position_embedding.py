# -*- coding: utf-8 -*-
"""Step 06. Add token embeddings and absolute position embeddings.

이 파일은 "같은 token ID라도 문장 안 위치가 다르면 어떻게 구분하지?"에 대한
실험이다. token embedding은 "무슨 token인가"를 담고, position embedding은
"몇 번째 자리인가"를 담는다.
"""

import torch
import tiktoken

from common import create_dataloader_v1, read_the_verdict


def main() -> None:
    raw_text = read_the_verdict()
    tokenizer = tiktoken.get_encoding("gpt2")

    batch_size = 8
    max_length = 4
    output_dim = 256

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
    )
    inputs, targets = next(iter(dataloader))

    # inputs는 아직 token ID다.
    # shape가 (8, 4)라는 말은 batch 8개, 각 문장 조각이 token 4개라는 뜻이다.
    print("inputs shape:")
    print(inputs.shape)
    print("targets shape:")
    print(targets.shape)

    # token embedding:
    # token ID를 vector로 바꾼다.
    # inputs shape이 (batch_size, max_length)이므로 결과는
    # (batch_size, max_length, output_dim)이 된다.
    #
    # 여기서 같은 token ID가 여러 번 나오면 같은 embedding row를 조회한다.
    # 하지만 아직 위치 정보는 없다.
    torch.manual_seed(123)
    token_embedding_layer = torch.nn.Embedding(tokenizer.n_vocab, output_dim)
    token_embeddings = token_embedding_layer(inputs)
    print("\ntoken_embeddings shape:")
    print(token_embeddings.shape)

    # position embedding:
    # token ID와 무관하게 0번째 자리, 1번째 자리, ... 위치마다 vector를 준다.
    # 여기서는 absolute position embedding이다.
    #
    # pos_embeddings shape가 (4, 256)인 이유:
    #   max_length=4라서 위치가 0, 1, 2, 3 네 개
    #   output_dim=256이라 각 위치 vector가 256차원
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print("\npos_embeddings shape:")
    print(pos_embeddings.shape)

    # 이제 "무슨 token인가"와 "몇 번째 자리인가"를 더한다.
    #
    # token_embeddings shape: (8, 4, 256)
    # pos_embeddings shape:   (4, 256)
    #
    # PyTorch broadcasting으로 pos_embeddings가 batch 8개 전체에 더해진다.
    # 그래서 최종 input_embeddings shape는 다시 (8, 4, 256)이다.
    #
    # 이게 Transformer에 들어가는 진짜 입력 벡터라고 보면 된다.
    input_embeddings = token_embeddings + pos_embeddings
    print("\ninput_embeddings shape:")
    print(input_embeddings.shape)


if __name__ == "__main__":
    main()
