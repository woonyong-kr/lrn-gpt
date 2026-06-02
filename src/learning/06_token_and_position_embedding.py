# -*- coding: utf-8 -*-
"""Step 06. Add token embeddings and absolute position embeddings.

이 파일은 "같은 token ID라도 문장 안 위치가 다르면 어떻게 구분하지?"에 대한
실험이다. token embedding은 "무슨 token인가"를 담고, position embedding은
"몇 번째 자리인가"를 담는다.

책에서는 token embedding만으로는 순서 정보가 없으므로 position embedding을 더한다.
Transformer는 RNN처럼 순서대로 한 token씩 읽는 구조가 아니라 한 번에 여러 위치를
처리하므로, 위치 정보를 입력 벡터에 명시적으로 넣어줘야 한다.
"""

import torch
import tiktoken

from common import create_dataloader_v1, read_the_verdict


def main() -> None:
    raw_text = read_the_verdict()
    tokenizer = tiktoken.get_encoding("gpt2")

    # batch_size:
    #   한 번에 볼 학습 샘플 개수
    # max_length:
    #   한 샘플 안의 token 개수, 즉 context length
    # output_dim:
    #   token vector와 position vector의 차원
    #   두 vector를 더하려면 차원이 같아야 한다.
    batch_size = 8
    max_length = 4
    output_dim = 256

    dataloader = create_dataloader_v1(raw_text, batch_size=batch_size, max_length=max_length, stride=max_length, shuffle=False)
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

    # token_embedding_layer:
    #   "token ID -> token vector" lookup table이다.
    #   tokenizer.n_vocab개의 token마다 output_dim 길이의 vector를 가진다.
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

    # pos_embedding_layer:
    #   "position index -> position vector" lookup table이다.
    #   여기서는 위치가 0, 1, 2, 3 네 개이므로 row가 4개다.
    #
    # token embedding table과 별개의 table이다.
    # token ID 40의 vector와 position 0의 vector는 서로 다른 의미를 가진다.
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    # torch.arange(context_length)는 tensor([0, 1, 2, 3])을 만든다.
    # 이 위치 번호들을 position embedding table에 넣어 위치 vector 4개를 꺼낸다.
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print("\npos_embeddings shape:")
    print(pos_embeddings.shape)

    # 이제 "무슨 token인가"와 "몇 번째 자리인가"를 더한다.
    # 이 더한 결과가 사용자가 말한 "알파 벡터"에 해당하는 값이고,
    # 보통은 input_embeddings 또는 Transformer 입력 벡터라고 부른다.
    #
    # token_embeddings shape: (8, 4, 256)
    # pos_embeddings shape:   (4, 256)
    #
    # PyTorch broadcasting으로 pos_embeddings가 batch 8개 전체에 더해진다.
    # 그래서 최종 input_embeddings shape는 다시 (8, 4, 256)이다.
    #
    # 이게 Transformer에 들어가는 진짜 입력 벡터라고 보면 된다.
    # 학습할 때 역전파는 token embedding table과 position embedding table 쪽으로도 흐른다.
    input_embeddings = token_embeddings + pos_embeddings
    print("\ninput_embeddings shape:")
    print(input_embeddings.shape)


if __name__ == "__main__":
    main()
