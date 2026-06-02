# -*- coding: utf-8 -*-
"""Step 07. 랜덤 Q/K/V 가중치로 self-attention을 직접 계산한다.

이 파일은 "어텐션 점수는 뭘 계산하는가?"를 숫자로 확인하는 실행 예제다.

핵심:
- inputs는 이미 token embedding vector라고 가정한다.
- 실제 Transformer는 inputs에 학습 가능한 W_Q, W_K, W_V를 곱해 Q, K, V를 만든다.
- 특정 token의 query가 모든 key와 dot product를 해서 attention score를 만든다.
- score를 softmax로 바꾸면 attention weight가 된다.
- attention weight로 value들을 섞으면 그 token 위치의 context vector가 된다.
"""

import math

import torch


def print_matrix(name: str, tensor: torch.Tensor) -> None:
    """작은 tensor를 shape와 함께 출력한다."""
    print(f"\n{name} shape = {tuple(tensor.shape)}")
    print(tensor)


def softmax_naive(x: torch.Tensor) -> torch.Tensor:
    """softmax를 직접 구현한다.

    이 함수는 교육용이다. 실제 코드에서는 수치 안정성이 더 좋은
    torch.softmax를 쓰는 것이 안전하다.
    """
    # exp(x):
    #   모든 score를 양수로 바꾼다.
    #
    # exp(x).sum(dim=0):
    #   1차원 벡터의 모든 값을 더한다.
    #
    # 각 exp(score)를 전체 합으로 나누면 모든 값의 합이 1이 된다.
    return torch.exp(x) / torch.exp(x).sum(dim=0)


def compute_naive_attention_for_journey(inputs: torch.Tensor, words: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q/K/V 없이 입력 벡터 자체로 'journey'의 attention을 계산한다.

    이 함수는 책에서 Q/K/V로 가기 전에 보여주는 간소화된 attention이다.
    실제 Transformer에서는 values 대신 inputs를 그대로 섞지 않고,
    inputs @ W_V로 만든 values를 섞는다.
    """
    print("\n" + "=" * 80)
    print("CASE 0. Naive attention without W_Q/W_K/W_V")
    print("=" * 80)

    # 여기서는 두 번째 token인 "journey"의 입력 벡터를 query처럼 사용한다.
    # 아직 W_Q를 곱하지 않았으므로 실제 Transformer의 query는 아니다.
    query = inputs[1]
    print("\nquery = inputs[1]  # 'journey' input vector")
    print(query)

    # attn_scores_2:
    #   "journey" 벡터와 모든 입력 token 벡터를 dot product한 raw score다.
    #
    # inputs.shape[0]은 token 개수다. 여기서는 6개 token이므로 6칸을 만든다.
    attn_scores_2 = torch.empty(inputs.shape[0])
    print("\nDot products: score_i = x_i dot query")
    for i, (word, x_i) in enumerate(zip(words, inputs)):
        attn_scores_2[i] = torch.dot(x_i, query)
        print(f"{word:8s}: dot({x_i.tolist()}, {query.tolist()}) = {attn_scores_2[i].item(): .4f}")

    # 직접 구현한 softmax와 PyTorch 내장 softmax를 비교한다.
    # 둘 다 raw score를 합이 1인 attention weight로 바꾼다.
    attn_weights_2_naive = softmax_naive(attn_scores_2)
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

    print("\nattn_scores_2:")
    print(attn_scores_2)
    print("\nattn_weights_2_naive = softmax_naive(attn_scores_2):")
    print(attn_weights_2_naive)
    print("\nattn_weights_2 = torch.softmax(attn_scores_2, dim=0):")
    print(attn_weights_2)
    print("sum(attn_weights_2):", attn_weights_2.sum().item())
    print("naive and torch.softmax are close:", torch.allclose(attn_weights_2_naive, attn_weights_2))

    # context_vec_2:
    #   "journey" 위치의 새 context vector다.
    #
    # 책에서 나오는 아래 형태의 코드가 바로 이 단계다.
    #
    #   query = inputs[1]
    #   context_vec_2 = torch.zeros(query.shape)
    #   for i, x_i in enumerate(inputs):
    #       context_vec_2 += attn_weights_2[i] * x_i
    #
    # 이것은 attention score를 구하는 단계가 아니라, 이미 구한 attention weight로
    # 모든 입력 벡터를 섞어 z^(2)를 만드는 마지막 weighted sum 단계다.
    #
    # 이 간소화 예제에서는 inputs 자체를 value처럼 사용한다.
    # 그래서 각 x_i에 attention weight를 곱해 모두 더한다.
    context_vec_2 = torch.zeros(query.shape)
    print("\nWeighted sum: context_vec_2 += attn_weights_2[i] * x_i")
    print("This loop builds z^(2), the context vector for 'journey'.")
    for i, (word, x_i) in enumerate(zip(words, inputs)):
        contribution = attn_weights_2[i] * x_i
        context_vec_2 += contribution
        print(f"{word:8s}: weight={attn_weights_2[i].item():.4f}, contribution={contribution.tolist()}")

    print("\ncontext_vec_2:")
    print(context_vec_2)

    return attn_scores_2, attn_weights_2, context_vec_2


def compute_all_naive_context_vectors(inputs: torch.Tensor, words: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """간소화된 self-attention을 모든 token 위치에 대해 한 번에 계산한다.

    앞 함수는 "journey" 하나만 query로 골라서 z^(2)를 만들었다.
    이 함수는 모든 token을 각각 query로 삼아서 z^(1)부터 z^(T)까지 전부 만든다.

    이 단계가 질문에서 말한 간소화된 self-attention의 끝이다.

        attn_scores = X @ X.T
        attn_weights = softmax(attn_scores, dim=-1)
        all_context_vecs = attn_weights @ X

    실제 Transformer에서는 마지막 줄이 `attn_weights @ V`가 된다.
    여기서는 Q/K/V를 따로 만들기 전의 교육용 예제라 inputs를 그대로 사용한다.
    """
    print("\n" + "=" * 80)
    print("CASE 0-B. Naive self-attention for all token positions")
    print("=" * 80)

    num_tokens = inputs.shape[0]

    # attn_scores shape: (T, T)
    # - row i: i번째 token을 query로 봤을 때의 점수들
    # - column j: j번째 token을 key/value 후보로 봤을 때의 점수
    #
    # attn_scores[i, j] = dot(x_i, x_j)
    # 즉 i번째 token이 j번째 token과 얼마나 비슷한지 계산한다.
    attn_scores = torch.empty(num_tokens, num_tokens)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)

    print_matrix("attn_scores[i, j] = dot(x_i, x_j)", attn_scores)

    # dim=-1은 마지막 차원, 여기서는 각 row의 column 방향을 뜻한다.
    # 그래서 row마다 softmax가 따로 적용되고, 각 row의 합은 1이 된다.
    #
    # row i 전체는 "i번째 token이 모든 token을 얼마나 참고할지"의 비율이다.
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print_matrix("attn_weights = torch.softmax(attn_scores, dim=-1)", attn_weights)

    # 책 표기에서 두 번째 token x^(2)는 Python index로 1번 row다.
    # 이 row의 합이 1이라는 것은 x^(2)가 문장 전체를 참고하는 비율이
    # softmax로 정규화되었다는 뜻이다.
    row_2_sum = attn_weights[1].sum()
    print("\nrow_2_sum = attn_weights[1].sum()")
    print(row_2_sum)

    # all_context_vecs shape: (T, d_model)
    #
    # 행렬곱으로 보면:
    #   (T, T) @ (T, d_model) = (T, d_model)
    #
    # row i는 다음 가중합이다.
    #   z_i = sum_j attn_weights[i, j] * x_j
    #
    # 즉 for문으로 하나씩 만들던 context vector를 모든 token 위치에 대해
    # 한 번에 계산한다.
    all_context_vecs = attn_weights @ inputs
    print_matrix("all_context_vecs = attn_weights @ inputs", all_context_vecs)

    print("\nall_context_vecs[1] is z^(2), the context vector for 'journey':")
    print(all_context_vecs[1])

    return attn_scores, attn_weights, all_context_vecs


def compute_attention_for_journey(inputs: torch.Tensor, words: list[str], w_query: torch.Tensor, w_key: torch.Tensor, w_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """두 번째 token인 'journey' 위치의 attention 계산을 단계별로 수행한다."""
    # inputs shape: (token 개수, 입력 벡터 차원)
    # w_query/w_key/w_value shape: (입력 벡터 차원, 출력 벡터 차원)
    #
    # 실제 Transformer의 self-attention은 원본 입력 벡터를 그대로 비교하지 않고,
    # 학습 가능한 projection matrix를 곱해 Query, Key, Value를 따로 만든다.
    #
    # queries:
    #   각 token이 "나는 어떤 정보를 찾고 있는가?"를 표현한 벡터
    # keys:
    #   각 token이 "나는 어떤 정보와 잘 매칭되는가?"를 표현한 벡터
    # values:
    #   attention weight로 실제 섞이게 될 정보 벡터
    queries = inputs @ w_query
    keys = inputs @ w_key
    values = inputs @ w_value

    print_matrix("queries = inputs @ W_Q", queries)
    print_matrix("keys = inputs @ W_K", keys)
    print_matrix("values = inputs @ W_V", values)

    # 여기서는 두 번째 단어 "journey"가 문장 안의 다른 단어들을
    # 얼마나 참고하는지 보고 싶으므로, journey 위치의 query만 꺼낸다.
    query_2 = queries[1]
    print("\nquery_2 is the query vector for 'journey':")
    print(query_2)

    # raw_scores[i]는 "journey의 query"와 "i번째 token의 key"가
    # 얼마나 잘 맞는지를 나타내는 점수다.
    #
    # 점수 계산은 dot product다.
    # 두 벡터가 비슷한 방향이면 값이 커지고, 덜 비슷하면 작아진다.
    raw_scores = torch.empty(inputs.shape[0])
    print("\nDot products: score_i = key_i dot query_2")
    for i, (word, key_i) in enumerate(zip(words, keys)):
        raw_scores[i] = torch.dot(key_i, query_2)
        print(f"{word:8s}: dot({key_i.tolist()}, {query_2.tolist()}) = {raw_scores[i].item(): .4f}")

    # scaled dot-product attention:
    # key/query 차원이 커질수록 dot product 값도 커지는 경향이 있다.
    # 값이 너무 커지면 softmax가 한 token에 과하게 몰리므로 sqrt(d_k)로 나눠준다.
    #
    # keys.shape[1]은 key 벡터의 차원 d_k다.
    scaled_scores = raw_scores / math.sqrt(keys.shape[1])

    # softmax는 raw score들을 합이 1인 비율로 바꾼다.
    # 이 값이 attention weight다.
    # 예: [0.17, 0.16, ...]이면 각 value를 그 비율만큼 섞는다는 뜻이다.
    attention_weights = torch.softmax(scaled_scores, dim=0)

    # context_vector:
    # "journey" 위치의 새 벡터다.
    # 문장 안 모든 value 벡터를 attention weight만큼 가중합해서 만든다.
    #
    # attention_weights shape: (token 개수,)
    # values shape:            (token 개수, value 차원)
    # 결과 shape:              (value 차원,)
    context_vector = attention_weights @ values

    print("\nraw attention scores:")
    print(raw_scores)
    print("\nscaled scores = raw_scores / sqrt(d_k):")
    print(scaled_scores)
    print("\nattention weights = softmax(scaled_scores):")
    print(attention_weights)
    print("sum(attention_weights):", attention_weights.sum().item())

    print("\nContext vector for 'journey' = attention_weights @ values:")
    print(context_vector)

    print("\nContext vector calculation by value row:")
    for word, weight, value in zip(words, attention_weights, values):
        # contribution은 해당 단어의 value 벡터가 최종 context vector에
        # 실제로 더해지는 양이다.
        contribution = weight * value
        print(f"{word:8s}: weight={weight.item():.4f}, contribution={contribution.tolist()}")

    return raw_scores, attention_weights, context_vector


def main() -> None:
    # 랜덤 가중치를 쓰더라도 매번 같은 결과가 나오도록 seed를 고정한다.
    # seed를 바꾸면 W_Q/W_K/W_V가 달라지고 attention score도 달라진다.
    torch.manual_seed(123)

    # tensor 출력이 너무 길거나 과학적 표기법으로 나오지 않도록 설정한다.
    torch.set_printoptions(precision=4, sci_mode=False)

    # 각 row가 어떤 token에 해당하는지 출력할 때 쓰는 이름 목록이다.
    words = ["your", "journey", "starts", "with", "one", "step"]

    # 사용자가 준 예제 입력이다.
    # 이미 token embedding이 만들어진 상태라고 가정한다.
    #
    # shape = (6, 3)
    # - token 6개
    # - 각 token은 3차원 벡터
    #
    # 주의:
    # journey부터 step까지는 일부러 같은 벡터다.
    # 그래서 position embedding을 더하지 않으면 이 token들은 attention 계산에서도
    # 구분되지 않는다.
    token_embeddings = torch.tensor([
        [0.43, 0.15, 0.89],  # your
        [0.55, 0.87, 0.66],  # journey
        [0.55, 0.87, 0.66],  # starts
        [0.55, 0.87, 0.66],  # with
        [0.55, 0.87, 0.66],  # one
        [0.55, 0.87, 0.66],  # step
    ])

    print_matrix("token_embeddings", token_embeddings)

    # 먼저 가장 단순한 attention을 본다.
    # 이 단계에서는 Q/K/V projection 없이 입력 벡터끼리 직접 dot product를 하고,
    # 그 비율로 입력 벡터를 가중합한다.
    _, _, context_vec_2 = compute_naive_attention_for_journey(token_embeddings, words)

    # 같은 간소화 attention을 이번에는 모든 token 위치에 대해 한 번에 계산한다.
    # all_context_vecs[1]은 위에서 for문으로 계산한 context_vec_2와 같아야 한다.
    _, _, all_context_vecs = compute_all_naive_context_vectors(token_embeddings, words)
    print("\nall_context_vecs[1] equals context_vec_2:", torch.allclose(all_context_vecs[1], context_vec_2))

    # d_in:
    #   입력 token embedding의 차원이다. 여기서는 각 token이 3차원이다.
    d_in = token_embeddings.shape[1]

    # d_out:
    #   Query/Key/Value를 몇 차원으로 만들지 정한다.
    #   실제 모델에서는 보통 head_dim에 해당한다.
    #   여기서는 숫자를 보기 쉽게 2차원으로 줄였다.
    d_out = 2

    # W_Q, W_K, W_V:
    #   작은 self-attention layer의 학습 가능한 projection weight다.
    #
    # 실제 학습에서는 처음에 보통 랜덤으로 초기화되고,
    # loss를 줄이는 방향으로 gradient descent가 이 값들을 계속 수정한다.
    #
    # 여기서는 학습을 하지 않고 "랜덤 초기값이면 계산이 어떻게 흐르는지"만 본다.
    w_query = torch.randn(d_in, d_out)
    w_key = torch.randn(d_in, d_out)
    w_value = torch.randn(d_in, d_out)

    print_matrix("W_Q random projection weight", w_query)
    print_matrix("W_K random projection weight", w_key)
    print_matrix("W_V random projection weight", w_value)

    # CASE 1:
    # token embedding만 사용한다.
    # 같은 입력 벡터는 같은 Q/K/V로 변환되므로 attention score도 같아진다.
    print("\n" + "=" * 80)
    print("CASE 1. Use only the token embeddings you gave")
    print("=" * 80)
    print("Because rows 2-6 are identical, random W_Q/W_K/W_V still produce identical Q/K/V rows for those positions.")
    compute_attention_for_journey(token_embeddings, words, w_query, w_key, w_value)

    # CASE 2:
    # 실제 Transformer 입력에 더 가깝게 position embedding을 더한다.
    #
    # 같은 token embedding이라도 위치마다 다른 position embedding을 더하면
    # 최종 입력 벡터가 달라진다.
    #
    # 이 예제에서는 position embedding도 랜덤으로 만든다.
    # 실제 GPT에서는 position embedding 역시 학습 가능한 파라미터다.
    print("\n" + "=" * 80)
    print("CASE 2. Add random position embeddings before attention")
    print("=" * 80)
    print("A Transformer normally adds position information before attention. Then even identical token vectors can become different by position.")

    # torch.randn_like(token_embeddings)는 token_embeddings와 같은 shape의
    # 랜덤 tensor를 만든다.
    #
    # * 0.1을 하는 이유:
    #   position 값이 token embedding보다 너무 커지지 않도록 작게 만든다.
    position_embeddings = torch.randn_like(token_embeddings) * 0.1

    # Transformer에 들어가는 실제 입력은 보통
    # token embedding + position embedding이다.
    transformer_inputs = token_embeddings + position_embeddings

    print_matrix("position_embeddings", position_embeddings)
    print_matrix("transformer_inputs = token_embeddings + position_embeddings", transformer_inputs)

    compute_attention_for_journey(transformer_inputs, words, w_query, w_key, w_value)


if __name__ == "__main__":
    main()
