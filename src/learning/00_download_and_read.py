# -*- coding: utf-8 -*-
"""Step 00. Download the sample text and inspect the raw characters.

여기서는 아직 token ID도, embedding vector도 나오지 않는다.
그냥 "학습에 넣을 원문 텍스트가 이렇게 생겼구나"를 보는 단계다.

책의 2장 시작점은 raw text다. LLM은 문자열을 바로 계산하지 못하므로,
앞으로 이 문자열을 token ID로 바꾸고, 다시 embedding vector로 바꾸게 된다.
"""

from common import FILE_PATH, download_the_verdict, read_the_verdict


def main() -> None:
    # 내가 먼저 확인하고 싶은 것은 "모델이 볼 재료가 얼마나 되지?"다.
    # 20479자는 아주 짧은 샘플이라 GPT 원리를 실험하기 좋은 크기다.
    download_the_verdict()
    raw_text = read_the_verdict()

    # 문자 수는 "데이터 규모"를 아주 거칠게 보는 값이다.
    # 모델 학습에서는 결국 문자 수가 아니라 token 수가 더 중요해진다.
    # 그래도 첫 단계에서는 원문이 어느 정도 길이인지 먼저 보는 게 좋다.
    print("파일 경로:", FILE_PATH)
    print("총 문자 개수:", len(raw_text))

    # 앞부분 99자를 출력해 원문이 정상적으로 다운로드되었는지 확인한다.
    # 아직 전처리하지 않았기 때문에 줄바꿈, 문장부호, 공백이 그대로 들어 있다.
    print(raw_text[:99])


if __name__ == "__main__":
    main()
