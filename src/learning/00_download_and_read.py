# -*- coding: utf-8 -*-
"""Step 00. Download the sample text and inspect the raw characters.

여기서는 아직 token ID도, embedding vector도 나오지 않는다.
그냥 "학습에 넣을 원문 텍스트가 이렇게 생겼구나"를 보는 단계다.
"""

from common import FILE_PATH, download_the_verdict, read_the_verdict


def main() -> None:
    # 내가 먼저 확인하고 싶은 것은 "모델이 볼 재료가 얼마나 되지?"다.
    # 20479자는 아주 짧은 샘플이라 GPT 원리를 실험하기 좋은 크기다.
    download_the_verdict()
    raw_text = read_the_verdict()

    print("파일 경로:", FILE_PATH)
    print("총 문자 개수:", len(raw_text))
    print(raw_text[:99])


if __name__ == "__main__":
    main()
