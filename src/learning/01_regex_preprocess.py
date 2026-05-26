# -*- coding: utf-8 -*-
"""Step 01. Split raw text with a simple regular expression.

여기서 하는 일은 "문장을 일단 조각내 보자"다.
BPE처럼 pair 빈도를 세고 병합하는 단계는 아니고, toy vocab을 만들기 위한
가벼운 전처리라고 보면 된다.
"""

from common import read_the_verdict, regex_tokenize


def main() -> None:
    raw_text = read_the_verdict()

    # 여기서 생기는 preprocessed는 아직 token ID가 아니다.
    # 그냥 "나중에 vocab에 넣을 문자열 조각 후보"다.
    # 내가 헷갈리지 말아야 할 점:
    #   문자열 조각 -> 아직 숫자 아님
    #   token ID -> vocab을 만든 뒤 붙는 숫자
    preprocessed = regex_tokenize(raw_text)

    print("전처리 token 후보 개수:", len(preprocessed))
    print("앞 30개 token 후보 개수:", len(preprocessed[:30]))
    print(preprocessed[:30])


if __name__ == "__main__":
    main()
