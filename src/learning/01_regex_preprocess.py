# -*- coding: utf-8 -*-
"""Step 01. Split raw text with a simple regular expression.

여기서 하는 일은 "문장을 일단 조각내 보자"다.
BPE처럼 pair 빈도를 세고 병합하는 단계는 아니고, toy vocab을 만들기 위한
가벼운 전처리라고 보면 된다.

책에서는 처음부터 완성형 BPE로 뛰어들지 않고, 단순 tokenizer를 만들어 보며
"텍스트 -> token 후보 -> token ID" 감각을 먼저 잡는다.
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

    # len(preprocessed)는 모델이 학습할 token 수가 아니다.
    # 이 값은 toy tokenizer 기준으로 나눈 문자열 조각 수다.
    # GPT-2 BPE tokenizer로 나누면 token 수가 달라진다.
    print("전처리 token 후보 개수:", len(preprocessed))

    # preprocessed[:30]은 앞에서부터 30개 token 후보를 잘라 본 것이다.
    # 이 단계에서 "I", "HAD", "--", "," 같은 조각들이 어떻게 분리되는지 확인한다.
    print("앞 30개 token 후보 개수:", len(preprocessed[:30]))
    print(preprocessed[:30])


if __name__ == "__main__":
    main()
