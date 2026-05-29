# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        # 숫자 입력 시 토큰 반환
        self.id_to_token = {}
        # 토큰 입력 시 숫자 반환
        self.token_to_id = {}
        self.merges = [] #merge rule 저장

        self._init_special_tokens()

    def _init_special_tokens(self):
        """
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """

        # 여기에서 변환코드 먼저
        # utf-8 명명 규칙 -> b\xed\x95\x9c 같은 형태로 모임 
        # -> 바이트로 해석 + '0x ed, 0x 95, 0x 9c' 같은 형태로 해석
        # -> 0xED = 237 / 0x95 = 149 / 0x9C = 156 == '한' 이라는 글자가 됨
        
        for i in range(0, len(SPECIAL_TOKENS)):
            self.id_to_token[i] = SPECIAL_TOKENS[i]
            self.token_to_id[SPECIAL_TOKENS[i]] = i

        for i in range(4, 260):
            self.id_to_token[i] = bytes([i - 4])
            self.token_to_id[bytes([i - 4])] = i

#region GETTER
    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]
#endregion
    
    def train(self, corpus: str):
        """
        코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        
        # word = [w.encode('utf-8') for w in text]
        # print(word) == \x00 \x01 ~
        # 옆 자리 encode 값과 서로 묶어서 pair로 만듦

        # id_to_token 할 때 조정할 +4값
        words = [w + 4 for w in corpus.encode('utf-8')]

        while (len(self.id_to_token) < self.vocab_size):
            word_pair = {}
            new_words = []

            # 현재 워드 별 얼마나 자주 붙어있는지 판단
            for i in range(len(words) - 1):
                # 여기에서 튜플형태로 만듦
                pair = (words[i], words[i+1])

                if pair not in word_pair:
                    word_pair[pair] = 0
                
                word_pair[pair] += 1

            # pair = b'\x02\x07' 같이 튜플로 만들어져있음. 포문 첫 줄
            # 그 페어들의 목록 중 가장 많이 나온 페어 확인
            best_pair = max(word_pair, key=lambda x: word_pair[x])

            # 베스트 페어를 보관할 위치 찾기
            new_id = len(self.id_to_token)
            # 베스트 페어 추가
            self.id_to_token[new_id] = self.id_to_token[best_pair[0]] + self.id_to_token[best_pair[1]]
            self.token_to_id[self.id_to_token[best_pair[0]] + self.id_to_token[best_pair[1]]] = new_id

            # 현재 단어 페어로 추가
            self.merges.append(best_pair)

            # 모든 페어 묶기
            i = 0
            while i < len(words) - 1:
                if (words[i], words[i+1]) == best_pair:
                    new_words.append(new_id)
                    i += 2
                else:
                    new_words.append(words[i])
                    i += 1

            if i < len(words):
                new_words.append(words[i])

            words = new_words

    def save(self, path: str | Path):
        """
        vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        # 이니셜라이즈 시 0~259까지 복원됨 == 저장할 데이터에 260 미만의 데이터는 필요가 없음
        # id_to_token으로 얻은 추가 데이터들은 b'\x02\x07'같은 연속된 형태로 보관돼있음.
        add_token = {int(k) :list(v) for k, v in self.id_to_token.items() if k >= 260}
        merges = self.merges
        
        total_data = {
            "vocab" : add_token,
            "merges" : merges 
        }

        with open(path, "w", encoding = "utf-8") as f:
            json.dump(total_data, f)

    def load(self, path: str | Path):
        """
        save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """

        # data 호출
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # k, v => vocab의 key, value
        for k, v in data["vocab"].items():
            key = int(k)
            value = bytes(v)
            self.id_to_token[key] = value
            self.token_to_id[value] = key

        for i in range(len(data["merges"])):
            self.merges.append(tuple(data["merges"][i]))

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """

        #1. UTF-8 byte ID 리스트를 만듭니다.
        byte_ids  = text.encode("utf-8") #문자열을 UTF-8 bytes 객체로 바꿈
        token_ids = []
        # +4 맞추기 용도인듯
        for byte in byte_ids:
            token_id = BYTE_OFFSET + byte
            token_ids.append(token_id)

        #2. train/load에서 얻은 merge rule을 학습 순서대로 적용
        for rule in self.merges:
            new_token_ids = []
            index = 0
            while index < len(token_ids) - 1:
                set = (token_ids[index], token_ids[index+1])
                if set == rule:
                    merged_bytes = self.id_to_token[rule[0]] + self.id_to_token[rule[1]]
                    merged_token_id = self.token_to_id[merged_bytes]
                    new_token_ids.append(merged_token_id)
                    index += 2
                else:
                    new_token_ids.append(token_ids[index])
                    index += 1

            if index < len(token_ids):
                new_token_ids.append(token_ids[index])

            token_ids = new_token_ids


        #3. add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙이기  
        if(add_bos_eos==True):
            token_ids = [self.get_bos_id()] + token_ids + [self.get_eos_id()]
        
        return token_ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """

        #1. ids 배열 순회하며 SPECIAL_TOKENS 삭제
        new_ids=[]
        if(skip_special==True):
            for id in ids:
                if (id == self.get_bos_id() or 
                    id == self.get_eos_id() or
                    id == self.get_pad_id() or
                    id == self.get_unk_id()):
                    continue
                else:
                    new_ids.append(id)
        else:
            new_ids=ids.copy()

        #2. id리스트 => 바이트 토큰
        byte_token = []
        for id in new_ids:
            token = self.id_to_token[id]
            byte_token.append(token)

        #3. 바이트 토큰 => 문자열
        byte_joined = b''.join(byte_token)
        text = bytes(byte_joined).decode("utf-8")

        return text
