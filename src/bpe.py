# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path


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
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = [] #merge rule 저장

    def _init_special_tokens(self):
        """
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """

        for token, idx in SPECIAL_IDS.items():
            self.id_to_token[idx] = token
            self.token_to_id[token] = idx

        for byte_value in range(NUM_BYTES):
            token_id = BYTE_OFFSET + byte_value
            '''bytes([정수])
            정수를 bytes객체로 바꾸는 코드

            bytes([97])
            # b'a'

            bytes([0])
            # b'\x00'

            bytes([236])
            # b'\xec'
            '''
            byte_token = bytes([byte_value])
            self.id_to_token[token_id] = byte_token
            self.token_to_id[byte_token] = token_id

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
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        raise NotImplementedError("BPETokenizer.train을 구현하세요.")

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        raise NotImplementedError("BPETokenizer.save를 구현하세요.")

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        raise NotImplementedError("BPETokenizer.load를 구현하세요.")

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
        for byte in byte_ids:
            token_id = BYTE_OFFSET + byte
            token_ids.append(token_id)

        #2. train/load에서 얻은 merge rule을 학습 순서대로 적용
        for rule in self.merges:

            new_token_ids = []

            index = 0
            while(index < len(token_ids)):
                set = (token_ids[index], token_ids[index+1])
                if(set == rule):
                    merged_token_id = self.token_to_id[rule]
                    new_token_ids.append(merged_token_id)
                    index += 2
                else:
                    new_token_ids.append(token_ids[index])
                    index+=1
            
            #룰 적용한 token_id로 업데이트
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
        ''' b''.join(byte_token)의 역할
        전: <class 'list'>
        byte_token: [b'\xec', b'\x95', b'\x88']

        후:<class 'bytes'>
        byte_data:  b'\xec\x95\x88'
        '''
        byte_joined = b''.join(byte_token)
        text = bytes(byte_joined).decode("utf-8")

        return text
