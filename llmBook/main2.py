import re
from pathlib import Path
from SimpleTokenizerV2 import Simpletokenizer

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = Simpletokenizer(raw_text)

enc_text = tokenizer.encode(raw_text)

print(len(enc_text))