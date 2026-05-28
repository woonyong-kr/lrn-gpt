import re
from pathlib import Path
from SimpleTokenizerV2 import Simpletokenizer

raw_text = Path("the-verdict.txt").read_text(encoding="utf-8")

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_word = sorted(set(preprocessed))
vocab_size = len(all_word)

vocab = {token: integer for integer, token in enumerate(all_word)}

# text = """"It's the last he painted, you know,"
#         Mrs. Gisburn said with pardonable pride."""

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}

# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)
# tokenizer.decode(tokenizer.encode(text)

tokenizer = Simpletokenizer(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the place."
text = " <|endoftext|> ".join ((text1, text2))
print(tokenizer.decode(tokenizer.encode(text)))