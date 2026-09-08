import pytest
from src.bpe import BPETokenizer


@pytest.mark.parametrize(
    "text,pair,ids",
    [
        ("aaaaa", (101, 101), [260, 260, 101]),
        ("abac", (101, 102), [260, 101, 103]),
    ],
)
def test_merge_nonoverlap_and_first_occurrence_tie(text, pair, ids):
    tokenizer = BPETokenizer(vocab_size=261).train(text)
    assert tokenizer.merges == [pair]
    assert tokenizer.encode(text) == ids
    assert tokenizer.decode(ids) == text


def test_utf8_unseen_bytes_and_serialization_roundtrip(tmp_path):
    tokenizer = BPETokenizer(vocab_size=300).train("한글과 UTF-8 🙂\n" * 10)
    text = "처음 보는 漢字 😀\n\t<bos>"
    ids = tokenizer.encode(text, add_bos_eos=True)
    tokenizer.save(tmp_path / "tokenizer.json")
    restored = BPETokenizer().load(tmp_path / "tokenizer.json")
    assert restored.encode(text, add_bos_eos=True) == ids
    assert ids[0] == 2 and ids[-1] == 3
    assert restored.decode(ids) == text
    assert restored.decode(restored.encode("")) == ""
    assert BPETokenizer(270, min_frequency=3).train("abab").merges == []
