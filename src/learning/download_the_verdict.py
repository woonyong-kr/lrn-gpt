# -*- coding: utf-8 -*-
"""Run all step-by-step learning examples.

The old file used to contain every experiment in one place. It now works as a
small launcher so the individual learning steps stay readable.

개별 파일을 하나씩 봐도 되고, 흐름을 다시 보고 싶으면 이 파일을 실행하면 된다.
"""

from pathlib import Path
import subprocess
import sys


STEPS = [
    "00_download_and_read.py",
    "01_regex_preprocess.py",
    "02_vocab_and_simple_tokenizer.py",
    "03_tiktoken_bpe.py",
    "04_dataset_and_dataloader.py",
    "05_token_embedding.py",
    "06_token_and_position_embedding.py",
]


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    for step in STEPS:
        script = base_dir / step
        print(f"\n===== {step} =====", flush=True)
        subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
