from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "python.exe"
RUNNER = ROOT / "scripts" / "run_lm_experiment.py"
OUT_DIR = ROOT / "docs" / "HY" / "testresult"


EXPERIMENTS = [
    ("E21", "long_epoch dropout 0.0", 0.0),
    ("E22", "long_epoch dropout 0.05", 0.05),
    ("E23", "long_epoch dropout 0.1", 0.1),
    ("E24", "long_epoch dropout 0.2", 0.2),
]


def output_path(exp_id: str, drop_rate: float) -> Path:
    rate = str(drop_rate).replace(".", "p")
    return OUT_DIR / f"{exp_id}_long_epoch_dropout_{rate}_result.md"


def command(exp_id: str, name: str, drop_rate: float) -> list[str]:
    return [
        str(PYTHON),
        str(RUNNER),
        "--experiment-id",
        exp_id,
        "--experiment-name",
        name,
        "--purpose",
        "긴 epoch에서 dropout이 train-val gap과 validation loss 안정성에 주는 영향을 확인한다.",
        "--change",
        f"num_epochs=20, drop_rate={drop_rate}",
        "--output-md",
        str(output_path(exp_id, drop_rate)),
        "--vocab-size",
        "3000",
        "--context-length",
        "128",
        "--emb-dim",
        "192",
        "--n-heads",
        "4",
        "--n-layers",
        "4",
        "--ffn-mult",
        "4",
        "--drop-rate",
        str(drop_rate),
        "--batch-size",
        "32",
        "--num-epochs",
        "20",
        "--lr",
        "0.0004",
        "--weight-decay",
        "0.1",
        "--eval-freq",
        "200",
        "--eval-iter",
        "20",
        "--seed",
        "123",
        "--prompt",
        "이 영화는",
        "--max-new-tokens",
        "120",
        "--temperature",
        "0.8",
        "--top-k",
        "20",
        "--amp",
        "bf16",
    ]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    for exp_id, name, drop_rate in EXPERIMENTS:
        out = output_path(exp_id, drop_rate)
        if out.exists():
            print(f"[SKIP] {exp_id} existing: {out}", flush=True)
            continue
        print(f"[RUN] {exp_id} {name}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(command(exp_id, name, drop_rate), cwd=ROOT)
        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            print(f"[FAIL] {exp_id} after {elapsed:.1f}s", file=sys.stderr, flush=True)
            return result.returncode
        print(f"[DONE] {exp_id} {elapsed:.1f}s -> {out}", flush=True)
    elapsed_all = time.perf_counter() - started_all
    print(f"[SUMMARY] elapsed_sec={elapsed_all:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
