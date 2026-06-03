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
    (
        "E43",
        "weight tying false 50ep",
        False,
        "num_epochs=50, weight_tying=False, dropout=0.1 baseline for long weight tying comparison",
    ),
    (
        "E44",
        "weight tying true 50ep",
        True,
        "num_epochs=50, weight_tying=True, dropout=0.1 long generalization comparison",
    ),
]


def slug(name: str) -> str:
    return name.replace(" ", "_").replace("-", "").replace(".", "p")


def output_path(exp_id: str, name: str) -> Path:
    return OUT_DIR / f"{exp_id}_{slug(name)}_result.md"


def command(exp_id: str, name: str, weight_tying: bool, change: str) -> list[str]:
    cmd = [
        str(PYTHON),
        str(RUNNER),
        "--experiment-id",
        exp_id,
        "--experiment-name",
        name,
        "--purpose",
        "weight_tying이 loss를 빠르게 낮추는 옵션인지, 긴 학습에서 일반화 정규화로 작동하는지 50 epoch 기준으로 확인한다.",
        "--change",
        change,
        "--output-md",
        str(output_path(exp_id, name)),
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
        "--activation",
        "gelu",
        "--drop-rate",
        "0.1",
        "--batch-size",
        "32",
        "--num-epochs",
        "50",
        "--lr",
        "0.0004",
        "--weight-decay",
        "0.1",
        "--eval-freq",
        "200",
        "--eval-iter",
        "20",
        "--stride",
        "128",
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
    if weight_tying:
        cmd.append("--weight-tying")
    return cmd


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    for exp_id, name, weight_tying, change in EXPERIMENTS:
        out = output_path(exp_id, name)
        if out.exists():
            print(f"[SKIP] {exp_id} existing: {out}", flush=True)
            continue
        print(f"[RUN] {exp_id} {name}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(command(exp_id, name, weight_tying, change), cwd=ROOT)
        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            print(f"[FAIL] {exp_id} after {elapsed:.1f}s", file=sys.stderr, flush=True)
            return result.returncode
        print(f"[DONE] {exp_id} {elapsed:.1f}s -> {out}", flush=True)
    print(f"[SUMMARY] elapsed_sec={time.perf_counter() - started_all:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
