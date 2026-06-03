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
    ("E58", "Final-A vocab2000", 2000),
    ("E59", "Final-A vocab3000", 3000),
]


def slug(name: str) -> str:
    return name.replace(" ", "_").replace("-", "").replace(".", "p")


def output_path(exp_id: str, name: str) -> Path:
    return OUT_DIR / f"{exp_id}_{slug(name)}_result.md"


def command(exp_id: str, name: str, vocab_size: int) -> list[str]:
    return [
        str(PYTHON),
        str(RUNNER),
        "--experiment-id",
        exp_id,
        "--experiment-name",
        name,
        "--purpose",
        "지금까지의 LLM 하이퍼파라미터 실험 결과를 조합한 최종 후보 A를 검증한다.",
        "--change",
        (
            f"final candidate A: vocab_size={vocab_size}, context_length=64, emb_dim=256, "
            "n_heads=2, n_layers=8, ffn_multiplier=6, activation=gelu, drop_rate=0.2, "
            "pre-LN, weight_tying=True, stride=64, num_epochs=50"
        ),
        "--output-md",
        str(output_path(exp_id, name)),
        "--vocab-size",
        str(vocab_size),
        "--context-length",
        "64",
        "--emb-dim",
        "256",
        "--n-heads",
        "2",
        "--n-layers",
        "8",
        "--ffn-mult",
        "6",
        "--activation",
        "gelu",
        "--drop-rate",
        "0.2",
        "--norm-first",
        "--weight-tying",
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
        "64",
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
    for exp_id, name, vocab_size in EXPERIMENTS:
        out = output_path(exp_id, name)
        if out.exists():
            print(f"[SKIP] {exp_id} existing: {out}", flush=True)
            continue
        print(f"[RUN] {exp_id} {name}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(command(exp_id, name, vocab_size), cwd=ROOT)
        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            print(f"[FAIL] {exp_id} after {elapsed:.1f}s", file=sys.stderr, flush=True)
            return result.returncode
        print(f"[DONE] {exp_id} {elapsed:.1f}s -> {out}", flush=True)
    print(f"[SUMMARY] elapsed_sec={time.perf_counter() - started_all:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
