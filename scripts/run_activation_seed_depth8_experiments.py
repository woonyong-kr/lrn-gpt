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
    ("E49", "activation GELU depth8 preLN seed42 20ep", "gelu", 42),
    ("E50", "activation ReLU depth8 preLN seed42 20ep", "relu", 42),
    ("E51", "activation SiLU depth8 preLN seed42 20ep", "silu", 42),
    ("E52", "activation GELU depth8 preLN seed123 20ep", "gelu", 123),
    ("E53", "activation ReLU depth8 preLN seed123 20ep", "relu", 123),
    ("E54", "activation SiLU depth8 preLN seed123 20ep", "silu", 123),
    ("E55", "activation GELU depth8 preLN seed2026 20ep", "gelu", 2026),
    ("E56", "activation ReLU depth8 preLN seed2026 20ep", "relu", 2026),
    ("E57", "activation SiLU depth8 preLN seed2026 20ep", "silu", 2026),
]


def slug(name: str) -> str:
    return name.replace(" ", "_").replace("-", "").replace(".", "p")


def output_path(exp_id: str, name: str) -> Path:
    return OUT_DIR / f"{exp_id}_{slug(name)}_result.md"


def command(exp_id: str, name: str, activation: str, seed: int) -> list[str]:
    return [
        str(PYTHON),
        str(RUNNER),
        "--experiment-id",
        exp_id,
        "--experiment-name",
        name,
        "--purpose",
        "GELU가 Transformer FFN에서 적합한지 확인하기 위해 8-layer pre-LN 20 epoch 조건에서 activation과 seed를 반복 비교한다.",
        "--change",
        f"n_layers=8, norm_first=True, activation={activation}, seed={seed}, num_epochs=20",
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
        "8",
        "--ffn-mult",
        "4",
        "--activation",
        activation,
        "--drop-rate",
        "0.1",
        "--norm-first",
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
        "--stride",
        "128",
        "--seed",
        str(seed),
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
    for exp_id, name, activation, seed in EXPERIMENTS:
        out = output_path(exp_id, name)
        if out.exists():
            print(f"[SKIP] {exp_id} existing: {out}", flush=True)
            continue
        print(f"[RUN] {exp_id} activation={activation} seed={seed}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(command(exp_id, name, activation, seed), cwd=ROOT)
        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            print(f"[FAIL] {exp_id} after {elapsed:.1f}s", file=sys.stderr, flush=True)
            return result.returncode
        print(f"[DONE] {exp_id} {elapsed:.1f}s -> {out}", flush=True)
    print(f"[SUMMARY] elapsed_sec={time.perf_counter() - started_all:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
