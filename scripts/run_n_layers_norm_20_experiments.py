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
    ("E45", "postLN layers8 20ep", 8, False, "n_layers=8, post-LN, num_epochs=20"),
    ("E46", "preLN layers8 20ep", 8, True, "n_layers=8, pre-LN, num_epochs=20"),
    ("E47", "postLN layers12 20ep", 12, False, "n_layers=12, post-LN, num_epochs=20"),
    ("E48", "preLN layers12 20ep", 12, True, "n_layers=12, pre-LN, num_epochs=20"),
]


def slug(name: str) -> str:
    return name.replace(" ", "_").replace("-", "").replace(".", "p")


def output_path(exp_id: str, name: str) -> Path:
    return OUT_DIR / f"{exp_id}_{slug(name)}_result.md"


def command(exp_id: str, name: str, n_layers: int, norm_first: bool, change: str) -> list[str]:
    cmd = [
        str(PYTHON),
        str(RUNNER),
        "--experiment-id",
        exp_id,
        "--experiment-name",
        name,
        "--purpose",
        "n_layers 효과를 20 epoch 기준에서 norm 위치와 함께 비교해 깊이 증가가 실제 일반화에 주는 영향을 확인한다.",
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
        str(n_layers),
        "--ffn-mult",
        "4",
        "--activation",
        "gelu",
        "--drop-rate",
        "0.1",
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
    if norm_first:
        cmd.append("--norm-first")
    return cmd


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    for exp_id, name, n_layers, norm_first, change in EXPERIMENTS:
        out = output_path(exp_id, name)
        if out.exists():
            print(f"[SKIP] {exp_id} existing: {out}", flush=True)
            continue
        print(f"[RUN] {exp_id} {name}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(command(exp_id, name, n_layers, norm_first, change), cwd=ROOT)
        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            print(f"[FAIL] {exp_id} after {elapsed:.1f}s", file=sys.stderr, flush=True)
            return result.returncode
        print(f"[DONE] {exp_id} {elapsed:.1f}s -> {out}", flush=True)
    print(f"[SUMMARY] elapsed_sec={time.perf_counter() - started_all:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
