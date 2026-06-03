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
    ("E28", "post-LN baseline 10ep", {"activation": "gelu"}, "num_epochs=10, GELU, post-LN, stride=128 baseline"),
    ("E25", "activation ReLU", {"activation": "relu"}, "num_epochs=10, activation=ReLU"),
    ("E26", "activation SiLU", {"activation": "silu"}, "num_epochs=10, activation=SiLU"),
    ("E27", "pre-LN", {"activation": "gelu", "norm_first": True}, "num_epochs=10, norm_first=True"),
    ("E29", "stride 64", {"activation": "gelu", "stride": 64}, "num_epochs=10, stride=64 overlapping window"),
]


def slug(name: str) -> str:
    return name.replace(" ", "_").replace("-", "").replace(".", "p")


def output_path(exp_id: str, name: str) -> Path:
    return OUT_DIR / f"{exp_id}_{slug(name)}_result.md"


def command(exp_id: str, name: str, opts: dict, change: str) -> list[str]:
    cmd = [
        str(PYTHON),
        str(RUNNER),
        "--experiment-id",
        exp_id,
        "--experiment-name",
        name,
        "--purpose",
        "epoch 10 기준에서 아직 비교하지 않은 Transformer 구조 요소의 영향을 확인한다.",
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
        opts.get("activation", "gelu"),
        "--drop-rate",
        "0.1",
        "--batch-size",
        "32",
        "--num-epochs",
        "10",
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
    if opts.get("norm_first"):
        cmd.append("--norm-first")
    if "stride" in opts:
        cmd.extend(["--stride", str(opts["stride"])])
    return cmd


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    for exp_id, name, opts, change in EXPERIMENTS:
        out = output_path(exp_id, name)
        if out.exists():
            print(f"[SKIP] {exp_id} existing: {out}", flush=True)
            continue
        print(f"[RUN] {exp_id} {name}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(command(exp_id, name, opts, change), cwd=ROOT)
        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            print(f"[FAIL] {exp_id} after {elapsed:.1f}s", file=sys.stderr, flush=True)
            return result.returncode
        print(f"[DONE] {exp_id} {elapsed:.1f}s -> {out}", flush=True)
    print(f"[SUMMARY] elapsed_sec={time.perf_counter() - started_all:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
