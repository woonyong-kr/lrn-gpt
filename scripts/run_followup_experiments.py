from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "python.exe"
RUNNER = ROOT / "scripts" / "run_lm_experiment.py"
OUT_DIR = ROOT / "docs" / "HY" / "testresult"


BASE = {
    "vocab_size": 3000,
    "context_length": 128,
    "emb_dim": 192,
    "n_heads": 4,
    "n_layers": 4,
    "ffn_mult": 4,
    "activation": "gelu",
    "drop_rate": 0.1,
    "qkv_bias": False,
    "weight_tying": False,
    "norm_first": False,
    "stride": 128,
    "num_epochs": 10,
    "eval_freq": 200,
}


EXPERIMENTS = [
    ("E31", "n_heads 1 10ep", {"n_heads": 1}, "n_heads=1, head_dim=192"),
    ("E32", "n_heads 2 10ep", {"n_heads": 2}, "n_heads=2, head_dim=96"),
    ("E33", "n_heads 8 10ep", {"n_heads": 8}, "n_heads=8, head_dim=24"),
    ("E34", "n_heads 12 10ep", {"n_heads": 12}, "n_heads=12, head_dim=16"),
    ("E35", "ffn multiplier 2 10ep", {"ffn_mult": 2}, "ffn_multiplier=2, num_epochs=10"),
    ("E36", "ffn multiplier 6 10ep", {"ffn_mult": 6}, "ffn_multiplier=6, num_epochs=10"),
    ("E37", "postLN layers8 10ep", {"n_layers": 8}, "post-LN, n_layers=8, num_epochs=10"),
    ("E38", "preLN layers8 10ep", {"n_layers": 8, "norm_first": True}, "pre-LN, n_layers=8, num_epochs=10"),
    ("E39", "qkv bias true 10ep", {"qkv_bias": True}, "qkv_bias=True, num_epochs=10"),
    ("E40", "qkv bias true layers8 10ep", {"qkv_bias": True, "n_layers": 8}, "qkv_bias=True, n_layers=8, num_epochs=10"),
    ("E41", "weight tying true 10ep", {"weight_tying": True}, "weight_tying=True, num_epochs=10"),
    ("E42", "weight tying true 20ep", {"weight_tying": True, "num_epochs": 20}, "weight_tying=True, num_epochs=20"),
]


def slug(name: str) -> str:
    return name.replace(" ", "_").replace("-", "").replace(".", "p")


def output_path(exp_id: str, name: str) -> Path:
    return OUT_DIR / f"{exp_id}_{slug(name)}_result.md"


def command(exp_id: str, name: str, opts: dict, change: str) -> list[str]:
    config = {**BASE, **opts}
    cmd = [
        str(PYTHON),
        str(RUNNER),
        "--experiment-id",
        exp_id,
        "--experiment-name",
        name,
        "--purpose",
        "기존 보고서에서 결론이 약했던 LLM 구조 하이퍼파라미터를 10 epoch 이상 기준으로 재검증한다.",
        "--change",
        change,
        "--output-md",
        str(output_path(exp_id, name)),
        "--vocab-size",
        str(config["vocab_size"]),
        "--context-length",
        str(config["context_length"]),
        "--emb-dim",
        str(config["emb_dim"]),
        "--n-heads",
        str(config["n_heads"]),
        "--n-layers",
        str(config["n_layers"]),
        "--ffn-mult",
        str(config["ffn_mult"]),
        "--activation",
        str(config["activation"]),
        "--drop-rate",
        str(config["drop_rate"]),
        "--batch-size",
        "32",
        "--num-epochs",
        str(config["num_epochs"]),
        "--lr",
        "0.0004",
        "--weight-decay",
        "0.1",
        "--eval-freq",
        str(config["eval_freq"]),
        "--eval-iter",
        "20",
        "--stride",
        str(config["stride"]),
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
    if config["qkv_bias"]:
        cmd.append("--qkv-bias")
    if config["weight_tying"]:
        cmd.append("--weight-tying")
    if config["norm_first"]:
        cmd.append("--norm-first")
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
