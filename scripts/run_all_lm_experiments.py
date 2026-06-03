from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "python.exe"
RUNNER = ROOT / "scripts" / "run_lm_experiment.py"


BASE = {
    "vocab_size": 3000,
    "context_length": 128,
    "emb_dim": 192,
    "n_heads": 4,
    "n_layers": 4,
    "ffn_mult": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "weight_tying": False,
    "batch_size": 32,
    "num_epochs": 5,
    "lr": 0.0004,
    "weight_decay": 0.1,
    "eval_freq": 500,
    "eval_iter": 20,
}


EXPERIMENTS = [
    ("E00", "기준 모델", {}, "baseline"),
    ("E01", "context_length 64", {"context_length": 64}, "context_length를 128에서 64로 감소"),
    ("E02", "context_length 192", {"context_length": 192}, "context_length를 128에서 192로 증가"),
    ("E03", "context_length 256", {"context_length": 256}, "context_length를 128에서 256으로 증가"),
    ("E04", "vocab_size 2000", {"vocab_size": 2000}, "vocab_size를 3000에서 2000으로 감소"),
    ("E05", "vocab_size 4000", {"vocab_size": 4000}, "vocab_size를 3000에서 4000으로 증가"),
    ("E06", "vocab_size 5000", {"vocab_size": 5000}, "vocab_size를 3000에서 5000으로 증가"),
    ("E07", "emb_dim 128", {"emb_dim": 128}, "emb_dim을 192에서 128로 감소"),
    ("E08", "emb_dim 256", {"emb_dim": 256}, "emb_dim을 192에서 256으로 증가"),
    ("E09", "n_heads 3", {"n_heads": 3}, "n_heads를 4에서 3으로 감소"),
    ("E10", "n_heads 6", {"n_heads": 6}, "n_heads를 4에서 6으로 증가"),
    ("E11", "n_layers 2", {"n_layers": 2}, "n_layers를 4에서 2로 감소"),
    ("E12", "n_layers 6", {"n_layers": 6}, "n_layers를 4에서 6으로 증가"),
    ("E13", "n_layers 8", {"n_layers": 8}, "n_layers를 4에서 8로 증가"),
    ("E14", "ffn_multiplier 2", {"ffn_mult": 2}, "ffn_multiplier를 4에서 2로 감소"),
    ("E15", "ffn_multiplier 6", {"ffn_mult": 6}, "ffn_multiplier를 4에서 6으로 증가"),
    ("E16", "drop_rate 0.0", {"drop_rate": 0.0}, "drop_rate를 0.1에서 0.0으로 감소"),
    ("E17", "drop_rate 0.05", {"drop_rate": 0.05}, "drop_rate를 0.1에서 0.05로 감소"),
    ("E18", "drop_rate 0.2", {"drop_rate": 0.2}, "drop_rate를 0.1에서 0.2로 증가"),
    ("E19", "qkv_bias True", {"qkv_bias": True}, "qkv_bias를 False에서 True로 변경"),
    ("E20", "weight_tying True", {"weight_tying": True}, "token embedding과 LM head weight를 공유"),
]


def value(config: dict, key: str):
    return config.get(key, BASE[key])


def output_path(exp_id: str, name: str) -> Path:
    slug = name.replace(" ", "_").replace(".", "p")
    return ROOT / "testresult" / f"{exp_id}_{slug}_result.md"


def build_command(exp_id: str, name: str, overrides: dict, change: str) -> list[str]:
    config = {**BASE, **overrides}
    cmd = [
        str(PYTHON),
        str(RUNNER),
        "--experiment-id",
        exp_id,
        "--experiment-name",
        name,
        "--purpose",
        "Notion 실험 DB에 정의된 LLM 하이퍼파라미터 변경 효과를 baseline과 비교한다.",
        "--change",
        change,
        "--output-md",
        str(output_path(exp_id, name)),
        "--vocab-size",
        str(value(config, "vocab_size")),
        "--context-length",
        str(value(config, "context_length")),
        "--emb-dim",
        str(value(config, "emb_dim")),
        "--n-heads",
        str(value(config, "n_heads")),
        "--n-layers",
        str(value(config, "n_layers")),
        "--ffn-mult",
        str(value(config, "ffn_mult")),
        "--drop-rate",
        str(value(config, "drop_rate")),
        "--batch-size",
        str(value(config, "batch_size")),
        "--num-epochs",
        str(value(config, "num_epochs")),
        "--lr",
        str(value(config, "lr")),
        "--weight-decay",
        str(value(config, "weight_decay")),
        "--eval-freq",
        str(value(config, "eval_freq")),
        "--eval-iter",
        str(value(config, "eval_iter")),
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
    return cmd


def main() -> int:
    if not PYTHON.exists():
        print(f"Python not found: {PYTHON}", file=sys.stderr)
        return 1

    (ROOT / "testresult").mkdir(exist_ok=True)
    started_all = time.perf_counter()
    completed = []
    skipped = []

    for exp_id, name, overrides, change in EXPERIMENTS:
        out = output_path(exp_id, name)
        legacy_e00 = ROOT / "testresult" / "E00_baseline_result.md"
        if exp_id == "E00" and legacy_e00.exists():
            skipped.append(str(legacy_e00))
            print(f"[SKIP] {exp_id} existing: {legacy_e00}", flush=True)
            continue
        if out.exists():
            skipped.append(str(out))
            print(f"[SKIP] {exp_id} existing: {out}", flush=True)
            continue

        cmd = build_command(exp_id, name, overrides, change)
        print(f"[RUN] {exp_id} {name}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(cmd, cwd=ROOT)
        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            print(f"[FAIL] {exp_id} after {elapsed:.1f}s", file=sys.stderr, flush=True)
            return result.returncode
        completed.append(str(out))
        print(f"[DONE] {exp_id} {elapsed:.1f}s -> {out}", flush=True)

    elapsed_all = time.perf_counter() - started_all
    print(f"[SUMMARY] completed={len(completed)} skipped={len(skipped)} elapsed_sec={elapsed_all:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
