# -*- coding: utf-8 -*-
"""Self-managing experiment loop for the mini GPT project.

The script is intentionally small-loop oriented: one invocation checks state,
runs at most one experiment, writes reports, and exits. A Codex heartbeat can
call it every few minutes without starting overlapping training jobs.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

try:
    from .experiments import LMExperimentConfig, estimate_steps_per_epoch, load_corpus, resolve_device, run_experiment_plan, write_plan
except ImportError:
    from experiments import LMExperimentConfig, estimate_steps_per_epoch, load_corpus, resolve_device, run_experiment_plan, write_plan


ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "docs" / "train"
RUNS_DIR = TRAIN_DIR / "runs"
STATE_PATH = TRAIN_DIR / "state.json"
LOCK_PATH = TRAIN_DIR / ".train_loop.lock"
README_PATH = TRAIN_DIR / "README.md"
HYPOTHESES_PATH = TRAIN_DIR / "hypotheses.md"
LEADERBOARD_PATH = TRAIN_DIR / "leaderboard.csv"
NEXT_PLAN_PATH = TRAIN_DIR / "next_plan.json"
NEXT_PLAN_SCHEMA_PATH = TRAIN_DIR / "next_plan.schema.json"
METRICS_SUMMARY_PATH = TRAIN_DIR / "metrics_summary.csv"
DASHBOARD_PATH = TRAIN_DIR / "dashboard.md"
VISUALS_DIR = TRAIN_DIR / "visuals"
TREND_SVG_PATH = VISUALS_DIR / "loss_overfit_trends.svg"
LATEST_RUN_SVG_PATH = VISUALS_DIR / "latest_run_metrics.svg"
DEFAULT_CORPUS_PATH = ROOT / "src" / "learning" / "the-verdict.txt"
FALLBACK_CORPUS_PATH = ROOT / "README.md"

LEADERBOARD_FIELDS = [
    "run_id",
    "timestamp",
    "hypothesis",
    "changed_variables",
    "seed",
    "vocab_size",
    "context_length",
    "stride",
    "batch_size",
    "epochs",
    "steps_per_epoch",
    "max_steps",
    "learning_rate",
    "weight_decay",
    "grad_clip",
    "emb_dim",
    "n_heads",
    "n_layers",
    "drop_rate",
    "qkv_bias",
    "ffn_mult",
    "norm_first",
    "norm_eps",
    "activation_name",
    "ffn_dropout_position",
    "attention_impl",
    "tie_embeddings",
    "init_std",
    "final_train_loss",
    "final_val_loss",
    "final_generalization_gap",
    "generalization_gap_delta",
    "train_val_improvement_gap",
    "overfit_score",
    "fit_status",
    "parameter_count",
    "tokens_per_sec",
    "elapsed_sec",
    "device",
    "artifact_dir",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one self-managed mini GPT experiment loop iteration.")
    parser.add_argument("--dry-run", action="store_true", help="Initialize docs and print the next plan without training.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--epochs", type=float, default=None, help="Override the generated run's epoch count.")
    parser.add_argument("--max-steps", type=int, default=None, help="Legacy override for compatibility; prefer --epochs for new research runs.")
    parser.add_argument("--corpus-path", type=Path, default=None)
    parser.add_argument("--char-limit", type=int, default=20_000)
    parser.add_argument("--force", action="store_true", help="Ignore a stale-looking lock and continue.")
    parser.add_argument("--plan-file", type=Path, default=NEXT_PLAN_PATH, help="Optional LLM-authored JSON plan to use before rule-based fallback.")
    args = parser.parse_args(argv)
    if args.epochs is not None and args.max_steps is not None:
        parser.error("--epochs and --max-steps are mutually exclusive")

    ensure_train_scaffold()
    hardware = inspect_hardware(args.device)
    state = load_state()
    update_readme(hardware)

    active_lock = read_active_lock()
    if active_lock and not args.force:
        state.update(
            {
                "status": "running",
                "running_pid": active_lock.get("pid"),
                "lock_info": active_lock,
                "last_updated": utc_now(),
                "last_note": "Existing training lock is active; skipped this heartbeat.",
            }
        )
        save_state(state)
        print(f"training already running: pid={active_lock.get('pid')} run={active_lock.get('run_id')}")
        return

    if active_lock and args.force:
        LOCK_PATH.unlink(missing_ok=True)

    leaderboard_rows = backfill_leaderboard_epoch_fields(read_leaderboard())
    if leaderboard_rows:
        write_leaderboard(leaderboard_rows)
    if leaderboard_rows:
        refresh_visual_metrics_from_leaderboard(leaderboard_rows)
    next_run_id = next_run_number(state, leaderboard_rows)
    plan = choose_next_experiment(next_run_id, leaderboard_rows, hardware)
    plan = apply_llm_plan_override(plan, args.plan_file, next_run_id)
    if args.epochs is not None:
        plan.config = replace(plan.config, epochs=args.epochs)
        plan.changed_variables = {**plan.changed_variables, "epochs": args.epochs}
    if args.max_steps is not None:
        plan.config = replace(plan.config, epochs=None, max_steps=args.max_steps)
        plan.changed_variables = {**plan.changed_variables, "max_steps": args.max_steps}

    if args.dry_run:
        state.update(
            {
                "status": "idle",
                "next_run": next_run_id,
                "last_updated": utc_now(),
                "hardware": hardware,
                "dry_run_next_hypothesis": plan.hypothesis,
                "dry_run_next_config": asdict(plan.config),
                "dry_run_plan_source": getattr(plan, "source", "rule_based"),
            }
        )
        save_state(state)
        print(json.dumps({"next_run": next_run_id, "hypothesis": plan.hypothesis, "plan_source": getattr(plan, "source", "rule_based"), "config": asdict(plan.config), "hardware": hardware}, ensure_ascii=False, indent=2))
        return

    artifact_dir = RUNS_DIR / f"run_{next_run_id:03d}_artifacts"
    report_path = RUNS_DIR / f"run_{next_run_id:03d}.md"
    create_lock(next_run_id, artifact_dir)

    state.update(
        {
            "status": "running",
            "current_run": next_run_id,
            "running_pid": os.getpid(),
            "last_experiment_dir": str(artifact_dir.relative_to(ROOT)),
            "last_updated": utc_now(),
            "hardware": hardware,
            "current_hypothesis": plan.hypothesis,
            "current_plan_source": getattr(plan, "source", "rule_based"),
        }
    )
    save_state(state)

    try:
        corpus_path = args.corpus_path or default_corpus_path()
        corpus = load_corpus(corpus_path, char_limit=args.char_limit)
        device = resolve_device(args.device)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        archive_plan_file(args.plan_file, artifact_dir)
        write_plan([plan.config], artifact_dir / "plan.csv")
        (artifact_dir / "plan.json").write_text(json.dumps([asdict(plan.config)], ensure_ascii=False, indent=2), encoding="utf-8")

        results = run_experiment_plan([plan.config], corpus=corpus, output_dir=artifact_dir, device=device)
        result = results[0]
        row = leaderboard_row(plan, result, artifact_dir)
        leaderboard_rows.append(row)
        write_leaderboard(leaderboard_rows)
        visual_paths = update_visual_metrics(leaderboard_rows, result, artifact_dir)
        best_row = choose_best_row(leaderboard_rows)
        write_run_report(report_path, plan, result, hardware, corpus_path, artifact_dir, best_row, visual_paths)
        append_hypothesis_log(plan, result, report_path)

        state.update(
            {
                "status": "completed",
                "current_run": next_run_id,
                "next_run": next_run_id + 1,
                "running_pid": None,
                "lock_info": None,
                "last_result": result_summary(result),
                "last_report": str(report_path.relative_to(ROOT)),
                "best_run": best_row,
                "dashboard_path": str(DASHBOARD_PATH.relative_to(ROOT)),
                "latest_visualization": str(visual_paths["latest_run"].relative_to(ROOT)),
                "last_updated": utc_now(),
            }
        )
        save_state(state)
        print(f"completed run {next_run_id:03d}; report={report_path.relative_to(ROOT)}")
    except Exception as exc:
        failure = {"error": repr(exc), "timestamp": utc_now(), "run_id": next_run_id}
        write_failure_report(report_path, plan, hardware, failure)
        state.update(
            {
                "status": "failed",
                "current_run": next_run_id,
                "running_pid": None,
                "lock_info": None,
                "last_error": failure,
                "last_report": str(report_path.relative_to(ROOT)),
                "last_updated": utc_now(),
            }
        )
        save_state(state)
        raise
    finally:
        LOCK_PATH.unlink(missing_ok=True)


class ExperimentPlan:
    def __init__(
        self,
        config: LMExperimentConfig,
        hypothesis: str,
        rationale: str,
        changed_variables: dict[str, Any],
        fixed_variables: list[str],
        expected_result: str,
        next_if_success: str,
        next_if_overfit: str,
    ):
        self.config = config
        self.hypothesis = hypothesis
        self.rationale = rationale
        self.changed_variables = changed_variables
        self.fixed_variables = fixed_variables
        self.expected_result = expected_result
        self.next_if_success = next_if_success
        self.next_if_overfit = next_if_overfit
        self.source = "rule_based"


def apply_llm_plan_override(plan: ExperimentPlan, plan_path: Path, run_id: int) -> ExperimentPlan:
    """Use an LLM-authored next_plan.json when present, with rule-based plan as fallback."""
    if not plan_path.exists():
        return plan

    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ignored invalid LLM plan JSON at {plan_path}: {exc}")
        return plan

    hypothesis = str(payload.get("hypothesis") or plan.hypothesis)
    config_values = asdict(plan.config)
    allowed_config_keys = {field.name for field in fields(LMExperimentConfig)}
    overrides = payload.get("config_overrides") or payload.get("config") or {}
    if not isinstance(overrides, dict):
        print(f"ignored LLM plan overrides because config_overrides is not an object: {plan_path}")
        overrides = {}

    changed_variables = payload.get("changed_variables")
    if not isinstance(changed_variables, dict):
        changed_variables = dict(overrides)

    ignored_keys: list[str] = []
    for key, value in overrides.items():
        if key in {"run_id", "hypothesis"}:
            ignored_keys.append(key)
            continue
        if key not in allowed_config_keys:
            ignored_keys.append(key)
            continue
        config_values[key] = value

    config_values["run_id"] = run_id
    config_values["hypothesis"] = hypothesis

    try:
        config = ensure_head_divisible(LMExperimentConfig(**config_values))
    except TypeError as exc:
        print(f"ignored LLM plan because config could not be built: {exc}")
        return plan

    overridden = ExperimentPlan(
        config=config,
        hypothesis=hypothesis,
        rationale=str(payload.get("rationale") or plan.rationale),
        changed_variables=changed_variables,
        fixed_variables=list(payload.get("fixed_variables") or plan.fixed_variables),
        expected_result=str(payload.get("expected_result") or plan.expected_result),
        next_if_success=str(payload.get("next_if_success") or plan.next_if_success),
        next_if_overfit=str(payload.get("next_if_overfit") or plan.next_if_overfit),
    )
    overridden.source = f"llm_plan:{plan_path}"
    if ignored_keys:
        overridden.changed_variables = {**overridden.changed_variables, "_ignored_keys": ignored_keys}
    return overridden


def archive_plan_file(plan_path: Path, artifact_dir: Path) -> None:
    """Archive and consume an LLM plan so the same hypothesis is not rerun forever."""
    if not plan_path.exists():
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(plan_path, artifact_dir / "llm_next_plan.json")
    plan_path.unlink(missing_ok=True)


def ensure_train_scaffold() -> None:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        save_state(default_state())
    if not HYPOTHESES_PATH.exists():
        HYPOTHESES_PATH.write_text("# 자동 실험 가설 기록\n\n아직 완료된 실험이 없습니다.\n", encoding="utf-8")
    if not LEADERBOARD_PATH.exists():
        write_leaderboard([])
    schema_text = json.dumps(next_plan_schema(), ensure_ascii=False, indent=2)
    if not NEXT_PLAN_SCHEMA_PATH.exists() or NEXT_PLAN_SCHEMA_PATH.read_text(encoding="utf-8") != schema_text:
        NEXT_PLAN_SCHEMA_PATH.write_text(schema_text, encoding="utf-8")


def next_plan_schema() -> dict[str, Any]:
    return {
        "description": "LLM-authored next experiment plan consumed by python -m src.train_loop_agent.",
        "required": ["hypothesis", "rationale", "expected_result", "config_overrides"],
        "properties": {
            "hypothesis": "Research hypothesis for the next run.",
            "rationale": "Why this hypothesis follows from the leaderboard and latest report.",
            "changed_variables": "Object containing the variables intentionally changed.",
            "fixed_variables": "List of variables intentionally held fixed.",
            "expected_result": "Expected train/validation/overfit behavior.",
            "next_if_success": "Follow-up if the result generalizes.",
            "next_if_overfit": "Follow-up if overfitting appears.",
            "config_overrides": {
                "allowed_keys": sorted(field.name for field in fields(LMExperimentConfig) if field.name not in {"run_id", "hypothesis", "max_steps"}),
                "note": "Only include small function/hyperparameter/training-condition changes. Use epochs for training length; max_steps is recorded only as an effective update count.",
            },
        },
    }


def default_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "idle",
        "current_run": 0,
        "next_run": 1,
        "running_pid": None,
        "lock_path": str(LOCK_PATH.relative_to(ROOT)),
        "last_experiment_dir": None,
        "best_run": None,
        "last_updated": utc_now(),
    }


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return default_state()
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = STATE_PATH.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(STATE_PATH)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_corpus_path() -> Path:
    return DEFAULT_CORPUS_PATH if DEFAULT_CORPUS_PATH.exists() else FALLBACK_CORPUS_PATH


def inspect_hardware(device_name: str) -> dict[str, Any]:
    memory_bytes = detect_memory_bytes()
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    resolved = resolve_device(device_name)
    return {
        "timestamp": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "memory_gb": None if memory_bytes is None else round(memory_bytes / (1024**3), 2),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "mps_available": mps_available,
        "resolved_device": str(resolved),
        "profile": choose_hardware_profile(str(resolved), memory_bytes),
    }


def detect_memory_bytes() -> int | None:
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return int(out)
        except (OSError, ValueError, subprocess.CalledProcessError):
            return None
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages) * int(page_size)
        except (OSError, ValueError):
            return None
    return None


def choose_hardware_profile(device: str, memory_bytes: int | None) -> str:
    memory_gb = 0 if memory_bytes is None else memory_bytes / (1024**3)
    if device == "cuda":
        return "cuda_balanced" if memory_gb < 32 else "cuda_large"
    if device == "mps":
        return "mps_balanced" if memory_gb >= 16 else "mps_small"
    return "cpu_small" if memory_gb < 16 else "cpu_balanced"


def hardware_baseline(run_id: int, hardware: dict[str, Any]) -> LMExperimentConfig:
    profile = hardware.get("profile", "cpu_small")
    if profile == "cuda_large":
        return LMExperimentConfig(run_id=run_id, hypothesis="", max_steps=80, batch_size=16, emb_dim=192, n_heads=4, n_layers=3, context_length=96)
    if profile == "cuda_balanced":
        return LMExperimentConfig(run_id=run_id, hypothesis="", max_steps=60, batch_size=12, emb_dim=128, n_heads=4, n_layers=2, context_length=64)
    if profile == "mps_balanced":
        return LMExperimentConfig(run_id=run_id, hypothesis="", max_steps=40, batch_size=8, emb_dim=128, n_heads=4, n_layers=2, context_length=64)
    if profile == "mps_small":
        return LMExperimentConfig(run_id=run_id, hypothesis="", max_steps=20, batch_size=4, emb_dim=96, n_heads=4, n_layers=2, context_length=48)
    if profile == "cpu_balanced":
        return LMExperimentConfig(run_id=run_id, hypothesis="", max_steps=20, batch_size=4, emb_dim=96, n_heads=4, n_layers=2, context_length=48)
    return LMExperimentConfig(run_id=run_id, hypothesis="", max_steps=10, batch_size=2, emb_dim=64, n_heads=4, n_layers=1, context_length=32)


def choose_next_experiment(run_id: int, leaderboard_rows: list[dict[str, str]], hardware: dict[str, Any]) -> ExperimentPlan:
    base = hardware_baseline(run_id, hardware)
    fixed = ["vocab_size", "context_length", "batch_size", "learning_rate", "emb_dim", "n_heads", "n_layers", "seed"]
    if not leaderboard_rows:
        hypothesis = "기준선 수립: 현재 하드웨어에 맞춘 작은 GPT 설정으로 train/val gap과 처리량 기준점을 만든다."
        config = replace(base, hypothesis=hypothesis)
        return ExperimentPlan(
            config=config,
            hypothesis=hypothesis,
            rationale="아직 완료된 실험이 없으므로 과적합 판단과 속도 비교의 기준점이 필요하다.",
            changed_variables=asdict(config),
            fixed_variables=[],
            expected_result="validation loss가 소폭 내려가고 generalization gap이 작게 유지된다.",
            next_if_success="같은 계열에서 seed 반복 또는 약한 capacity 증가를 검토한다.",
            next_if_overfit="dropout, weight_decay, tie_embeddings를 강화한다.",
        )

    latest = leaderboard_rows[-1]
    best = choose_best_row(leaderboard_rows) or latest
    latest_status = latest.get("fit_status", "mixed")
    latest_gap = parse_float(latest.get("final_generalization_gap"))
    base = config_from_row(best, run_id, hardware)

    if latest_status == "overfit_risk" or latest_gap > 0.08:
        config = replace(
            base,
            run_id=run_id,
            seed=base.seed + 17,
            drop_rate=min(0.3, base.drop_rate + 0.05),
            weight_decay=max(base.weight_decay, 0.05),
            emb_dim=max(64, base.emb_dim // 2 if base.emb_dim > 128 else base.emb_dim),
            n_layers=max(1, base.n_layers - 1),
            ffn_mult=max(2, base.ffn_mult - 1),
            tie_embeddings=True,
        )
        config = ensure_head_divisible(config)
        hypothesis = "과적합 완화: gap이 커진 설정에서 regularization을 강화하고 모델 용량을 조금 낮춘다."
        return ExperimentPlan(
            config=replace(config, hypothesis=hypothesis),
            hypothesis=hypothesis,
            rationale="최근 결과가 validation보다 train 개선에 치우쳤으므로 외우는 힘을 줄이고 일반화 신호를 확인한다.",
            changed_variables={"drop_rate": config.drop_rate, "weight_decay": config.weight_decay, "emb_dim": config.emb_dim, "n_layers": config.n_layers, "ffn_mult": config.ffn_mult, "tie_embeddings": config.tie_embeddings},
            fixed_variables=fixed,
            expected_result="final_val_loss가 유지되거나 내려가면서 final_generalization_gap과 overfit_score가 감소한다.",
            next_if_success="현재 regularization 근처에서 seed 반복으로 안정성을 확인한다.",
            next_if_overfit="context_length/stride와 데이터 중복을 재검토한다.",
        )

    if latest_status == "underfit_or_too_short":
        next_epochs = round((base.epochs or (base.max_steps / 20)) * 1.5, 6)
        config = replace(
            base,
            run_id=run_id,
            seed=base.seed + 11,
            epochs=min(8.0, max(0.25, next_epochs)),
            learning_rate=5e-4 if base.learning_rate <= 3e-4 else base.learning_rate,
            emb_dim=min(256, base.emb_dim + 32),
            n_layers=min(4, base.n_layers + 1),
        )
        config = ensure_head_divisible(config)
        hypothesis = "과소학습 완화: 학습 epoch와 표현력을 늘려 validation 개선 여지가 있는지 확인한다."
        return ExperimentPlan(
            config=replace(config, hypothesis=hypothesis),
            hypothesis=hypothesis,
            rationale="train과 validation이 모두 충분히 내려가지 않았으므로 학습량 또는 표현력이 부족할 가능성이 있다.",
            changed_variables={"epochs": config.epochs, "learning_rate": config.learning_rate, "emb_dim": config.emb_dim, "n_layers": config.n_layers},
            fixed_variables=fixed,
            expected_result="train/val loss가 함께 내려가고 gap 증가는 작게 유지된다.",
            next_if_success="capacity 증가 폭을 유지하되 seed를 바꿔 재검증한다.",
            next_if_overfit="방금 늘린 capacity를 되돌리고 regularization을 강화한다.",
        )

    if latest_status == "generalizing":
        config = replace(base, run_id=run_id, seed=base.seed + 101)
        hypothesis = "안정성 검증: 좋아 보이는 설정을 seed만 바꿔 재현 가능한 개선인지 확인한다."
        return ExperimentPlan(
            config=replace(config, hypothesis=hypothesis),
            hypothesis=hypothesis,
            rationale="validation이 개선되고 gap이 안정적이므로 같은 설정이 초기화 운인지 확인한다.",
            changed_variables={"seed": config.seed},
            fixed_variables=[field for field in fixed if field != "seed"],
            expected_result="다른 seed에서도 validation loss와 gap이 비슷한 범위에 머문다.",
            next_if_success="가벼운 activation 또는 ffn_mult 변경으로 추가 개선을 탐색한다.",
            next_if_overfit="seed 변화에서 gap이 커지면 더 보수적인 regularization을 적용한다.",
        )

    activation_cycle = ["gelu_exact", "quick_gelu", "silu", "mish", "swiglu", "geglu"]
    activation = activation_cycle[(run_id - 1) % len(activation_cycle)]
    config = replace(base, run_id=run_id, seed=base.seed + 23, activation_name=activation)
    hypothesis = f"활성함수 탐색: {activation}가 기준 계열 대비 validation loss와 overfit_score를 개선하는지 확인한다."
    return ExperimentPlan(
        config=replace(config, hypothesis=hypothesis),
        hypothesis=hypothesis,
        rationale="최근 결과가 혼합 상태이므로 전체 구조를 바꾸지 않는 함수 교체로 개선 여지를 탐색한다.",
        changed_variables={"activation_name": activation, "seed": config.seed},
        fixed_variables=[field for field in fixed if field != "seed"],
        expected_result="validation loss가 내려가되 parameter_count와 overfit_score가 과하게 증가하지 않는다.",
        next_if_success="같은 activation을 seed 반복으로 검증한다.",
        next_if_overfit="gated activation이면 일반 activation으로 되돌리고 regularization을 강화한다.",
    )


def ensure_head_divisible(config: LMExperimentConfig) -> LMExperimentConfig:
    if config.emb_dim % config.n_heads == 0:
        return config
    for heads in [8, 4, 2, 1]:
        if config.emb_dim % heads == 0:
            return replace(config, n_heads=heads)
    return replace(config, emb_dim=64, n_heads=4)


def config_from_row(row: dict[str, str], run_id: int, hardware: dict[str, Any]) -> LMExperimentConfig:
    fallback = hardware_baseline(run_id, hardware)
    return LMExperimentConfig(
        run_id=run_id,
        hypothesis="",
        seed=parse_int(row.get("seed"), fallback.seed),
        vocab_size=parse_int(row.get("vocab_size"), fallback.vocab_size),
        context_length=parse_int(row.get("context_length"), fallback.context_length),
        stride=parse_optional_int(row.get("stride")),
        batch_size=parse_int(row.get("batch_size"), fallback.batch_size),
        epochs=parse_optional_float(row.get("epochs"), fallback.epochs),
        max_steps=parse_int(row.get("max_steps"), fallback.max_steps),
        learning_rate=parse_float(row.get("learning_rate"), fallback.learning_rate),
        weight_decay=parse_float(row.get("weight_decay"), fallback.weight_decay),
        grad_clip=parse_optional_float(row.get("grad_clip"), fallback.grad_clip),
        emb_dim=parse_int(row.get("emb_dim"), fallback.emb_dim),
        n_heads=parse_int(row.get("n_heads"), fallback.n_heads),
        n_layers=parse_int(row.get("n_layers"), fallback.n_layers),
        drop_rate=parse_float(row.get("drop_rate"), fallback.drop_rate),
        qkv_bias=parse_bool(row.get("qkv_bias"), fallback.qkv_bias),
        ffn_mult=parse_int(row.get("ffn_mult"), fallback.ffn_mult),
        norm_first=parse_bool(row.get("norm_first"), fallback.norm_first),
        norm_eps=parse_float(row.get("norm_eps"), fallback.norm_eps),
        activation_name=row.get("activation_name") or fallback.activation_name,
        ffn_dropout_position=row.get("ffn_dropout_position") or fallback.ffn_dropout_position,
        attention_impl=row.get("attention_impl") or fallback.attention_impl,
        tie_embeddings=parse_bool(row.get("tie_embeddings"), fallback.tie_embeddings),
        init_std=parse_float(row.get("init_std"), fallback.init_std),
    )


def read_active_lock() -> dict[str, Any] | None:
    if not LOCK_PATH.exists():
        return None
    try:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"pid": None, "stale": False, "reason": "unreadable lock"}
    pid = lock.get("pid")
    if isinstance(pid, int) and process_is_alive(pid):
        return lock
    LOCK_PATH.unlink(missing_ok=True)
    return None


def create_lock(run_id: int, artifact_dir: Path) -> None:
    payload = {"pid": os.getpid(), "run_id": run_id, "artifact_dir": str(artifact_dir.relative_to(ROOT)), "created_at": utc_now()}
    LOCK_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_leaderboard() -> list[dict[str, str]]:
    if not LEADERBOARD_PATH.exists():
        return []
    with LEADERBOARD_PATH.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def backfill_leaderboard_epoch_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Populate epoch columns for legacy rows that only stored max_steps."""
    backfilled: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        if updated.get("epochs") not in (None, "") and updated.get("steps_per_epoch") not in (None, ""):
            updated["changed_variables"] = epochize_changed_variables(updated.get("changed_variables"), updated.get("epochs"), updated.get("max_steps"))
            backfilled.append(updated)
            continue

        artifact_dir_text = updated.get("artifact_dir") or ""
        result = load_artifact_result(ROOT / artifact_dir_text) if artifact_dir_text else None
        max_steps = parse_int((result or {}).get("max_steps") or updated.get("max_steps"), 0)
        steps_per_epoch = parse_int((result or {}).get("steps_per_epoch"), 0)

        if steps_per_epoch <= 0 and result:
            train_token_count = parse_int(result.get("train_token_count"), 0)
            context_length = parse_int(result.get("context_length") or updated.get("context_length"), 0)
            batch_size = parse_int(result.get("batch_size") or updated.get("batch_size"), 0)
            stride = parse_optional_int(result.get("stride") if result.get("stride") not in (None, "") else updated.get("stride"))
            if train_token_count > 0 and context_length > 0 and batch_size > 0:
                steps_per_epoch = estimate_steps_per_epoch(train_token_count, context_length, batch_size, stride)

        epochs = parse_optional_float((result or {}).get("epochs"), None)
        if epochs is None and max_steps > 0 and steps_per_epoch > 0:
            epochs = max_steps / steps_per_epoch

        if epochs is not None:
            updated["epochs"] = round(epochs, 6)
        if steps_per_epoch > 0:
            updated["steps_per_epoch"] = steps_per_epoch
        if max_steps > 0:
            updated["max_steps"] = max_steps
        updated["changed_variables"] = epochize_changed_variables(updated.get("changed_variables"), updated.get("epochs"), updated.get("max_steps"))
        backfilled.append(updated)
    return backfilled


def epochize_changed_variables(changed_variables: Any, epochs: Any, max_steps: Any) -> Any:
    if changed_variables in (None, "") or epochs in (None, ""):
        return changed_variables
    try:
        payload = json.loads(str(changed_variables))
    except json.JSONDecodeError:
        return changed_variables
    if not isinstance(payload, dict) or "max_steps" not in payload:
        return changed_variables
    payload = dict(payload)
    legacy_steps = payload.pop("max_steps")
    payload.setdefault("epochs", epochs)
    payload.setdefault("effective_max_steps", max_steps or legacy_steps)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def write_leaderboard(rows: list[dict[str, Any]]) -> None:
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEADERBOARD_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LEADERBOARD_FIELDS})


def leaderboard_row(plan: ExperimentPlan, result: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    config = plan.config
    return {
        "run_id": config.run_id,
        "timestamp": utc_now(),
        "hypothesis": plan.hypothesis,
        "changed_variables": json.dumps(plan.changed_variables, ensure_ascii=False, sort_keys=True),
        "seed": config.seed,
        "vocab_size": config.vocab_size,
        "context_length": config.context_length,
        "stride": "" if config.stride is None else config.stride,
        "batch_size": config.batch_size,
        "epochs": result.get("epochs", config.epochs),
        "steps_per_epoch": result.get("steps_per_epoch"),
        "max_steps": result.get("max_steps", config.max_steps),
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "grad_clip": "" if config.grad_clip is None else config.grad_clip,
        "emb_dim": config.emb_dim,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "drop_rate": config.drop_rate,
        "qkv_bias": config.qkv_bias,
        "ffn_mult": config.ffn_mult,
        "norm_first": config.norm_first,
        "norm_eps": config.norm_eps,
        "activation_name": config.activation_name,
        "ffn_dropout_position": config.ffn_dropout_position,
        "attention_impl": config.attention_impl,
        "tie_embeddings": config.tie_embeddings,
        "init_std": config.init_std,
        "final_train_loss": result.get("final_train_loss"),
        "final_val_loss": result.get("final_val_loss"),
        "final_generalization_gap": result.get("final_generalization_gap"),
        "generalization_gap_delta": result.get("generalization_gap_delta"),
        "train_val_improvement_gap": result.get("train_val_improvement_gap"),
        "overfit_score": result.get("overfit_score"),
        "fit_status": result.get("fit_status"),
        "parameter_count": result.get("parameter_count"),
        "tokens_per_sec": result.get("tokens_per_sec"),
        "elapsed_sec": result.get("elapsed_sec"),
        "device": result.get("device"),
        "artifact_dir": str(artifact_dir.relative_to(ROOT)),
    }


def choose_best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None

    def score(row: dict[str, Any]) -> float:
        val_loss = parse_float(row.get("final_val_loss"), 1e9)
        overfit = parse_float(row.get("overfit_score"), 0.0)
        gap = max(0.0, parse_float(row.get("final_generalization_gap"), 0.0))
        status_penalty = 0.5 if row.get("fit_status") == "overfit_risk" else 0.0
        return val_loss + 0.25 * overfit + 0.5 * gap + status_penalty

    return min(rows, key=score)


def next_run_number(state: dict[str, Any], leaderboard_rows: list[dict[str, str]]) -> int:
    state_next = parse_int(state.get("next_run"), 1)
    if not leaderboard_rows:
        return max(1, state_next)
    max_seen = max(parse_int(row.get("run_id"), 0) for row in leaderboard_rows)
    return max(state_next, max_seen + 1)


def refresh_visual_metrics_from_leaderboard(leaderboard_rows: list[dict[str, Any]]) -> dict[str, Path]:
    backfill_run_metric_svgs(leaderboard_rows)
    latest = leaderboard_rows[-1]
    artifact_dir_text = latest.get("artifact_dir") or "docs/train/visuals"
    artifact_dir = ROOT / artifact_dir_text
    latest_result = load_artifact_result(artifact_dir) or latest
    return update_visual_metrics(leaderboard_rows, latest_result, artifact_dir)


def update_visual_metrics(leaderboard_rows: list[dict[str, Any]], latest_result: dict[str, Any], artifact_dir: Path, *, write_artifact_latest: bool = True) -> dict[str, Path]:
    summary_rows = build_metrics_summary(leaderboard_rows)
    if summary_rows:
        write_csv_rows(summary_rows, METRICS_SUMMARY_PATH)

    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    TREND_SVG_PATH.write_text(render_trend_svg(summary_rows), encoding="utf-8")

    latest_run_svg = artifact_dir / "run_metrics.svg"
    if write_artifact_latest:
        latest_run_svg.write_text(render_latest_run_svg(latest_result), encoding="utf-8")
        shutil.copy2(latest_run_svg, LATEST_RUN_SVG_PATH)
    elif latest_run_svg.exists():
        shutil.copy2(latest_run_svg, LATEST_RUN_SVG_PATH)
    else:
        latest_run_svg = LATEST_RUN_SVG_PATH
        latest_run_svg.write_text(render_latest_run_svg(latest_result), encoding="utf-8")

    write_dashboard(summary_rows)
    return {
        "dashboard": DASHBOARD_PATH,
        "metrics_summary": METRICS_SUMMARY_PATH,
        "trend": TREND_SVG_PATH,
        "latest_run": latest_run_svg,
        "latest_run_global": LATEST_RUN_SVG_PATH,
    }


def load_artifact_result(artifact_dir: Path) -> dict[str, Any] | None:
    jsonl_path = artifact_dir / "results.jsonl"
    if jsonl_path.exists():
        lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            try:
                return json.loads(lines[-1])
            except json.JSONDecodeError:
                return None
    csv_path = artifact_dir / "results.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if rows:
            return rows[-1]
    return None


def backfill_run_metric_svgs(leaderboard_rows: list[dict[str, Any]]) -> None:
    for row in leaderboard_rows:
        artifact_dir_text = row.get("artifact_dir")
        if not artifact_dir_text:
            continue
        artifact_dir = ROOT / artifact_dir_text
        result = load_artifact_result(artifact_dir)
        if result is None:
            continue
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "run_metrics.svg").write_text(render_latest_run_svg(result), encoding="utf-8")


def build_metrics_summary(leaderboard_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best = choose_best_row(leaderboard_rows)
    best_run_id = None if best is None else str(best.get("run_id"))
    for row in leaderboard_rows:
        gap = parse_float(row.get("final_generalization_gap"))
        overfit = parse_float(row.get("overfit_score"))
        val_loss = parse_float(row.get("final_val_loss"), 1e9)
        status = str(row.get("fit_status") or "")
        status_penalty = 0.5 if status == "overfit_risk" else 0.0
        selection_score = val_loss + 0.25 * overfit + 0.5 * max(0.0, gap) + status_penalty
        run_id = str(row.get("run_id") or "")
        rows.append(
            {
                "run_id": run_id,
                "fit_status": status,
                "risk_level": overfit_risk_level(row),
                "best_candidate": str(run_id == best_run_id),
                "selection_score": round(selection_score, 6),
                "epochs": numeric_or_blank(row.get("epochs")),
                "steps_per_epoch": numeric_or_blank(row.get("steps_per_epoch")),
                "max_steps": numeric_or_blank(row.get("max_steps")),
                "final_train_loss": numeric_or_blank(row.get("final_train_loss")),
                "final_val_loss": numeric_or_blank(row.get("final_val_loss")),
                "final_generalization_gap": numeric_or_blank(row.get("final_generalization_gap")),
                "generalization_gap_delta": numeric_or_blank(row.get("generalization_gap_delta")),
                "train_val_improvement_gap": numeric_or_blank(row.get("train_val_improvement_gap")),
                "overfit_score": numeric_or_blank(row.get("overfit_score")),
                "parameter_count": numeric_or_blank(row.get("parameter_count")),
                "tokens_per_sec": numeric_or_blank(row.get("tokens_per_sec")),
                "device": row.get("device", ""),
                "changed_variables": row.get("changed_variables", ""),
                "hypothesis": row.get("hypothesis", ""),
                "artifact_dir": row.get("artifact_dir", ""),
            }
        )
    return rows


def write_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def overfit_risk_level(row: dict[str, Any]) -> str:
    status = str(row.get("fit_status") or "")
    gap = parse_float(row.get("final_generalization_gap"))
    overfit = parse_float(row.get("overfit_score"))
    gap_delta = parse_float(row.get("generalization_gap_delta"))
    if status == "overfit_risk" or overfit >= 0.15 or gap >= 0.05 or gap_delta >= 0.05:
        return "high"
    if overfit >= 0.08 or gap >= 0.03 or gap_delta >= 0.03:
        return "medium"
    return "low"


def numeric_or_blank(value: Any) -> Any:
    if value in (None, ""):
        return ""
    parsed = parse_float(value)
    return round(parsed, 6)


def render_trend_svg(summary_rows: list[dict[str, Any]]) -> str:
    width, height = 980, 700
    parts = [
        svg_header(width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(36, 44, "mini GPT Training Metrics", size=24, weight="700"),
        svg_text(36, 70, "Loss trend and overfit signals across automation runs", size=13, fill="#475569"),
    ]
    parts.extend(render_line_panel(summary_rows, 58, 105, 860, 230, "Train / Validation Loss", [("final_train_loss", "#2563eb", "train"), ("final_val_loss", "#dc2626", "val")], zero_floor=False))
    parts.extend(render_line_panel(summary_rows, 58, 405, 860, 210, "Generalization / Overfit", [("final_generalization_gap", "#d97706", "gap"), ("overfit_score", "#7c3aed", "overfit")], zero_floor=True, thresholds=[(0.05, "gap risk"), (0.12, "score watch")]))
    parts.append("</svg>")
    return "\n".join(parts)


def render_latest_run_svg(result: dict[str, Any]) -> str:
    width, height = 820, 520
    loss_items = [
        ("initial train", parse_float(result.get("initial_train_loss"))),
        ("final train", parse_float(result.get("final_train_loss"))),
        ("initial val", parse_float(result.get("initial_val_loss"))),
        ("final val", parse_float(result.get("final_val_loss"))),
    ]
    overfit_items = [
        ("final gap", parse_float(result.get("final_generalization_gap"))),
        ("gap delta", parse_float(result.get("generalization_gap_delta"))),
        ("improve gap", parse_float(result.get("train_val_improvement_gap"))),
        ("overfit score", parse_float(result.get("overfit_score"))),
    ]
    status = str(result.get("fit_status") or "unknown")
    run_id = result.get("run_id", "?")
    epoch_text = f"epochs={numeric_or_blank(result.get('epochs'))} / steps={numeric_or_blank(result.get('max_steps'))}"
    parts = [
        svg_header(width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(34, 42, f"Run {run_id} Metrics", size=23, weight="700"),
        svg_text(34, 68, f"fit_status={status} / {epoch_text} / device={result.get('device', '')} / tokens_per_sec={numeric_or_blank(result.get('tokens_per_sec'))}", size=13, fill="#475569"),
    ]
    parts.extend(render_bar_panel(loss_items, 48, 108, 710, 170, "Loss Snapshot", "#2563eb"))
    parts.extend(render_bar_panel(overfit_items, 48, 338, 710, 125, "Overfit Signals", "#7c3aed", max_hint=0.18))
    parts.append("</svg>")
    return "\n".join(parts)


def render_line_panel(
    rows: list[dict[str, Any]],
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    series: list[tuple[str, str, str]],
    *,
    zero_floor: bool,
    thresholds: list[tuple[float, str]] | None = None,
) -> list[str]:
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>',
        svg_text(x + 18, y + 28, title, size=16, weight="700"),
    ]
    plot_x, plot_y, plot_w, plot_h = x + 54, y + 48, width - 95, height - 86
    values = [parse_float(row.get(key)) for row in rows for key, _, _ in series if row.get(key) not in (None, "")]
    min_v, max_v = chart_bounds(values, zero_floor=zero_floor)
    parts.extend(draw_axes(plot_x, plot_y, plot_w, plot_h, min_v, max_v))
    if thresholds:
        for threshold, label in thresholds:
            if min_v <= threshold <= max_v:
                ty = map_y(threshold, min_v, max_v, plot_y, plot_h)
                parts.append(f'<line x1="{plot_x}" y1="{ty:.2f}" x2="{plot_x + plot_w}" y2="{ty:.2f}" stroke="#ef4444" stroke-dasharray="5 5" opacity="0.6"/>')
                parts.append(svg_text(plot_x + plot_w + 8, ty + 4, label, size=11, fill="#ef4444"))
    for index, (key, color, label) in enumerate(series):
        points: list[tuple[float, float]] = []
        for row_index, row in enumerate(rows):
            value = row.get(key)
            if value in (None, ""):
                continue
            px = map_x(row_index, max(1, len(rows)), plot_x, plot_w)
            py = map_y(parse_float(value), min_v, max_v, plot_y, plot_h)
            points.append((px, py))
        if points:
            point_text = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
            parts.append(f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="3"/>')
            for px, py in points:
                parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="{color}"/>')
        legend_x = x + width - 165
        legend_y = y + 26 + index * 20
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(svg_text(legend_x + 18, legend_y, label, size=12, fill="#334155"))
    for row_index, row in enumerate(rows):
        px = map_x(row_index, max(1, len(rows)), plot_x, plot_w)
        parts.append(svg_text(px - 8, plot_y + plot_h + 28, str(row.get("run_id", "")), size=11, fill="#64748b"))
    return parts


def render_bar_panel(items: list[tuple[str, float]], x: int, y: int, width: int, height: int, title: str, color: str, max_hint: float | None = None) -> list[str]:
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>',
        svg_text(x + 18, y + 28, title, size=16, weight="700"),
    ]
    max_value = max([value for _, value in items] + [max_hint or 0.0, 1e-9])
    bar_x = x + 145
    bar_w_max = width - 230
    for index, (label, value) in enumerate(items):
        row_y = y + 54 + index * 28
        bar_w = 0 if max_value == 0 else (value / max_value) * bar_w_max
        parts.append(svg_text(x + 18, row_y + 14, label, size=12, fill="#334155"))
        parts.append(f'<rect x="{bar_x}" y="{row_y}" width="{bar_w:.2f}" height="18" rx="4" fill="{color}" opacity="0.82"/>')
        parts.append(svg_text(bar_x + bar_w + 8, row_y + 14, f"{value:.4f}", size=12, fill="#0f172a"))
    return parts


def draw_axes(x: int, y: int, width: int, height: int, min_v: float, max_v: float) -> list[str]:
    parts = [
        f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" stroke="#94a3b8"/>',
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" stroke="#94a3b8"/>',
    ]
    for tick in range(5):
        value = min_v + (max_v - min_v) * tick / 4
        ty = map_y(value, min_v, max_v, y, height)
        parts.append(f'<line x1="{x}" y1="{ty:.2f}" x2="{x + width}" y2="{ty:.2f}" stroke="#e2e8f0"/>')
        parts.append(svg_text(x - 48, ty + 4, f"{value:.3f}", size=10, fill="#64748b"))
    return parts


def chart_bounds(values: list[float], *, zero_floor: bool) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    min_v = 0.0 if zero_floor else min(values)
    max_v = max(values)
    if min_v == max_v:
        padding = 0.1 if max_v == 0 else abs(max_v) * 0.08
        return (max(0.0, min_v - padding) if zero_floor else min_v - padding, max_v + padding)
    padding = (max_v - min_v) * 0.08
    lower = max(0.0, min_v - padding) if zero_floor else min_v - padding
    return (lower, max_v + padding)


def map_x(index: int, count: int, x: int, width: int) -> float:
    if count <= 1:
        return x + width / 2
    return x + width * index / (count - 1)


def map_y(value: float, min_v: float, max_v: float, y: int, height: int) -> float:
    if max_v == min_v:
        return y + height / 2
    return y + height - (value - min_v) / (max_v - min_v) * height


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">'


def svg_text(x: float, y: float, text: Any, *, size: int = 12, fill: str = "#0f172a", weight: str = "400") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}">{html.escape(str(text))}</text>'


def write_dashboard(summary_rows: list[dict[str, Any]]) -> None:
    latest = summary_rows[-1] if summary_rows else {}
    best = min(summary_rows, key=lambda row: parse_float(row.get("selection_score"), 1e9)) if summary_rows else {}
    recent_rows = summary_rows[-10:]
    table_rows = "\n".join(
        f"| {row.get('run_id')} | {row.get('fit_status')} | {row.get('risk_level')} | {row.get('epochs')} | {row.get('max_steps')} | {row.get('final_train_loss')} | {row.get('final_val_loss')} | {row.get('final_generalization_gap')} | {row.get('overfit_score')} |"
        for row in recent_rows
    )
    content = f"""# mini GPT 학습 대시보드

자동화가 갱신하는 시각 지표 모음입니다.

## 현재 요약

- 최신 run: `{latest.get("run_id", "")}` / status=`{latest.get("fit_status", "")}` / risk=`{latest.get("risk_level", "")}`
- 최신 epochs: `{latest.get("epochs", "")}` / effective steps=`{latest.get("max_steps", "")}` / steps_per_epoch=`{latest.get("steps_per_epoch", "")}`
- 최신 final_val_loss: `{latest.get("final_val_loss", "")}`
- 최신 generalization gap: `{latest.get("final_generalization_gap", "")}`
- 최신 overfit_score: `{latest.get("overfit_score", "")}`
- 현재 best 후보: run `{best.get("run_id", "")}` / score=`{best.get("selection_score", "")}` / val=`{best.get("final_val_loss", "")}`

## 전체 추세

![loss and overfit trends](visuals/loss_overfit_trends.svg)

## 최신 run 상세

![latest run metrics](visuals/latest_run_metrics.svg)

## 최근 10회

| run | status | risk | epochs | steps | train | val | gap | overfit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{table_rows}

## 파일

- `metrics_summary.csv`: 시각화와 해석을 위한 정규화된 지표
- `visuals/loss_overfit_trends.svg`: 전체 loss/gap/overfit 추세
- `visuals/latest_run_metrics.svg`: 최신 run의 loss와 과적합 신호
"""
    DASHBOARD_PATH.write_text(content, encoding="utf-8")


def write_run_report(report_path: Path, plan: ExperimentPlan, result: dict[str, Any], hardware: dict[str, Any], corpus_path: Path, artifact_dir: Path, best_row: dict[str, Any] | None, visual_paths: dict[str, Path] | None = None) -> None:
    config = plan.config
    best_line = "없음" if best_row is None else f"run {best_row.get('run_id')} / val={best_row.get('final_val_loss')} / status={best_row.get('fit_status')}"
    latest_visual = "run_metrics.svg"
    trend_visual = "../visuals/loss_overfit_trends.svg"
    dashboard_path = "../dashboard.md"
    content = f"""# run {config.run_id:03d} 실험 보고서

## 이번 가설

{plan.hypothesis}

## 왜 이 가설을 세웠는가

{plan.rationale}

## 가설 작성 주체

{getattr(plan, "source", "rule_based")}

## 바꾼 변수

```json
{json.dumps(plan.changed_variables, ensure_ascii=False, indent=2)}
```

## 고정한 변수

{", ".join(plan.fixed_variables) if plan.fixed_variables else "첫 기준선이므로 고정 비교군 없음"}

## 기대 결과

{plan.expected_result}

## 실험 설정

```json
{json.dumps(asdict(config), ensure_ascii=False, indent=2)}
```

## 학습 길이

| 항목 | 값 |
| --- | --- |
| epochs | {result.get("epochs")} |
| steps_per_epoch | {result.get("steps_per_epoch")} |
| effective max_steps | {result.get("max_steps")} |

## 실행 환경

```json
{json.dumps(hardware, ensure_ascii=False, indent=2)}
```

- corpus: `{corpus_path.relative_to(ROOT) if corpus_path.is_relative_to(ROOT) else corpus_path}`
- artifact_dir: `{artifact_dir.relative_to(ROOT)}`

## 실제 결과

| 지표 | 값 |
| --- | --- |
| initial_train_loss | {result.get("initial_train_loss")} |
| initial_val_loss | {result.get("initial_val_loss")} |
| final_train_loss | {result.get("final_train_loss")} |
| final_val_loss | {result.get("final_val_loss")} |
| final_generalization_gap | {result.get("final_generalization_gap")} |
| generalization_gap_delta | {result.get("generalization_gap_delta")} |
| train_val_improvement_gap | {result.get("train_val_improvement_gap")} |
| overfit_score | {result.get("overfit_score")} |
| fit_status | {result.get("fit_status")} |
| epochs | {result.get("epochs")} |
| steps_per_epoch | {result.get("steps_per_epoch")} |
| effective max_steps | {result.get("max_steps")} |
| parameter_count | {result.get("parameter_count")} |
| tokens_per_sec | {result.get("tokens_per_sec")} |
| elapsed_sec | {result.get("elapsed_sec")} |
| device | {result.get("device")} |

## 시각 지표

![이번 run 지표]({artifact_dir.name}/{latest_visual})

![전체 loss/overfit 추세]({trend_visual})

- 대시보드: `{dashboard_path}`
- 지표 요약 CSV: `../metrics_summary.csv`

## 과적합 판단

{interpret_result(result)}

## 결론

현재 best 후보: {best_line}

## 다음 실험 제안

- 성공 시: {plan.next_if_success}
- 과적합 시: {plan.next_if_overfit}
"""
    report_path.write_text(content, encoding="utf-8")


def write_failure_report(report_path: Path, plan: ExperimentPlan, hardware: dict[str, Any], failure: dict[str, Any]) -> None:
    content = f"""# run {plan.config.run_id:03d} 실패 보고서

## 이번 가설

{plan.hypothesis}

## 실패 정보

```json
{json.dumps(failure, ensure_ascii=False, indent=2)}
```

## 실행 환경

```json
{json.dumps(hardware, ensure_ascii=False, indent=2)}
```

## 다음 조치

동일 설정을 반복하기 전에 corpus 길이, device 지원 여부, batch/context 크기를 확인한다.
"""
    report_path.write_text(content, encoding="utf-8")


def append_hypothesis_log(plan: ExperimentPlan, result: dict[str, Any], report_path: Path) -> None:
    entry = f"""
## run {plan.config.run_id:03d} - {utc_now()}

- 보고서: `{report_path.relative_to(ROOT)}`
- 이번 가설: {plan.hypothesis}
- 근거: {plan.rationale}
- 바꾼 변수: `{json.dumps(plan.changed_variables, ensure_ascii=False, sort_keys=True)}`
- 기대 결과: {plan.expected_result}
- 실제 결과: epochs={result.get("epochs")}, steps={result.get("max_steps")}, final_val_loss={result.get("final_val_loss")}, gap={result.get("final_generalization_gap")}, overfit_score={result.get("overfit_score")}, fit_status={result.get("fit_status")}
- 과적합 판단: {interpret_result(result)}
- 다음 가설: 성공 시 {plan.next_if_success} / 과적합 시 {plan.next_if_overfit}
"""
    with HYPOTHESES_PATH.open("a", encoding="utf-8") as file:
        file.write(entry)


def interpret_result(result: dict[str, Any]) -> str:
    status = result.get("fit_status")
    gap = parse_float(result.get("final_generalization_gap"), 0.0)
    overfit = parse_float(result.get("overfit_score"), 0.0)
    if status == "overfit_risk":
        return f"과적합 위험. final gap={gap:.4f}, overfit_score={overfit:.4f}. 다음 실험은 regularization 강화가 우선이다."
    if status == "generalizing":
        return f"일반화 개선 신호. final gap={gap:.4f}, overfit_score={overfit:.4f}. seed 반복으로 재현성을 확인할 만하다."
    if status == "underfit_or_too_short":
        return f"과소학습 또는 너무 짧은 학습. final gap={gap:.4f}. epoch/capacity 증가를 검토한다."
    if status == "val_regressed":
        return f"validation 악화. final gap={gap:.4f}. learning rate나 capacity를 보수적으로 조정한다."
    return f"혼합 신호. final gap={gap:.4f}, overfit_score={overfit:.4f}. 한 축만 바꾼 후속 실험이 필요하다."


def result_summary(result: dict[str, Any]) -> dict[str, Any]:
    keys = ["epochs", "steps_per_epoch", "max_steps", "final_train_loss", "final_val_loss", "final_generalization_gap", "overfit_score", "fit_status", "parameter_count", "tokens_per_sec", "device"]
    return {key: result.get(key) for key in keys}


def update_readme(hardware: dict[str, Any]) -> None:
    content = f"""# mini GPT 자동 학습 실험 루프

이 폴더는 Codex 자동화가 반복적으로 가설을 세우고, 한 회차 실험을 실행하고, 결과를 분석한 뒤 다음 가설을 남기기 위한 작업 공간입니다.

## 현재 하드웨어 요약

```json
{json.dumps(hardware, ensure_ascii=False, indent=2)}
```

## 실행 방식

자동화는 몇 분마다 다음 명령을 실행합니다.

```bash
python -m src.train_loop_agent
```

각 실행은 다음 순서로 동작합니다.

1. `docs/train/state.json`을 읽는다.
2. `docs/train/.train_loop.lock`이 살아 있으면 진행 중으로 보고 종료한다.
3. 직전 결과와 leaderboard를 분석한다.
4. Codex/LLM이 작성한 `next_plan.json`이 있으면 그 가설을 우선 사용하고, 없으면 규칙 기반 fallback으로 다음 가설과 설정을 만든다.
5. 한 회차 실험을 실행한다.
6. `docs/train/runs/run_XXX.md` 보고서를 쓴다.
7. `docs/train/leaderboard.csv`와 `docs/train/hypotheses.md`를 갱신한다.
8. `docs/train/dashboard.md`와 `docs/train/visuals/`의 시각 지표를 갱신한다.

## 결과 해석 기준

- `epochs`는 사람이 지정하는 학습 길이이고, 실행 시 `steps_per_epoch`와 곱해 실제 optimizer update 수로 환산된다.
- 과거 호환성을 위해 `max_steps`도 결과에 기록하지만, 새 실험 계획 입력은 `epochs`를 사용한다.
- `final_val_loss`가 낮을수록 좋다.
- `final_generalization_gap = final_val_loss - final_train_loss`가 커지면 과적합 위험이다.
- `overfit_score`는 낮을수록 좋다.
- `fit_status == "generalizing"`이면 다음에는 seed 반복으로 재현성을 확인한다.
- `fit_status == "overfit_risk"`이면 dropout, weight decay, tying, 모델 축소를 우선한다.

## 시각화

- `dashboard.md`: loss, generalization gap, overfit_score를 한 화면에서 보는 요약 대시보드
- `metrics_summary.csv`: 시각화와 해석에 쓰는 정규화된 지표 테이블
- `visuals/loss_overfit_trends.svg`: 모든 run의 train/val loss와 과적합 신호 추세
- `visuals/latest_run_metrics.svg`: 최신 run의 loss와 과적합 신호 막대 그래프
- `runs/run_XXX_artifacts/run_metrics.svg`: 각 회차 보고서에 포함되는 run별 상세 그래프

## 주요 파일

- `state.json`: 현재 상태와 best run
- `leaderboard.csv`: 모든 완료 실험 요약
- `metrics_summary.csv`: 주요 loss/과적합 지표 요약
- `dashboard.md`: 사람이 바로 볼 수 있는 시각 대시보드
- `visuals/`: SVG 그래프
- `hypotheses.md`: 가설과 결론 누적 기록
- `runs/run_XXX.md`: 회차별 보고서
- `runs/run_XXX_artifacts/`: 해당 실험의 plan/result 파일
- `next_plan.json`: Codex/LLM이 자유롭게 세운 다음 가설. 실행되면 해당 run artifact로 이동한다.
- `next_plan.schema.json`: `next_plan.json` 작성 형식
"""
    README_PATH.write_text(content, encoding="utf-8")


def parse_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_optional_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    return parse_float(value, 0.0)


def parse_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return parse_int(value)


def parse_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    main()
