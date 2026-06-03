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
import math
import os
import platform
import re
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
    from .experiments import LMExperimentConfig, _split_corpus_text, estimate_steps_per_epoch, load_corpus, resolve_device, run_experiment_plan, write_plan
except ImportError:
    from experiments import LMExperimentConfig, _split_corpus_text, estimate_steps_per_epoch, load_corpus, resolve_device, run_experiment_plan, write_plan


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
EFFECT_MAP_PATH = TRAIN_DIR / "effect_map.md"
TOKENIZER_PROFILE_PATH = TRAIN_DIR / "tokenizer_profile.md"
RESEARCH_QUESTIONS_PATH = TRAIN_DIR / "research_questions.md"
CORRELATION_REPORT_PATH = TRAIN_DIR / "correlation_report.md"
VISUALS_DIR = TRAIN_DIR / "visuals"
TREND_SVG_PATH = VISUALS_DIR / "loss_overfit_trends.svg"
LATEST_RUN_SVG_PATH = VISUALS_DIR / "latest_run_metrics.svg"
EFFECT_MAP_SVG_PATH = VISUALS_DIR / "effect_map_pairs.svg"
CORRELATION_SVG_PATH = VISUALS_DIR / "correlation_evidence.svg"
DEFAULT_CORPUS_PATH = ROOT / "data" / "nsmc_lm_train.txt"
FALLBACK_CORPUS_PATH = ROOT / "src" / "learning" / "the-verdict.txt"
LAST_RESORT_CORPUS_PATH = ROOT / "README.md"
HY_TESTRESULT_DIR = ROOT / "docs" / "HY" / "testresult"
DEFAULT_CHAR_LIMIT = 20_000
_DEFAULT_SPLIT_CHAR_COUNT_CACHE: dict[float, tuple[int, int]] = {}

LEADERBOARD_FIELDS = [
    "run_id",
    "timestamp",
    "hypothesis",
    "changed_variables",
    "seed",
    "vocab_size",
    "actual_vocab_size",
    "bpe_merge_count",
    "corpus_char_count",
    "corpus_sha256",
    "train_char_count",
    "val_char_count",
    "train_token_count",
    "val_token_count",
    "train_tokens_per_char",
    "val_tokens_per_char",
    "train_chars_per_token",
    "val_chars_per_token",
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
    "final_train_nats_per_char",
    "final_val_nats_per_char",
    "final_train_bits_per_char",
    "final_val_bits_per_char",
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
        dry_run_config = asdict(plan.config)
        if dry_run_config.get("epochs") is not None:
            dry_run_config.pop("max_steps", None)
        state.update(
            {
                "status": "idle",
                "next_run": next_run_id,
                "last_updated": utc_now(),
                "hardware": hardware,
                "dry_run_next_hypothesis": plan.hypothesis,
                "dry_run_next_config": dry_run_config,
                "dry_run_plan_source": getattr(plan, "source", "rule_based"),
            }
        )
        save_state(state)
        print(json.dumps({"next_run": next_run_id, "hypothesis": plan.hypothesis, "plan_source": getattr(plan, "source", "rule_based"), "config": dry_run_config, "hardware": hardware}, ensure_ascii=False, indent=2))
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
        baseline_run: str | int | None = None,
        changed_variable: str | None = None,
        expected_intermediate_change: str | None = None,
        primary_metric: str | None = None,
        guardrail_metric: str | None = None,
        interpretation_rule: str | None = None,
        correlation_target: str | None = None,
        control_variables: list[str] | None = None,
        evidence_level: str | None = None,
    ):
        self.config = config
        self.hypothesis = hypothesis
        self.rationale = rationale
        self.changed_variables = changed_variables
        self.fixed_variables = fixed_variables
        self.expected_result = expected_result
        self.next_if_success = next_if_success
        self.next_if_overfit = next_if_overfit
        self.baseline_run = baseline_run
        self.changed_variable = changed_variable
        self.expected_intermediate_change = expected_intermediate_change
        self.primary_metric = primary_metric
        self.guardrail_metric = guardrail_metric
        self.interpretation_rule = interpretation_rule
        self.correlation_target = correlation_target
        self.control_variables = control_variables or []
        self.evidence_level = evidence_level
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
        baseline_run=payload.get("baseline_run", plan.baseline_run),
        changed_variable=str(payload.get("changed_variable") or plan.changed_variable or ""),
        expected_intermediate_change=str(payload.get("expected_intermediate_change") or plan.expected_intermediate_change or ""),
        primary_metric=str(payload.get("primary_metric") or plan.primary_metric or ""),
        guardrail_metric=str(payload.get("guardrail_metric") or plan.guardrail_metric or ""),
        interpretation_rule=str(payload.get("interpretation_rule") or plan.interpretation_rule or ""),
        correlation_target=str(payload.get("correlation_target") or plan.correlation_target or ""),
        control_variables=list(payload.get("control_variables") or plan.control_variables or []),
        evidence_level=str(payload.get("evidence_level") or plan.evidence_level or ""),
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
        "description": "python -m src.train_loop_agent가 읽어 실행하는 다음 실험 계획 형식입니다. 사람이 읽는 모든 설명 문장은 한국어로 작성합니다.",
        "language_policy": "hypothesis, rationale, expected_result, next_if_success, next_if_overfit, expected_intermediate_change, interpretation_rule은 한국어 문장으로 작성합니다. metric key와 config key 같은 식별자만 영어 원문을 유지합니다.",
        "required": ["hypothesis", "rationale", "expected_result", "baseline_run", "changed_variable", "expected_intermediate_change", "primary_metric", "guardrail_metric", "interpretation_rule", "correlation_target", "control_variables", "evidence_level", "config_overrides"],
        "properties": {
            "hypothesis": "다음 run에서 검증할 연구 가설입니다.",
            "rationale": "leaderboard와 최신 보고서에서 이 가설이 나온 이유입니다.",
            "baseline_run": "짝비교 기준으로 삼을 run id입니다. 첫 기준선일 때만 null을 사용합니다.",
            "changed_variable": "해석 가능성을 위해 의도적으로 하나만 바꾸는 변수입니다.",
            "expected_intermediate_change": "최종 점수 전에 관찰될 것으로 기대하는 메커니즘 변화입니다. 예: train_loss 하락, gap 상승.",
            "primary_metric": "성공 여부를 1차로 판단하는 주 지표입니다.",
            "guardrail_metric": "주 지표를 기각할 수 있는 과적합 또는 비교 가능성 보호 지표입니다.",
            "interpretation_rule": "결과가 무엇을 뜻하는지 판단하는 자연어 규칙입니다.",
            "correlation_target": "검증하려는 상관 관계입니다. 예: vocab_size -> val_nats_per_char.",
            "control_variables": "상관 해석을 가능하게 하려고 고정하는 변수 목록입니다.",
            "evidence_level": "observed_correlation, matched_pair, controlled_repeat, factorial_test 중 하나를 사용합니다.",
            "changed_variables": "의도적으로 바꾼 변수를 담은 객체입니다.",
            "fixed_variables": "의도적으로 고정한 변수 목록입니다.",
            "expected_result": "예상되는 train/validation/overfit 변화입니다.",
            "next_if_success": "결과가 일반화되면 이어갈 후속 실험입니다.",
            "next_if_overfit": "과적합이 나타나면 이어갈 후속 실험입니다.",
            "config_overrides": {
                "allowed_keys": sorted(field.name for field in fields(LMExperimentConfig) if field.name not in {"run_id", "hypothesis", "max_steps"}),
                "note": "작은 함수/하이퍼파라미터/학습 조건 변경만 포함합니다. 학습 길이는 epochs로 작성하고, max_steps는 실제 업데이트 수 기록용 결과 필드로만 둡니다.",
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
    if DEFAULT_CORPUS_PATH.exists():
        return DEFAULT_CORPUS_PATH
    if FALLBACK_CORPUS_PATH.exists():
        return FALLBACK_CORPUS_PATH
    return LAST_RESORT_CORPUS_PATH


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
        return LMExperimentConfig(run_id=run_id, hypothesis="", epochs=2.0, batch_size=16, emb_dim=192, n_heads=4, n_layers=3, context_length=96)
    if profile == "cuda_balanced":
        return LMExperimentConfig(run_id=run_id, hypothesis="", epochs=2.0, batch_size=12, emb_dim=128, n_heads=4, n_layers=2, context_length=64)
    if profile == "mps_balanced":
        return LMExperimentConfig(run_id=run_id, hypothesis="", epochs=2.5, batch_size=8, emb_dim=128, n_heads=4, n_layers=2, context_length=64)
    if profile == "mps_small":
        return LMExperimentConfig(run_id=run_id, hypothesis="", epochs=1.5, batch_size=4, emb_dim=96, n_heads=4, n_layers=2, context_length=48)
    if profile == "cpu_balanced":
        return LMExperimentConfig(run_id=run_id, hypothesis="", epochs=1.5, batch_size=4, emb_dim=96, n_heads=4, n_layers=2, context_length=48)
    return LMExperimentConfig(run_id=run_id, hypothesis="", epochs=1.0, batch_size=2, emb_dim=64, n_heads=4, n_layers=1, context_length=32)


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
            baseline_run=None,
            changed_variable="baseline",
            expected_intermediate_change="train_loss와 val_loss가 함께 내려가는지 확인한다.",
            primary_metric="final_val_loss",
            guardrail_metric="final_generalization_gap, overfit_score, loss/과적합 그래프",
            interpretation_rule="validation loss가 내려가도 gap과 overfit_score가 같이 커지면 성공으로 보지 않는다.",
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
            baseline_run=best.get("run_id"),
            changed_variable="regularization_capacity_bundle",
            expected_intermediate_change="train_loss 하락 속도가 줄고 gap/overfit_score가 내려간다.",
            primary_metric="overfit_score",
            guardrail_metric="final_val_loss, loss/과적합 그래프",
            interpretation_rule="overfit_score가 줄어도 final_val_loss가 크게 악화되면 underfit으로 본다.",
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
            baseline_run=best.get("run_id"),
            changed_variable="epochs_capacity_bundle",
            expected_intermediate_change="train_loss와 val_loss가 함께 내려가되 gap 증가는 작게 유지된다.",
            primary_metric="final_val_loss",
            guardrail_metric="final_generalization_gap, overfit_score, loss/과적합 그래프",
            interpretation_rule="val_loss 개선보다 gap/overfit 증가가 크면 학습량/용량 증가를 거부한다.",
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
            baseline_run=best.get("run_id"),
            changed_variable="seed",
            expected_intermediate_change="평균 성능보다 분산과 gap 안정성이 드러난다.",
            primary_metric="final_val_loss의 seed 분산 범위",
            guardrail_metric="final_generalization_gap, overfit_score, loss/과적합 그래프",
            interpretation_rule="seed만 바꾼 run이 gap을 키우면 설정 문제가 아니라 seed variance로 분류한다.",
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
        baseline_run=best.get("run_id"),
        changed_variable="activation_name",
        expected_intermediate_change="train/val loss 형태는 비슷하되 gradient 비선형성 때문에 val_loss 또는 gap이 소폭 바뀐다.",
        primary_metric="final_val_loss",
        guardrail_metric="overfit_score, loss/과적합 그래프",
        interpretation_rule="activation 변경은 parameter_count가 같을 때만 직접 해석하고, overfit_score가 커지면 채택하지 않는다.",
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
        artifact_dir_text = updated.get("artifact_dir") or ""
        result = load_artifact_result(ROOT / artifact_dir_text) if artifact_dir_text else None
        if result:
            for key in LEADERBOARD_FIELDS:
                if updated.get(key) in (None, "") and result.get(key) not in (None, ""):
                    updated[key] = result.get(key)

        if updated.get("epochs") not in (None, "") and updated.get("steps_per_epoch") not in (None, ""):
            updated["changed_variables"] = epochize_changed_variables(updated.get("changed_variables"), updated.get("epochs"), updated.get("max_steps"))
            backfill_tokenizer_scale(updated)
            backfilled.append(updated)
            continue

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
        backfill_tokenizer_scale(updated)
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
        "actual_vocab_size": result.get("actual_vocab_size"),
        "bpe_merge_count": result.get("bpe_merge_count"),
        "corpus_char_count": result.get("corpus_char_count"),
        "corpus_sha256": result.get("corpus_sha256"),
        "train_char_count": result.get("train_char_count"),
        "val_char_count": result.get("val_char_count"),
        "train_token_count": result.get("train_token_count"),
        "val_token_count": result.get("val_token_count"),
        "train_tokens_per_char": result.get("train_tokens_per_char"),
        "val_tokens_per_char": result.get("val_tokens_per_char"),
        "train_chars_per_token": result.get("train_chars_per_token"),
        "val_chars_per_token": result.get("val_chars_per_token"),
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
        "final_train_nats_per_char": result.get("final_train_nats_per_char"),
        "final_val_nats_per_char": result.get("final_val_nats_per_char"),
        "final_train_bits_per_char": result.get("final_train_bits_per_char"),
        "final_val_bits_per_char": result.get("final_val_bits_per_char"),
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
    update_interpretability_artifacts(leaderboard_rows, summary_rows)
    return {
        "dashboard": DASHBOARD_PATH,
        "metrics_summary": METRICS_SUMMARY_PATH,
        "trend": TREND_SVG_PATH,
        "latest_run": latest_run_svg,
        "latest_run_global": LATEST_RUN_SVG_PATH,
        "effect_map": EFFECT_MAP_SVG_PATH,
        "correlation": CORRELATION_SVG_PATH,
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
                "actual_vocab_size": numeric_or_blank(row.get("actual_vocab_size")),
                "bpe_merge_count": numeric_or_blank(row.get("bpe_merge_count")),
                "corpus_char_count": numeric_or_blank(row.get("corpus_char_count")),
                "corpus_sha256": row.get("corpus_sha256", ""),
                "train_char_count": numeric_or_blank(row.get("train_char_count")),
                "val_char_count": numeric_or_blank(row.get("val_char_count")),
                "train_token_count": numeric_or_blank(row.get("train_token_count")),
                "val_token_count": numeric_or_blank(row.get("val_token_count")),
                "train_tokens_per_char": numeric_or_blank(row.get("train_tokens_per_char")),
                "val_tokens_per_char": numeric_or_blank(row.get("val_tokens_per_char")),
                "train_chars_per_token": numeric_or_blank(row.get("train_chars_per_token")),
                "val_chars_per_token": numeric_or_blank(row.get("val_chars_per_token")),
                "final_train_loss": numeric_or_blank(row.get("final_train_loss")),
                "final_val_loss": numeric_or_blank(row.get("final_val_loss")),
                "final_train_nats_per_char": numeric_or_blank(row.get("final_train_nats_per_char")),
                "final_val_nats_per_char": numeric_or_blank(row.get("final_val_nats_per_char")),
                "final_train_bits_per_char": numeric_or_blank(row.get("final_train_bits_per_char")),
                "final_val_bits_per_char": numeric_or_blank(row.get("final_val_bits_per_char")),
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


CONFIG_KEYS = [
    "seed",
    "vocab_size",
    "context_length",
    "stride",
    "batch_size",
    "epochs",
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
]

MATCH_VARIABLES = [
    "epochs",
    "stride",
    "seed",
    "activation_name",
    "ffn_dropout_position",
    "drop_rate",
    "weight_decay",
    "ffn_mult",
    "n_layers",
    "emb_dim",
    "attention_impl",
    "tie_embeddings",
    "init_std",
    "context_length",
    "vocab_size",
]

CORRELATION_FEATURES = [
    ("vocab_size", "tokenizer"),
    ("actual_vocab_size", "tokenizer"),
    ("bpe_merge_count", "tokenizer"),
    ("val_tokens_per_char", "tokenizer"),
    ("val_chars_per_token", "tokenizer"),
    ("train_token_count", "dataset_tokenized"),
    ("val_token_count", "dataset_tokenized"),
    ("train_char_count", "dataset"),
    ("val_char_count", "dataset"),
    ("corpus_char_count", "dataset"),
    ("context_length", "windowing"),
    ("stride", "windowing"),
    ("epochs", "training_length"),
    ("learning_rate", "optimization"),
    ("weight_decay", "regularization"),
    ("drop_rate", "regularization"),
    ("parameter_count", "model_capacity"),
    ("emb_dim", "model_capacity"),
    ("n_heads", "model_capacity"),
    ("n_layers", "model_capacity"),
    ("ffn_mult", "model_capacity"),
    ("tokens_per_sec", "hardware_runtime"),
]

CORRELATION_OUTCOMES = [
    "final_val_loss",
    "final_val_nats_per_char",
    "final_val_bits_per_char",
    "final_generalization_gap",
    "overfit_score",
    "tokens_per_sec",
]


def update_interpretability_artifacts(leaderboard_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    records = hydrate_leaderboard_records(leaderboard_rows)
    effects = derive_matched_effects(records)
    hy_rows = parse_hy_testresult_rows()
    correlation_rows = build_correlation_records(records, hy_rows)
    correlations = compute_correlation_results(correlation_rows)
    EFFECT_MAP_SVG_PATH.write_text(render_effect_map_svg(effects), encoding="utf-8")
    CORRELATION_SVG_PATH.write_text(render_correlation_svg(correlations), encoding="utf-8")
    write_effect_map(effects)
    write_tokenizer_profile(records, hy_rows)
    write_correlation_report(correlations, correlation_rows)
    write_research_questions(effects, records, hy_rows, summary_rows)


def hydrate_leaderboard_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        artifact_dir_text = record.get("artifact_dir")
        if artifact_dir_text:
            artifact_result = load_artifact_result(ROOT / artifact_dir_text)
            if artifact_result:
                record.update({key: value for key, value in artifact_result.items() if value not in (None, "")})
        backfill_tokenizer_scale(record)
        records.append(record)
    return records


def backfill_tokenizer_scale(record: dict[str, Any]) -> None:
    """Infer char-normalized tokenizer metrics for legacy results when possible."""
    train_tokens = parse_float(record.get("train_token_count"))
    val_tokens = parse_float(record.get("val_token_count"))
    train_chars = parse_float(record.get("train_char_count"))
    val_chars = parse_float(record.get("val_char_count"))

    if (train_chars <= 0 or val_chars <= 0) and (train_tokens > 0 or val_tokens > 0):
        train_ratio = parse_float(record.get("train_ratio"), 0.9)
        inferred_train_chars, inferred_val_chars = default_split_char_counts(train_ratio)
        if train_chars <= 0 and inferred_train_chars > 0:
            record["train_char_count"] = inferred_train_chars
            train_chars = float(inferred_train_chars)
        if val_chars <= 0 and inferred_val_chars > 0:
            record["val_char_count"] = inferred_val_chars
            val_chars = float(inferred_val_chars)
        record.setdefault("tokenizer_scale_source", "inferred_default_corpus_split")
        if record.get("corpus_char_count") in (None, ""):
            record["corpus_char_count"] = inferred_train_chars + inferred_val_chars
        if record.get("corpus_sha256") in (None, ""):
            record["corpus_sha256"] = "legacy_default_20k"

    if record.get("corpus_char_count") in (None, "") and (train_chars > 0 or val_chars > 0):
        record["corpus_char_count"] = int(train_chars + val_chars)
    if record.get("corpus_sha256") in (None, "") and record.get("corpus_char_count") not in (None, ""):
        record["corpus_sha256"] = "legacy_default_20k"

    if train_tokens > 0 and train_chars > 0:
        record["train_tokens_per_char"] = round(train_tokens / train_chars, 6)
        record["train_chars_per_token"] = round(train_chars / train_tokens, 6)
    if val_tokens > 0 and val_chars > 0:
        record["val_tokens_per_char"] = round(val_tokens / val_chars, 6)
        record["val_chars_per_token"] = round(val_chars / val_tokens, 6)

    train_tokens_per_char = parse_float(record.get("train_tokens_per_char"))
    val_tokens_per_char = parse_float(record.get("val_tokens_per_char"))
    final_train_loss = parse_float(record.get("final_train_loss"))
    final_val_loss = parse_float(record.get("final_val_loss"))
    if final_train_loss > 0 and train_tokens_per_char > 0:
        record["final_train_nats_per_char"] = round(final_train_loss * train_tokens_per_char, 6)
        record["final_train_bits_per_char"] = round(final_train_loss * train_tokens_per_char / 0.6931471805599453, 6)
    if final_val_loss > 0 and val_tokens_per_char > 0:
        record["final_val_nats_per_char"] = round(final_val_loss * val_tokens_per_char, 6)
        record["final_val_bits_per_char"] = round(final_val_loss * val_tokens_per_char / 0.6931471805599453, 6)


def default_split_char_counts(train_ratio: float) -> tuple[int, int]:
    normalized_ratio = round(train_ratio, 6)
    if normalized_ratio not in _DEFAULT_SPLIT_CHAR_COUNT_CACHE:
        corpus = load_corpus(default_corpus_path(), DEFAULT_CHAR_LIMIT)
        train_text, val_text = _split_corpus_text(corpus, train_ratio=train_ratio)
        _DEFAULT_SPLIT_CHAR_COUNT_CACHE[normalized_ratio] = (len(train_text), len(val_text))
    return _DEFAULT_SPLIT_CHAR_COUNT_CACHE[normalized_ratio]


def derive_matched_effects(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    for variable in MATCH_VARIABLES:
        groups: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = {}
        for record in records:
            if record.get(variable) in (None, ""):
                continue
            signature = config_signature_for_effect(record, variable)
            groups.setdefault(signature, []).append(record)

        for group_records in groups.values():
            unique_values = {normalized_config_value(record.get(variable)) for record in group_records}
            if len(group_records) < 2 or len(unique_values) < 2:
                continue
            ordered = sorted(group_records, key=lambda record: (sort_value(record.get(variable)), parse_int(record.get("run_id"), 0)))
            for before, after in zip(ordered, ordered[1:]):
                if normalized_config_value(before.get(variable)) == normalized_config_value(after.get(variable)):
                    continue
                effect = matched_effect(variable, before, after)
                if effect:
                    effects.append(effect)

    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for effect in effects:
        key = (effect["axis"], str(effect["baseline_run"]), str(effect["variant_run"]))
        deduped[key] = effect
    return sorted(deduped.values(), key=lambda effect: (MATCH_VARIABLES.index(effect["axis"]) if effect["axis"] in MATCH_VARIABLES else 99, parse_int(effect["variant_run"], 0)))


def config_signature_for_effect(record: dict[str, Any], variable: str) -> tuple[tuple[str, str], ...]:
    ignored = {variable}
    if variable == "epochs":
        ignored.update({"max_steps", "steps_per_epoch"})
    if variable == "stride":
        ignored.update({"epochs", "steps_per_epoch"})
    if variable == "seed":
        ignored.add("seed")
    return tuple((key, normalized_config_value(record.get(key))) for key in CONFIG_KEYS if key not in ignored)


def matched_effect(variable: str, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any] | None:
    before_val = parse_float(before.get("final_val_loss"), None)
    after_val = parse_float(after.get("final_val_loss"), None)
    if before_val is None or after_val is None:
        return None
    before_gap = parse_float(before.get("final_generalization_gap"))
    after_gap = parse_float(after.get("final_generalization_gap"))
    before_overfit = parse_float(before.get("overfit_score"))
    after_overfit = parse_float(after.get("overfit_score"))
    delta_val = after_val - before_val
    delta_gap = after_gap - before_gap
    delta_overfit = after_overfit - before_overfit
    return {
        "axis": variable,
        "baseline_run": before.get("run_id"),
        "variant_run": after.get("run_id"),
        "from": normalized_config_value(before.get(variable)),
        "to": normalized_config_value(after.get(variable)),
        "delta_val_loss": round(delta_val, 6),
        "delta_gap": round(delta_gap, 6),
        "delta_overfit": round(delta_overfit, 6),
        "delta_nats_per_char": optional_delta(after.get("final_val_nats_per_char"), before.get("final_val_nats_per_char")),
        "interpretation": interpret_effect_delta(delta_val, delta_gap, delta_overfit),
        "confidence": effect_confidence(variable, before, after),
    }


def interpret_effect_delta(delta_val: float, delta_gap: float, delta_overfit: float) -> str:
    if delta_val < -0.001 and delta_overfit <= 0 and delta_gap <= 0:
        return "loss와 과적합이 함께 개선되어 채택 후보"
    if delta_val < -0.001 and delta_overfit > 0:
        return "loss는 개선됐지만 과적합 비용이 증가"
    if delta_val > 0.001 and delta_overfit < 0:
        return "regularization trade-off: loss 비용을 내고 과적합 완화"
    if abs(delta_val) <= 0.001 and abs(delta_overfit) <= 0.01:
        return "효과가 작아 seed 반복 전에는 결론 보류"
    if delta_val > 0.001 and delta_overfit >= 0:
        return "loss와 과적합이 모두 악화되어 제외 후보"
    return "혼합 신호: primary metric과 guardrail을 함께 판단"


def effect_confidence(variable: str, before: dict[str, Any], after: dict[str, Any]) -> str:
    if variable in {"seed"}:
        return "variance-check"
    if normalized_config_value(before.get("seed")) == normalized_config_value(after.get("seed")):
        return "matched"
    return "partial"


def optional_delta(after_value: Any, before_value: Any) -> str:
    if after_value in (None, "") or before_value in (None, ""):
        return ""
    return str(round(parse_float(after_value) - parse_float(before_value), 6))


def normalized_config_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, bool):
        return str(value)
    text = str(value)
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered
    try:
        number = float(text)
    except ValueError:
        return text
    return str(round(number, 6))


def sort_value(value: Any) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def write_effect_map(effects: list[dict[str, Any]]) -> None:
    rows = "\n".join(
        f"| {effect['axis']} | {effect['baseline_run']} -> {effect['variant_run']} | {effect['from']} -> {effect['to']} | {effect['delta_val_loss']} | {effect['delta_gap']} | {effect['delta_overfit']} | {effect['delta_nats_per_char']} | {effect['interpretation']} | {effect['confidence']} |"
        for effect in effects[:36]
    )
    content = f"""# 변수 효과 지도

이 문서는 전체 run을 점수 순으로만 보지 않고, 가능한 matched-pair 비교로 다시 읽기 위한 지도입니다.

중요 원칙:

- loss만 단독으로 좋아진 결과는 채택하지 않는다.
- loss와 함께 `final_generalization_gap`, `train_val_improvement_gap`, `overfit_score` 그래프를 확인한다.
- tokenizer/vocab이 달라진 실험은 token-level loss가 아니라 `nats_per_char` 또는 `bits_per_char`를 같이 본다.
- 한 번에 여러 변수가 바뀐 run은 인과 해석 신뢰도를 낮게 둔다.

![짝비교 변수 효과 그래프](visuals/effect_map_pairs.svg)

## 짝비교 변화량

`delta`는 variant run - baseline run입니다. `delta_val_loss < 0`이면 validation loss 개선이고, `delta_overfit > 0`이면 과적합 비용 증가입니다.

| 변수축 | 비교 run | 변경 | 검증 손실 변화 | 일반화 gap 변화 | 과적합 점수 변화 | 문자당 nats 변화 | 해석 | 신뢰도 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
{rows}

## 읽는 방법

- `loss 개선 + overfit 감소`: 바로 채택 후보입니다.
- `loss 개선 + overfit 증가`: 성능은 좋아졌지만 외우는 비용을 냈다는 뜻입니다. fresh seed 또는 더 짧은 epoch로 확인합니다.
- `loss 악화 + overfit 감소`: regularization이 너무 강하거나 학습이 부족할 수 있습니다.
- `vocab_size`, BPE merge, dataset/tokenization이 바뀐 경우: 이 표의 token loss delta만으로 결론내리지 말고 `tokenizer_profile.md`를 먼저 봅니다.
"""
    EFFECT_MAP_PATH.write_text(content, encoding="utf-8")


def parse_hy_testresult_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not HY_TESTRESULT_DIR.exists():
        return rows
    for path in sorted(HY_TESTRESULT_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        row: dict[str, Any] = {"file": str(path.relative_to(ROOT)), "title": path.stem}
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if title_match:
            row["title"] = title_match.group(1)
        for line in text.splitlines():
            match = re.match(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$", line)
            if not match:
                continue
            key = match.group(1).strip()
            value = match.group(2).strip()
            row[key] = value
        val_tokens = parse_float(row.get("val_tokens"))
        val_chars = parse_float(row.get("val_chars"))
        val_loss = parse_float(row.get("final_val_loss"))
        if val_tokens > 0 and val_chars > 0:
            row["val_tokens_per_char"] = round(val_tokens / val_chars, 6)
            row["val_chars_per_token"] = round(val_chars / val_tokens, 6)
            row["final_val_nats_per_char"] = round(val_loss * val_tokens / val_chars, 6)
            row["final_val_bits_per_char"] = round(val_loss * val_tokens / val_chars / 0.6931471805599453, 6)
        rows.append(row)
    return rows


def write_tokenizer_profile(records: list[dict[str, Any]], hy_rows: list[dict[str, Any]]) -> None:
    recent_records = [record for record in records if record.get("val_token_count") not in (None, "")]
    recent_records = sorted(recent_records, key=lambda record: parse_int(record.get("run_id"), 0))[-12:]
    auto_rows = "\n".join(
        f"| {record.get('run_id')} | {record.get('vocab_size')} | {record.get('actual_vocab_size', '')} | {record.get('bpe_merge_count', '')} | {record.get('val_token_count', '')} | {round(parse_float(record.get('val_tokens_per_char')), 6)} | {round(parse_float(record.get('val_chars_per_token')), 6)} | {round(parse_float(record.get('final_val_nats_per_char')), 6)} | {round(parse_float(record.get('final_val_bits_per_char')), 6)} | {record.get('tokenizer_scale_source', 'recorded')} |"
        for record in recent_records
    )
    hy_vocab_rows = [row for row in hy_rows if row.get("vocab_size") not in (None, "")]
    hy_table = "\n".join(
        f"| {row.get('title')} | {row.get('vocab_size', '')} | {row.get('train_tokens', '')} | {row.get('val_tokens', '')} | {row.get('val_tokens_per_char', '')} | {row.get('val_chars_per_token', '')} | {row.get('final_val_loss', '')} | {row.get('final_val_nats_per_char', '')} | {row.get('final_val_bits_per_char', '')} |"
        for row in hy_vocab_rows
    )
    content = f"""# 토크나이저 / BPE / 데이터셋 프로필

이 문서는 vocab size, BPE merge, dataset tokenization이 loss 해석에 미치는 영향을 분리하기 위한 자료입니다.

핵심:

- token-level `final_val_loss`는 vocab/tokenizer가 같을 때만 직접 비교합니다.
- vocab size가 바뀌면 한 token이 담당하는 문자 수가 바뀌므로 `final_val_nats_per_char`와 `final_val_bits_per_char`를 함께 봅니다.
- `vocab_size`를 키우면 보통 `tokens_per_char`는 줄지만, LM head/embedding parameter가 늘고 token class 난이도도 바뀝니다.
- 따라서 vocab/BPE 실험은 model hyperparameter 실험과 같은 leaderboard에서 단순 순위 비교하지 않습니다.

## 자동 루프 최근 토크나이저 상태

| run | 요청 vocab | 실제 vocab | BPE merge 수 | 검증 token 수 | 검증 token/문자 | 검증 문자/token | 검증 nats/문자 | 검증 bits/문자 | 출처 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{auto_rows}

## docs/HY/testresult 토크나이저 관련 관찰

이 표는 수동 E01-E20 결과에서 tokenization 관련 수치를 다시 계산한 것입니다. 모델 크기와 context도 섞여 있으므로, 여기서는 “loss scale이 왜 달라지는가”를 보는 용도로 사용합니다.

| 실험 | vocab_size | 학습 token 수 | 검증 token 수 | 검증 token/문자 | 검증 문자/token | token 기준 검증 loss | 검증 nats/문자 | 검증 bits/문자 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{hy_table}

## 해석 규칙

1. 같은 tokenizer/vocab이면 `final_val_loss`, `gap`, `overfit_score`를 직접 비교한다.
2. tokenizer/vocab이 다르면 `final_val_loss` 단독 비교를 금지한다.
3. vocab/BPE 실험의 primary metric은 `final_val_nats_per_char` 또는 생성 샘플 품질이다.
4. vocab/BPE 변경은 parameter_count와 steps_per_epoch도 함께 바꾸므로 model 실험과 분리한다.
"""
    TOKENIZER_PROFILE_PATH.write_text(content, encoding="utf-8")


def write_research_questions(effects: list[dict[str, Any]], records: list[dict[str, Any]], hy_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    best = min(summary_rows, key=lambda row: parse_float(row.get("selection_score"), 1e9)) if summary_rows else {}
    latest = summary_rows[-1] if summary_rows else {}
    content = f"""# 해석 가능한 다음 연구 질문

현재 목표는 새 run을 많이 쌓는 것이 아니라, 변수를 올리면 무엇이 좋아지고 무엇이 나빠지는지 설명 가능한 지도를 만드는 것입니다.

## 현재 기준점

- overfit-aware best: run `{best.get("run_id", "")}` / val `{best.get("final_val_loss", "")}` / overfit `{best.get("overfit_score", "")}`
- latest: run `{latest.get("run_id", "")}` / val `{latest.get("final_val_loss", "")}` / overfit `{latest.get("overfit_score", "")}`
- 모든 새 결과는 loss 그래프와 overfit 그래프가 함께 있어야 유효합니다.

## 우선순위 질문

1. BPE/tokenizer 상태와 dataset 상태는 loss scale과 학습 내용에 어떤 상관을 갖는가?
   - 짝비교 단계: dataset 고정 후 vocab/BPE만 변경, 그 다음 tokenizer 고정 후 dataset만 변경
   - 주 지표: `final_val_nats_per_char`, 생성 샘플 품질
   - 보호 지표: `final_generalization_gap`, `overfit_score`, loss/overfit 그래프

2. `epochs` 2.56 -> 2.69는 low-risk seed에서 raw validation을 개선하지만, overfit 비용을 얼마나 반복적으로 키우는가?
   - 짝비교: 같은 seed, 같은 tokenizer, 같은 model, epochs만 변경
   - 주 지표: `final_val_loss`
   - 보호 지표: `final_generalization_gap`, `overfit_score`

3. `stride` 24 -> 20은 high-gap seed rescue인가, 아니면 기본 정책으로도 유리한가?
   - 짝비교: 같은 seed와 실제 업데이트 수, stride만 변경
   - 주 지표: `overfit_score`
   - 보호 지표: `final_val_loss`

4. vocab/BPE merge는 token loss 착시를 얼마나 만드는가?
   - 짝비교 단계: vocab만 변경, model/training 고정
   - 주 지표: `final_val_nats_per_char`, `final_val_bits_per_char`
   - 보호 지표: 생성 샘플, parameter_count, steps_per_epoch

5. capacity를 올리면 무엇이 좋아지는가?
   - 짝비교: `emb_dim`, `n_layers`, `ffn_mult` 중 하나만 변경
   - 주 지표: `final_val_loss`
   - 보호 지표: `parameter_count`, `tokens_per_sec`, `overfit_score`

## 금지할 해석

- vocab_size가 다른 run끼리 token-level `final_val_loss`만 비교하지 않는다.
- loss가 낮아졌는데 overfit 그래프가 없으면 결론으로 쓰지 않는다.
- seed와 hyperparameter가 동시에 바뀐 run은 인과 결론으로 쓰지 않는다.
- 한 번에 여러 변수를 바꾼 run은 rescue 후보로만 보고 effect map의 핵심 근거로 쓰지 않는다.

## 다음 실험 제안 형식

```json
{{
  "baseline_run": 112,
  "changed_variable": "seed",
  "expected_intermediate_change": "2.69 epoch 후보의 seed variance를 확인한다.",
  "primary_metric": "low-risk 범위 안의 final_val_loss",
  "guardrail_metric": "final_generalization_gap과 overfit_score",
  "interpretation_rule": "fresh seed에서도 low-risk면 2.69 epochs를 후보로 유지하고, gap이 커지면 2.56 epochs를 기본값으로 둔다."
}}
```
"""
    RESEARCH_QUESTIONS_PATH.write_text(content, encoding="utf-8")


def build_correlation_records(records: list[dict[str, Any]], hy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    correlation_records: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        row["scope"] = "auto_loop"
        row["experiment_id"] = f"run_{record.get('run_id')}"
        row["dataset_signature"] = record.get("corpus_sha256") or "unknown_auto_dataset"
        row["overfit_score_source"] = "computed"
        correlation_records.append(row)
    for row in hy_rows:
        correlation_records.append(normalize_hy_correlation_record(row))
    return correlation_records


def normalize_hy_correlation_record(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "scope": "hy_testresult",
        "experiment_id": row.get("title", row.get("file", "")),
        "dataset_signature": f"hy_chars_{row.get('train_chars', '')}_{row.get('val_chars', '')}",
        "overfit_score_source": "gap_proxy",
    }
    key_map = {
        "vocab_size": "vocab_size",
        "actual_vocab_size": "vocab_size",
        "context_length": "context_length",
        "stride": "stride",
        "batch_size": "batch_size",
        "epochs": "num_epochs",
        "learning_rate": "lr",
        "weight_decay": "weight_decay",
        "emb_dim": "emb_dim",
        "n_heads": "n_heads",
        "n_layers": "n_layers",
        "drop_rate": "drop_rate",
        "qkv_bias": "qkv_bias",
        "ffn_mult": "ffn_mult",
        "tie_embeddings": "weight_tying",
        "parameter_count": "parameter_count",
        "train_char_count": "train_chars",
        "val_char_count": "val_chars",
        "train_token_count": "train_tokens",
        "val_token_count": "val_tokens",
        "corpus_char_count": "corpus_char_count",
        "final_train_loss": "final_train_loss",
        "final_val_loss": "final_val_loss",
        "tokens_per_sec": "tokens_per_sec",
    }
    for target_key, source_key in key_map.items():
        value = row.get(source_key)
        if target_key == "corpus_char_count":
            train_chars = number_or_none(row.get("train_chars")) or 0.0
            val_chars = number_or_none(row.get("val_chars")) or 0.0
            value = train_chars + val_chars if train_chars or val_chars else ""
        parsed = number_or_none(value)
        if parsed is not None:
            normalized[target_key] = parsed
    train_tokens = number_or_none(normalized.get("train_token_count"))
    val_tokens = number_or_none(normalized.get("val_token_count"))
    train_chars = number_or_none(normalized.get("train_char_count"))
    val_chars = number_or_none(normalized.get("val_char_count"))
    train_loss = number_or_none(normalized.get("final_train_loss"))
    val_loss = number_or_none(normalized.get("final_val_loss"))
    if train_tokens and train_chars:
        normalized["train_tokens_per_char"] = train_tokens / train_chars
        normalized["train_chars_per_token"] = train_chars / train_tokens
    if val_tokens and val_chars:
        normalized["val_tokens_per_char"] = val_tokens / val_chars
        normalized["val_chars_per_token"] = val_chars / val_tokens
    if train_loss is not None and val_loss is not None:
        gap = val_loss - train_loss
        normalized["final_generalization_gap"] = gap
        normalized["overfit_score"] = max(0.0, gap)
    if train_loss is not None and normalized.get("train_tokens_per_char"):
        normalized["final_train_nats_per_char"] = train_loss * normalized["train_tokens_per_char"]
        normalized["final_train_bits_per_char"] = normalized["final_train_nats_per_char"] / 0.6931471805599453
    if val_loss is not None and normalized.get("val_tokens_per_char"):
        normalized["final_val_nats_per_char"] = val_loss * normalized["val_tokens_per_char"]
        normalized["final_val_bits_per_char"] = normalized["final_val_nats_per_char"] / 0.6931471805599453
    return normalized


def compute_correlation_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    scopes = sorted({str(row.get("scope")) for row in rows if row.get("scope")})
    for scope in scopes:
        scoped_rows = [row for row in rows if row.get("scope") == scope]
        for feature, family in CORRELATION_FEATURES:
            for outcome in CORRELATION_OUTCOMES:
                if feature == outcome:
                    continue
                pairs = numeric_pairs(scoped_rows, feature, outcome)
                if len(pairs) < 3:
                    continue
                xs = [pair[0] for pair in pairs]
                ys = [pair[1] for pair in pairs]
                if len(set(xs)) < 2 or len(set(ys)) < 2:
                    continue
                pearson = pearson_correlation(xs, ys)
                spearman = spearman_correlation(xs, ys)
                if pearson is None or spearman is None:
                    continue
                results.append(
                    {
                        "scope": scope,
                        "family": family,
                        "feature": feature,
                        "outcome": outcome,
                        "n": len(pairs),
                        "pearson": round(pearson, 4),
                        "spearman": round(spearman, 4),
                        "interpretation": interpret_correlation_strength(pearson, len(pairs)),
                    }
                )
    return sorted(results, key=lambda result: (result["scope"], -abs(parse_float(result["pearson"])), result["feature"], result["outcome"]))


def numeric_pairs(rows: list[dict[str, Any]], x_key: str, y_key: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        x_value = number_or_none(row.get(x_key))
        y_value = number_or_none(row.get(y_key))
        if x_value is not None and y_value is not None:
            pairs.append((x_value, y_value))
    return pairs


def number_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return 1.0
    if lowered in {"false", "no"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / math.sqrt(denom_x * denom_y)


def spearman_correlation(xs: list[float], ys: list[float]) -> float | None:
    return pearson_correlation(rank_values(xs), rank_values(ys))


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor
        while end + 1 < len(indexed) and indexed[end + 1][0] == indexed[cursor][0]:
            end += 1
        average_rank = (cursor + end + 2) / 2.0
        for _, original_index in indexed[cursor : end + 1]:
            ranks[original_index] = average_rank
        cursor = end + 1
    return ranks


def interpret_correlation_strength(value: float, n: int) -> str:
    magnitude = abs(value)
    if n < 5:
        return "표본 부족: 방향 신호만 참고"
    if magnitude >= 0.75:
        return "강한 관찰 상관: 통제 반복 우선 후보"
    if magnitude >= 0.45:
        return "중간 관찰 상관: confound 확인 필요"
    return "약한 관찰 상관 또는 noise"


def write_correlation_report(correlations: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    top_rows = "\n".join(
        f"| {item['scope']} | {item['family']} | {item['feature']} -> {item['outcome']} | {item['n']} | {item['pearson']} | {item['spearman']} | {item['interpretation']} |"
        for item in sorted(correlations, key=lambda item: -abs(parse_float(item.get("pearson"))))[:42]
    )
    scope_rows = "\n".join(
        f"| {summary['scope']} | {summary['rows']} | {summary['datasets']} | {summary['vocab_values']} | {summary['tokenizer_scales']} | {summary['overfit_metric']} |"
        for summary in summarize_correlation_scopes(rows)
    )
    content = f"""# 상관 증거 보고서

이 문서는 BPE/tokenizer 상태, dataset 상태, model/training 조건이 loss와 과적합에 어떤 상관을 갖는지 보기 위한 관찰 보고서입니다.

중요한 경계:

- 아래 상관계수는 현재 로그에서 나온 관찰 상관입니다. 인과 증명으로 바로 쓰지 않습니다.
- 자동 loop와 `docs/HY/testresult`는 dataset, 하드웨어, 실험 스케일이 다르므로 섞어서 계산하지 않습니다.
- tokenizer 또는 dataset이 달라지는 비교는 token-level loss보다 `final_val_nats_per_char`, `final_val_bits_per_char`, gap/overfit을 우선합니다.
- loss 그래프와 overfit 그래프가 없는 결과는 상관 주장 근거로 쓰지 않습니다.

![상관 증거 그래프](visuals/correlation_evidence.svg)

## 범위별 자료 구성

| 범위 | 행 수 | dataset 그룹 수 | vocab 값 수 | tokenizer scale 수 | 과적합 지표 |
| --- | ---: | ---: | ---: | ---: | --- |
{scope_rows}

## 가장 강한 관찰 상관

| 범위 | 변수군 | 관계 | n | 피어슨 r | 스피어만 rho | 해석 |
| --- | --- | --- | ---: | ---: | ---: | --- |
{top_rows}

## 어떻게 상관을 증명에 가깝게 만들 것인가

| 주장 | 통제 설계 | 성공 근거 |
| --- | --- | --- |
| BPE 상태가 학습 내용을 바꾼다 | dataset, model, optimizer, epochs, seed set을 고정하고 vocab/BPE만 변경한다 | `val_tokens_per_char` 변화와 `final_val_nats_per_char`, 생성 샘플, gap/overfit 변화가 같은 방향으로 반복된다 |
| dataset 상태가 학습 내용을 바꾼다 | tokenizer를 고정한 뒤 dataset만 바꾼다 | tokenization scale이 고정된 상태에서 loss/overfit/generation 변화가 dataset별로 반복된다 |
| dataset과 tokenizer가 상호작용한다 | 2x2 factorial: dataset A/B x tokenizer A/B | tokenizer가 자신의 dataset에서만 유리하거나, cross-tokenizer에서 gap이 증가하는 interaction이 반복된다 |
| model capacity와 tokenizer 효과가 섞인다 | vocab sweep 중 parameter_count를 기록하고 capacity sweep은 별도 phase로 분리한다 | vocab 효과가 `parameter_count` 효과와 분리되어 `nats_per_char`에서 남는다 |

## 다음 실험 설계 규칙

1. correlation claim 하나당 독립변수 하나만 움직인다.
2. 최소 3 seed 반복 후 평균과 분산을 같이 본다.
3. 모든 결과는 train/val loss curve와 gap/overfit curve를 포함한다.
4. vocab/BPE 비교는 token loss가 아니라 char-normalized loss를 primary metric으로 둔다.
5. dataset 변경 실험은 corpus hash, train/val char count, train/val token count를 필수 기록한다.
"""
    CORRELATION_REPORT_PATH.write_text(content, encoding="utf-8")


def summarize_correlation_scopes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for scope in sorted({str(row.get("scope")) for row in rows if row.get("scope")}):
        scoped = [row for row in rows if row.get("scope") == scope]
        datasets = {str(row.get("dataset_signature") or row.get("corpus_sha256") or "") for row in scoped}
        vocab_values = {number_or_none(row.get("vocab_size")) for row in scoped if number_or_none(row.get("vocab_size")) is not None}
        tokenizer_scales = {round(number_or_none(row.get("val_tokens_per_char")) or 0.0, 6) for row in scoped if number_or_none(row.get("val_tokens_per_char")) is not None}
        overfit_sources = {str(row.get("overfit_score_source") or "computed") for row in scoped}
        summaries.append(
            {
                "scope": scope,
                "rows": len(scoped),
                "datasets": len(datasets),
                "vocab_values": len(vocab_values),
                "tokenizer_scales": len(tokenizer_scales),
                "overfit_metric": ", ".join(sorted(overfit_sources)),
            }
        )
    return summaries


def render_correlation_svg(correlations: list[dict[str, Any]]) -> str:
    shown = [
        item
        for item in sorted(correlations, key=lambda item: -abs(parse_float(item.get("pearson"))))
        if item.get("outcome") in {"final_val_nats_per_char", "final_generalization_gap", "overfit_score"}
    ][:14]
    width = 1080
    height = max(360, 122 + len(shown) * 34)
    parts = [
        svg_header(width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(34, 42, "토크나이저 / 데이터셋 상관 증거", size=23, weight="700"),
        svg_text(34, 68, "현재 값은 관찰 상관입니다. 인과 주장은 통제 반복이 필요합니다.", size=13, fill="#475569"),
        svg_text(500, 100, "피어슨 r", size=12, fill="#334155", weight="700"),
    ]
    if not shown:
        parts.append(svg_text(34, 150, "상관 막대를 그릴 만큼 다양한 기록이 아직 부족합니다.", size=14, fill="#64748b"))
        parts.append("</svg>")
        return "\n".join(parts)
    for index, item in enumerate(shown):
        y = 132 + index * 34
        label = f"{item['scope']} / {item['feature']} -> {item['outcome']} (n={item['n']})"
        parts.append(svg_text(34, y + 5, label, size=12, fill="#0f172a"))
        parts.extend(render_delta_bar(parse_float(item["pearson"]), 520, y - 10, 260, 1.0, good_when_negative=False))
        parts.append(svg_text(815, y + 5, f"rho={parse_float(item['spearman']):+.2f}", size=11, fill="#64748b"))
    parts.append("</svg>")
    return "\n".join(parts)


def render_effect_map_svg(effects: list[dict[str, Any]]) -> str:
    shown = effects[:14]
    width = 980
    height = max(360, 118 + len(shown) * 32)
    parts = [
        svg_header(width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(34, 42, "짝비교 변수 효과 지도", size=23, weight="700"),
        svg_text(34, 68, "검증 손실 변화와 과적합 점수 변화를 반드시 함께 읽습니다.", size=13, fill="#475569"),
        svg_text(280, 98, "검증 손실 변화", size=12, fill="#334155", weight="700"),
        svg_text(610, 98, "과적합 점수 변화", size=12, fill="#334155", weight="700"),
    ]
    val_deltas = [abs(parse_float(effect.get("delta_val_loss"))) for effect in shown]
    overfit_deltas = [abs(parse_float(effect.get("delta_overfit"))) for effect in shown]
    val_scale = max(val_deltas + [0.01])
    overfit_scale = max(overfit_deltas + [0.03])
    for index, effect in enumerate(shown):
        y = 128 + index * 32
        label = f"{effect['axis']} {effect['baseline_run']}->{effect['variant_run']}"
        parts.append(svg_text(34, y + 5, label, size=12, fill="#0f172a"))
        parts.extend(render_delta_bar(parse_float(effect["delta_val_loss"]), 300, y - 10, 230, val_scale, good_when_negative=True))
        parts.extend(render_delta_bar(parse_float(effect["delta_overfit"]), 630, y - 10, 230, overfit_scale, good_when_negative=True))
        parts.append(svg_text(875, y + 5, str(effect["confidence"]), size=11, fill="#64748b"))
    parts.append("</svg>")
    return "\n".join(parts)


def render_delta_bar(value: float, x: int, y: int, width: int, scale: float, *, good_when_negative: bool) -> list[str]:
    center = x + width / 2
    bar_max = width / 2 - 8
    bar_w = 0 if scale == 0 else min(bar_max, abs(value) / scale * bar_max)
    color = "#16a34a" if (value < 0 and good_when_negative) else "#dc2626" if value > 0 else "#64748b"
    x0 = center - bar_w if value < 0 else center
    return [
        f'<line x1="{center:.2f}" y1="{y - 4}" x2="{center:.2f}" y2="{y + 18}" stroke="#94a3b8"/>',
        f'<rect x="{x0:.2f}" y="{y}" width="{bar_w:.2f}" height="14" rx="3" fill="{color}" opacity="0.85"/>',
        svg_text(x + width + 8, y + 12, f"{value:+.4f}", size=11, fill="#334155"),
    ]


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
        svg_text(36, 44, "mini GPT 학습 지표", size=24, weight="700"),
        svg_text(36, 70, "자동 실험 run 전체의 loss 추세와 과적합 신호", size=13, fill="#475569"),
    ]
    parts.extend(render_line_panel(summary_rows, 58, 105, 860, 230, "학습 / 검증 손실", [("final_train_loss", "#2563eb", "학습"), ("final_val_loss", "#dc2626", "검증")], zero_floor=False))
    parts.extend(render_line_panel(summary_rows, 58, 405, 860, 210, "일반화 / 과적합", [("final_generalization_gap", "#d97706", "gap"), ("overfit_score", "#7c3aed", "과적합")], zero_floor=True, thresholds=[(0.05, "gap 위험"), (0.12, "점수 주의")]))
    parts.append("</svg>")
    return "\n".join(parts)


def render_latest_run_svg(result: dict[str, Any]) -> str:
    width, height = 820, 520
    loss_items = [
        ("초기 학습", parse_float(result.get("initial_train_loss"))),
        ("최종 학습", parse_float(result.get("final_train_loss"))),
        ("초기 검증", parse_float(result.get("initial_val_loss"))),
        ("최종 검증", parse_float(result.get("final_val_loss"))),
    ]
    overfit_items = [
        ("최종 gap", parse_float(result.get("final_generalization_gap"))),
        ("gap 변화", parse_float(result.get("generalization_gap_delta"))),
        ("개선 불균형", parse_float(result.get("train_val_improvement_gap"))),
        ("과적합 점수", parse_float(result.get("overfit_score"))),
    ]
    status = str(result.get("fit_status") or "unknown")
    run_id = result.get("run_id", "?")
    epoch_text = f"epochs={numeric_or_blank(result.get('epochs'))} / steps={numeric_or_blank(result.get('max_steps'))}"
    parts = [
        svg_header(width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(34, 42, f"Run {run_id} 지표", size=23, weight="700"),
        svg_text(34, 68, f"상태={status} / {epoch_text} / 장치={result.get('device', '')} / 초당 token={numeric_or_blank(result.get('tokens_per_sec'))}", size=13, fill="#475569"),
    ]
    parts.extend(render_bar_panel(loss_items, 48, 108, 710, 170, "손실 요약", "#2563eb"))
    parts.extend(render_bar_panel(overfit_items, 48, 338, 710, 125, "과적합 신호", "#7c3aed", max_hint=0.18))
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

## 해석 원칙

- loss만 있는 결과는 결론으로 쓰지 않는다.
- 모든 채택 판단은 train/validation loss, generalization gap, overfit_score 그래프를 함께 본다.
- vocab/BPE/tokenizer가 달라진 비교는 token-level loss 대신 `nats_per_char`와 `bits_per_char`를 같이 본다.
- 한 번에 여러 변수가 바뀐 run은 후보 발견용이고, 인과 해석은 matched-pair 비교에서만 한다.

## 현재 요약

- 최신 run: `{latest.get("run_id", "")}` / 학습 상태=`{latest.get("fit_status", "")}` / 위험도=`{latest.get("risk_level", "")}`
- 최신 epochs: `{latest.get("epochs", "")}` / 실제 업데이트 수=`{latest.get("max_steps", "")}` / epoch당 step=`{latest.get("steps_per_epoch", "")}`
- 최신 검증 손실(`final_val_loss`): `{latest.get("final_val_loss", "")}`
- 최신 일반화 gap: `{latest.get("final_generalization_gap", "")}`
- 최신 과적합 점수(`overfit_score`): `{latest.get("overfit_score", "")}`
- 현재 best 후보: run `{best.get("run_id", "")}` / 선택 점수=`{best.get("selection_score", "")}` / 검증 손실=`{best.get("final_val_loss", "")}`

## 전체 추세

![loss와 과적합 추세](visuals/loss_overfit_trends.svg)

## 최신 run 상세

![최신 run 지표](visuals/latest_run_metrics.svg)

## 변수 효과 지도

![짝비교 변수 효과](visuals/effect_map_pairs.svg)

## 상관 증거

![상관 증거](visuals/correlation_evidence.svg)

## 최근 10회

| run | 학습 상태 | 위험도 | epochs | step 수 | 학습 손실 | 검증 손실 | gap | 과적합 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{table_rows}

## 파일

- `metrics_summary.csv`: 시각화와 해석을 위한 정규화된 지표
- `visuals/loss_overfit_trends.svg`: 전체 loss/gap/overfit 추세
- `visuals/latest_run_metrics.svg`: 최신 run의 loss와 과적합 신호
- `visuals/effect_map_pairs.svg`: 짝비교 기준 변수 효과 지도
- `visuals/correlation_evidence.svg`: tokenizer/dataset/model/training 축의 관찰 상관 그래프
- `effect_map.md`: 변수별 loss/과적합 trade-off 해석
- `tokenizer_profile.md`: BPE/vocab/dataset tokenization 해석
- `correlation_report.md`: BPE, dataset, model, training 축의 상관 증거와 통제 실험 설계
- `research_questions.md`: 다음 실험을 이해 가능한 질문으로 정리
"""
    DASHBOARD_PATH.write_text(content, encoding="utf-8")


def write_run_report(report_path: Path, plan: ExperimentPlan, result: dict[str, Any], hardware: dict[str, Any], corpus_path: Path, artifact_dir: Path, best_row: dict[str, Any] | None, visual_paths: dict[str, Path] | None = None) -> None:
    config = plan.config
    display_config = asdict(config)
    if display_config.get("epochs") is not None:
        display_config.pop("max_steps", None)
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

## 해석 가능한 비교 설계

| 항목 | 값 |
| --- | --- |
| 기준 run | {plan.baseline_run if plan.baseline_run not in (None, "") else "없음"} |
| 의도적으로 바꾼 변수 | {plan.changed_variable or "미지정"} |
| 기대하는 중간 변화 | {plan.expected_intermediate_change or "미지정"} |
| 주 지표 | {plan.primary_metric or "final_val_loss"} |
| 보호 지표 | {plan.guardrail_metric or "final_generalization_gap, overfit_score"} |
| 해석 규칙 | {plan.interpretation_rule or "loss 개선은 과적합 그래프와 함께 해석한다."} |
| 상관 목표 | {plan.correlation_target or "미지정"} |
| 통제 변수 | {", ".join(plan.control_variables) if plan.control_variables else "미지정"} |
| 증거 수준 | {plan.evidence_level or "observed_correlation"} |

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
{json.dumps(display_config, ensure_ascii=False, indent=2)}
```

## 학습 길이

| 항목 | 값 |
| --- | --- |
| epochs | {result.get("epochs")} |
| steps_per_epoch | {result.get("steps_per_epoch")} |
| 실제 업데이트 수 | {result.get("max_steps")} |

## 토크나이저 / 데이터셋 프로필

| 지표 | 값 |
| --- | --- |
| 요청 vocab_size | {result.get("vocab_size")} |
| actual_vocab_size | {result.get("actual_vocab_size")} |
| bpe_merge_count | {result.get("bpe_merge_count")} |
| corpus_char_count | {result.get("corpus_char_count")} |
| corpus_sha256 | {result.get("corpus_sha256")} |
| train_char_count | {result.get("train_char_count")} |
| val_char_count | {result.get("val_char_count")} |
| train_token_count | {result.get("train_token_count")} |
| val_token_count | {result.get("val_token_count")} |
| train_tokens_per_char | {result.get("train_tokens_per_char")} |
| val_tokens_per_char | {result.get("val_tokens_per_char")} |
| train_chars_per_token | {result.get("train_chars_per_token")} |
| val_chars_per_token | {result.get("val_chars_per_token")} |

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
| final_train_nats_per_char | {result.get("final_train_nats_per_char")} |
| final_val_nats_per_char | {result.get("final_val_nats_per_char")} |
| final_train_bits_per_char | {result.get("final_train_bits_per_char")} |
| final_val_bits_per_char | {result.get("final_val_bits_per_char")} |
| final_generalization_gap | {result.get("final_generalization_gap")} |
| generalization_gap_delta | {result.get("generalization_gap_delta")} |
| train_val_improvement_gap | {result.get("train_val_improvement_gap")} |
| overfit_score | {result.get("overfit_score")} |
| fit_status | {result.get("fit_status")} |
| epochs | {result.get("epochs")} |
| steps_per_epoch | {result.get("steps_per_epoch")} |
| 실제 업데이트 수 | {result.get("max_steps")} |
| parameter_count | {result.get("parameter_count")} |
| tokens_per_sec | {result.get("tokens_per_sec")} |
| elapsed_sec | {result.get("elapsed_sec")} |
| device | {result.get("device")} |

## 시각 지표 (필수)

이 보고서는 loss와 과적합 신호를 함께 보는 그래프가 있어야만 유효한 결과로 본다. token-level loss만 단독으로 해석하지 않는다.

![이번 run 지표]({artifact_dir.name}/{latest_visual})

![전체 loss/overfit 추세]({trend_visual})

- 대시보드: `{dashboard_path}`
- 지표 요약 CSV: `../metrics_summary.csv`
- effect map: `../effect_map.md`
- tokenizer profile: `../tokenizer_profile.md`
- 상관 보고서: `../correlation_report.md`

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
- 비교 설계: 기준 run={plan.baseline_run}, 변경 변수={plan.changed_variable}, 주 지표={plan.primary_metric}, 보호 지표={plan.guardrail_metric}
- 상관 검증: 목표={plan.correlation_target}, 통제 변수={plan.control_variables}, 증거 수준={plan.evidence_level}
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
    keys = [
        "epochs",
        "steps_per_epoch",
        "max_steps",
        "actual_vocab_size",
        "bpe_merge_count",
        "corpus_char_count",
        "corpus_sha256",
        "val_tokens_per_char",
        "val_chars_per_token",
        "final_train_loss",
        "final_val_loss",
        "final_val_nats_per_char",
        "final_val_bits_per_char",
        "final_generalization_gap",
        "overfit_score",
        "fit_status",
        "parameter_count",
        "tokens_per_sec",
        "device",
    ]
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
- `final_val_loss`가 낮을수록 좋지만, loss만 단독으로 채택 근거가 될 수 없다.
- `final_generalization_gap = final_val_loss - final_train_loss`가 커지면 과적합 위험이다.
- `overfit_score`는 낮을수록 좋다.
- 모든 결론은 train/validation loss 그래프와 gap/overfit 그래프가 함께 있을 때만 유효하다.
- vocab/BPE/tokenizer가 달라진 비교는 token-level loss 대신 `final_val_nats_per_char` 또는 `final_val_bits_per_char`를 같이 본다.
- 새 가설은 가능하면 baseline run 하나와 단일 변경 변수 하나를 지정한 matched-pair 실험이어야 한다.
- `fit_status == "generalizing"`이면 다음에는 seed 반복으로 재현성을 확인한다.
- `fit_status == "overfit_risk"`이면 dropout, weight decay, tying, 모델 축소를 우선한다.

## 시각화

- `dashboard.md`: loss, generalization gap, overfit_score를 한 화면에서 보는 요약 대시보드
- `metrics_summary.csv`: 시각화와 해석에 쓰는 정규화된 지표 테이블
- `visuals/loss_overfit_trends.svg`: 모든 run의 train/val loss와 과적합 신호 추세
- `visuals/latest_run_metrics.svg`: 최신 run의 loss와 과적합 신호 막대 그래프
- `visuals/effect_map_pairs.svg`: 변수 변경별 loss/overfit 변화량을 함께 보여주는 그래프
- `visuals/correlation_evidence.svg`: tokenizer/dataset/model/training 축과 결과 지표의 관찰 상관 그래프
- `runs/run_XXX_artifacts/run_metrics.svg`: 각 회차 보고서에 포함되는 run별 상세 그래프

## 주요 파일

- `state.json`: 현재 상태와 best run
- `leaderboard.csv`: 모든 완료 실험 요약
- `metrics_summary.csv`: 주요 loss/과적합 지표 요약
- `dashboard.md`: 사람이 바로 볼 수 있는 시각 대시보드
- `effect_map.md`: matched-pair 기준 변수 효과 지도
- `tokenizer_profile.md`: BPE merge, vocab, token/char scale 해석
- `correlation_report.md`: 관찰 상관과 통제 실험으로 증명할 claim 분리
- `research_questions.md`: 다음 실험 질문과 금지할 해석
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
