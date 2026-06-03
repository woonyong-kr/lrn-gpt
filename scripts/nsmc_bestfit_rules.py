#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rule-based NSMC best-fit experiment planner."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs" / "nsmc_bestfit"
LOCAL_DIR = ROOT / "local" / "nsmc_bestfit"
MATRIX_PATH = DOCS_DIR / "run_matrix.csv"
NEXT_PLAN_PATH = DOCS_DIR / "next_plan.json"
DECISION_REPORT_PATH = DOCS_DIR / "decision_report.md"
RULES_PATH = DOCS_DIR / "rules.md"
RUNS_DIR = LOCAL_DIR / "runs"
CACHE_DIR = LOCAL_DIR / "cache"
TRAIN_TEXT = ROOT / "data" / "nsmc_lm_train.txt"
VAL_TEXT = ROOT / "data" / "nsmc_lm_val.txt"
PRIMARY_METRIC = "best_val_bits_per_char"
EXPLORATORY_SEED = 123
CONFIRMATORY_SEEDS = (123, 321, 777, 2026, 42)


@dataclass(frozen=True)
class Condition:
    phase: str
    condition_id: str
    purpose: str
    sweep_axis: str
    axis_value: str
    stage: str = "exploratory"
    vocab_size: int = 16000
    tokenizer_min_frequency: int = 3
    context_length: int = 256
    batch_size: int = 8
    emb_dim: int = 256
    n_heads: int = 8
    n_layers: int = 6
    drop_rate: float = 0.10
    ffn_mult: int = 4
    norm_first: bool = True
    activation_name: str = "gelu"
    attention_impl: str = "sdpa"
    qkv_bias: bool = False
    tie_embeddings: bool = True
    init_std: float = 0.02
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    epochs: float = 5.0
    epoch_milestones: str = ""


BASE = Condition(
    phase="phase1_tokenizer",
    condition_id="TOK_V16000_MF3",
    purpose="NSMC tokenizer baseline vocab=16000 min_frequency=3",
    sweep_axis="tokenizer",
    axis_value="vocab=16000,min_frequency=3",
)


MATRIX_FIELDS = list(asdict(BASE).keys()) + ["run_number", "repeat_index", "seed"]
PHASE_ORDER = ("phase1_tokenizer", "phase2_learning_rate", "phase3_capacity", "phase4_regularization", "phase5_confirm")
PHASE_LABELS = {
    "phase1_tokenizer": "tokenizer efficiency",
    "phase2_learning_rate": "optimization stability",
    "phase3_capacity": "model capacity",
    "phase4_regularization": "overfit control",
    "phase5_confirm": "seed confirmation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--write", action="store_true", help="Write matrix, next plan, and rule reports.")
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_matrix(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def run_dir_for(run_number: int, runs_dir: Path) -> Path:
    return runs_dir / f"run_{run_number:04d}"


def to_int(value: Any) -> int:
    return int(float(str(value).strip()))


def to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compact_float(value: float) -> str:
    return f"{value:g}"


def row_for(condition: Condition, run_number: int, seed: int, repeat_index: int = 1) -> dict[str, Any]:
    row = asdict(condition)
    row["run_number"] = run_number
    row["repeat_index"] = repeat_index
    row["seed"] = seed
    return row


def load_completed_results(runs_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return results
    for result_path in sorted(runs_dir.glob("run_*/result.json")):
        try:
            payload = read_json(result_path)
        except Exception:
            continue
        if payload.get("status") != "completed":
            continue
        payload["_result_path"] = str(result_path)
        payload["_run_dir"] = str(result_path.parent)
        results.append(payload)
    return results


def completed_run_numbers(results: list[dict[str, Any]]) -> set[int]:
    numbers: set[int] = set()
    for result in results:
        run_number = result.get("run_number")
        if run_number is not None:
            numbers.add(to_int(run_number))
    return numbers


def pending_rows(rows: list[dict[str, str]], results: list[dict[str, Any]], runs_dir: Path) -> list[dict[str, str]]:
    completed = completed_run_numbers(results)
    pending: list[dict[str, str]] = []
    for row in rows:
        run_number = to_int(row["run_number"])
        if run_number in completed:
            continue
        if (run_dir_for(run_number, runs_dir) / "result.json").exists():
            continue
        pending.append(row)
    return pending


def next_run_number(rows: list[dict[str, str]]) -> int:
    if not rows:
        return 1
    return max(to_int(row["run_number"]) for row in rows) + 1


def condition_exists(rows: list[dict[str, str]], condition_id: str) -> bool:
    return any(row["condition_id"] == condition_id for row in rows)


def append_conditions(rows: list[dict[str, str]], conditions: list[Condition], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = [dict(row) for row in rows]
    run_number = next_run_number(rows)
    for condition in conditions:
        if condition_exists(rows, condition.condition_id):
            continue
        for repeat_index, seed in enumerate(seeds, start=1):
            updated.append(row_for(condition, run_number=run_number, seed=seed, repeat_index=repeat_index))
            run_number += 1
    return updated


def tokenizer_conditions() -> list[Condition]:
    candidates = (
        (8000, 2),
        (12000, 2),
        (16000, 3),
        (24000, 3),
        (32000, 5),
    )
    return [
        replace(
            BASE,
            condition_id=f"TOK_V{vocab_size}_MF{min_frequency}",
            purpose=f"NSMC byte-level BPE tokenizer vocab={vocab_size} min_frequency={min_frequency}",
            axis_value=f"vocab={vocab_size},min_frequency={min_frequency}",
            vocab_size=vocab_size,
            tokenizer_min_frequency=min_frequency,
        )
        for vocab_size, min_frequency in candidates
    ]


def learning_rate_conditions(champion: dict[str, Any]) -> list[Condition]:
    values = (1e-4, 2e-4, 3e-4, 5e-4)
    return [
        replace(
            condition_from_result(champion),
            phase="phase2_learning_rate",
            condition_id=f"LR{int(round(learning_rate * 1_000_000)):04d}",
            purpose=f"NSMC learning-rate sweep lr={compact_float(learning_rate)}",
            sweep_axis="learning_rate",
            axis_value=compact_float(learning_rate),
            learning_rate=learning_rate,
        )
        for learning_rate in values
    ]


def capacity_conditions(champion: dict[str, Any]) -> list[Condition]:
    variants = (
        ("CAP_SMALL", 192, 6, 4),
        ("CAP_BASE", 256, 8, 6),
        ("CAP_WIDE", 384, 8, 6),
        ("CAP_DEEP", 256, 8, 8),
    )
    base = condition_from_result(champion)
    return [
        replace(
            base,
            phase="phase3_capacity",
            condition_id=condition_id,
            purpose=f"NSMC capacity sweep emb_dim={emb_dim} heads={n_heads} layers={n_layers}",
            sweep_axis="capacity",
            axis_value=f"emb_dim={emb_dim},n_heads={n_heads},n_layers={n_layers}",
            emb_dim=emb_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )
        for condition_id, emb_dim, n_heads, n_layers in variants
    ]


def regularization_conditions(champion: dict[str, Any]) -> list[Condition]:
    base = condition_from_result(champion)
    variants = (
        ("REG_LOW_DO", 0.05, to_float(champion.get("weight_decay"), 0.05)),
        ("REG_BASE_DO", 0.10, to_float(champion.get("weight_decay"), 0.05)),
        ("REG_HIGH_DO", 0.15, to_float(champion.get("weight_decay"), 0.05)),
        ("REG_HIGH_WD", to_float(champion.get("drop_rate"), 0.10), 0.10),
    )
    return [
        replace(
            base,
            phase="phase4_regularization",
            condition_id=condition_id,
            purpose=f"NSMC regularization sweep dropout={compact_float(drop_rate)} weight_decay={compact_float(weight_decay)}",
            sweep_axis="regularization",
            axis_value=f"drop_rate={compact_float(drop_rate)},weight_decay={compact_float(weight_decay)}",
            drop_rate=drop_rate,
            weight_decay=weight_decay,
        )
        for condition_id, drop_rate, weight_decay in variants
    ]


def confirm_conditions(champion: dict[str, Any]) -> list[Condition]:
    base = condition_from_result(champion)
    return [
        replace(
            base,
            phase="phase5_confirm",
            condition_id="CONFIRM_BEST",
            purpose="confirm current best NSMC configuration across multiple seeds",
            sweep_axis="seed",
            axis_value="confirmatory",
            stage="confirmatory",
        )
    ]


def condition_from_result(result: dict[str, Any]) -> Condition:
    return Condition(
        phase=str(result.get("phase", BASE.phase)),
        condition_id=str(result.get("condition_id", BASE.condition_id)),
        purpose=str(result.get("purpose", BASE.purpose)),
        sweep_axis=str(result.get("sweep_axis", BASE.sweep_axis)),
        axis_value=str(result.get("axis_value", BASE.axis_value)),
        stage=str(result.get("stage", BASE.stage)),
        vocab_size=to_int(result.get("vocab_size", BASE.vocab_size)),
        tokenizer_min_frequency=to_int(result.get("tokenizer_min_frequency", BASE.tokenizer_min_frequency)),
        context_length=to_int(result.get("context_length", BASE.context_length)),
        batch_size=to_int(result.get("batch_size", BASE.batch_size)),
        emb_dim=to_int(result.get("emb_dim", BASE.emb_dim)),
        n_heads=to_int(result.get("n_heads", BASE.n_heads)),
        n_layers=to_int(result.get("n_layers", BASE.n_layers)),
        drop_rate=to_float(result.get("drop_rate"), BASE.drop_rate),
        ffn_mult=to_int(result.get("ffn_mult", BASE.ffn_mult)),
        norm_first=str(result.get("norm_first", BASE.norm_first)).lower() in {"1", "true", "yes"},
        activation_name=str(result.get("activation_name", BASE.activation_name)),
        attention_impl=str(result.get("attention_impl", BASE.attention_impl)),
        qkv_bias=str(result.get("qkv_bias", BASE.qkv_bias)).lower() in {"1", "true", "yes"},
        tie_embeddings=str(result.get("tie_embeddings", BASE.tie_embeddings)).lower() in {"1", "true", "yes"},
        init_std=to_float(result.get("init_std"), BASE.init_std),
        learning_rate=to_float(result.get("learning_rate"), BASE.learning_rate),
        weight_decay=to_float(result.get("weight_decay"), BASE.weight_decay),
        grad_clip=to_float(result.get("grad_clip"), BASE.grad_clip),
        epochs=to_float(result.get("epochs", result.get("resolved_epochs")), BASE.epochs),
        epoch_milestones=str(result.get("epoch_milestones", "")),
    )


def result_metric(result: dict[str, Any], key: str, default: float = float("inf")) -> float:
    value = to_float(result.get(key), default)
    return value if math.isfinite(value) else default


def selection_score(result: dict[str, Any]) -> float:
    bits = result_metric(result, PRIMARY_METRIC, result_metric(result, "final_val_bits_per_char"))
    rebound = max(0.0, result_metric(result, "final_minus_best_val_loss", 0.0))
    gap = max(0.0, result_metric(result, "final_generalization_gap", 0.0))
    compute = max(1.0, result_metric(result, "compute_proxy", result_metric(result, "compute_proxy_param_tokens", 1.0)))
    tokens_per_sec = max(1.0, result_metric(result, "tokens_per_sec_after_warmup", result_metric(result, "tokens_per_sec", 1.0)))
    compute_penalty = 0.015 * math.log10(compute)
    speed_bonus = 0.01 * math.log10(tokens_per_sec)
    return bits + 0.25 * rebound + 0.10 * gap + compute_penalty - speed_bonus


def completed_by_phase(results: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    return [result for result in results if result.get("phase") == phase]


def current_phase(rows: list[dict[str, str]]) -> str | None:
    if not rows:
        return None
    seen = {row["phase"] for row in rows}
    for phase in reversed(PHASE_ORDER):
        if phase in seen:
            return phase
    return None


def phase_is_complete(rows: list[dict[str, str]], results: list[dict[str, Any]], phase: str) -> bool:
    phase_rows = [row for row in rows if row["phase"] == phase]
    if not phase_rows:
        return False
    completed = completed_run_numbers(results)
    return all(to_int(row["run_number"]) in completed for row in phase_rows)


def best_result(results: list[dict[str, Any]], phases: tuple[str, ...] = PHASE_ORDER) -> dict[str, Any] | None:
    candidates = [result for result in results if result.get("phase") in phases]
    if not candidates:
        return None
    return min(candidates, key=selection_score)


def build_decision(rows: list[dict[str, str]], results: list[dict[str, Any]], runs_dir: Path, matrix_path: Path = MATRIX_PATH, cache_dir: Path = CACHE_DIR, train_text: Path = TRAIN_TEXT, val_text: Path = VAL_TEXT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pending = pending_rows(rows, results, runs_dir)
    action = "wait_for_pending"
    reason = "pending rows already exist"
    added_phase = None

    if not rows:
        rows = append_conditions(rows, tokenizer_conditions(), seeds=(EXPLORATORY_SEED,))
        pending = pending_rows(rows, results, runs_dir)
        action = "bootstrap_tokenizer_phase"
        reason = "no NSMC best-fit matrix existed"
        added_phase = "phase1_tokenizer"
    elif not pending:
        phase = current_phase(rows)
        if phase == "phase1_tokenizer" and phase_is_complete(rows, results, "phase1_tokenizer"):
            champion = best_result(completed_by_phase(results, "phase1_tokenizer"), phases=("phase1_tokenizer",))
            if champion is not None:
                rows = append_conditions(rows, learning_rate_conditions(champion), seeds=(EXPLORATORY_SEED,))
                action = "append_learning_rate_phase"
                reason = f"tokenizer phase complete; champion={champion.get('condition_id')}"
                added_phase = "phase2_learning_rate"
        elif phase == "phase2_learning_rate" and phase_is_complete(rows, results, "phase2_learning_rate"):
            champion = best_result(results, phases=("phase1_tokenizer", "phase2_learning_rate"))
            if champion is not None:
                rows = append_conditions(rows, capacity_conditions(champion), seeds=(EXPLORATORY_SEED,))
                action = "append_capacity_phase"
                reason = f"learning-rate phase complete; champion={champion.get('condition_id')}"
                added_phase = "phase3_capacity"
        elif phase == "phase3_capacity" and phase_is_complete(rows, results, "phase3_capacity"):
            champion = best_result(results, phases=("phase1_tokenizer", "phase2_learning_rate", "phase3_capacity"))
            if champion is not None:
                rows = append_conditions(rows, regularization_conditions(champion), seeds=(EXPLORATORY_SEED,))
                action = "append_regularization_phase"
                reason = f"capacity phase complete; champion={champion.get('condition_id')}"
                added_phase = "phase4_regularization"
        elif phase == "phase4_regularization" and phase_is_complete(rows, results, "phase4_regularization"):
            champion = best_result(results, phases=("phase1_tokenizer", "phase2_learning_rate", "phase3_capacity", "phase4_regularization"))
            if champion is not None:
                rows = append_conditions(rows, confirm_conditions(champion), seeds=CONFIRMATORY_SEEDS)
                action = "append_confirm_phase"
                reason = f"regularization phase complete; champion={champion.get('condition_id')}"
                added_phase = "phase5_confirm"
        elif phase == "phase5_confirm" and phase_is_complete(rows, results, "phase5_confirm"):
            action = "complete"
            reason = "confirmatory seed phase complete"
        pending = pending_rows(rows, results, runs_dir)

    next_row = pending[0] if pending else None
    champion = best_result(results)
    plan = {
        "updated_at": now(),
        "action": action,
        "reason": reason,
        "added_phase": added_phase,
        "planned_runs": len(rows),
        "completed_runs": len(results),
        "pending_runs": len(pending),
        "next_run": next_row,
        "current_champion": summarize_result(champion) if champion else None,
        "primary_metric": PRIMARY_METRIC,
        "selection_rule": "lower best_val_bits_per_char, then lower rebound/gap/compute, with small throughput bonus",
        "train_text": str(train_text),
        "val_text": str(val_text),
        "matrix": str(matrix_path),
        "cache_dir": str(cache_dir),
        "runs_dir": str(runs_dir),
    }
    return rows, plan


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_number": result.get("run_number"),
        "condition_id": result.get("condition_id"),
        "phase": result.get("phase"),
        "seed": result.get("seed"),
        "score": selection_score(result),
        "best_val_bits_per_char": result.get("best_val_bits_per_char"),
        "final_val_bits_per_char": result.get("final_val_bits_per_char"),
        "final_minus_best_val_loss": result.get("final_minus_best_val_loss"),
        "final_generalization_gap": result.get("final_generalization_gap"),
        "val_tokens_per_char": result.get("cache_vocab_manifest", {}).get("val_tokens_per_char"),
        "val_chars_per_token": result.get("cache_vocab_manifest", {}).get("val_chars_per_token"),
        "tokens_per_sec_after_warmup": result.get("tokens_per_sec_after_warmup"),
        "parameter_count": result.get("parameter_count"),
        "tokens_seen": result.get("tokens_seen"),
        "compute_proxy": result.get("compute_proxy"),
    }


def phase_table(rows: list[dict[str, str]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = completed_run_numbers(results)
    table: list[dict[str, Any]] = []
    for phase in PHASE_ORDER:
        phase_rows = [row for row in rows if row["phase"] == phase]
        if not phase_rows:
            continue
        phase_results = completed_by_phase(results, phase)
        scores = [selection_score(result) for result in phase_results]
        table.append(
            {
                "phase": phase,
                "label": PHASE_LABELS[phase],
                "planned": len(phase_rows),
                "completed": sum(1 for row in phase_rows if to_int(row["run_number"]) in completed),
                "best_score": min(scores) if scores else None,
                "mean_score": mean(scores) if scores else None,
            }
        )
    return table


def write_rules(path: Path) -> None:
    lines = [
        "# NSMC Best-Fit 규칙",
        "",
        "이 흐름은 사람이 매번 다음 실험을 고르는 대신, 완료된 결과를 보고 다음 phase를 추가한다.",
        "",
        "## 기본 데이터",
        "",
        f"- LM train: `{TRAIN_TEXT}`",
        f"- LM val: `{VAL_TEXT}`",
        "- tokenizer/cache/run 산출물은 `local/nsmc_bestfit`에 둔다.",
        "- 보고서와 matrix는 `docs/nsmc_bestfit`에 둔다.",
        "- 누적 선형 그래프는 `docs/nsmc_bestfit/linear_graphs/figure_index.md`에 둔다.",
        "",
        "## 선택 기준",
        "",
        "1. tokenizer가 다른 실험은 token loss만 비교하지 않고 `best_val_bits_per_char`를 우선한다.",
        "2. 같은 수준이면 `final_minus_best_val_loss`가 작은 쪽을 고른다. 후반 rebound가 작다는 뜻이다.",
        "3. 그 다음은 `final_generalization_gap`이 작은 쪽을 고른다. train만 좋아지고 val이 벌어지는 후보를 낮춘다.",
        "4. 그 다음은 `compute_proxy = parameter_count * tokens_seen`이 작은 쪽을 고른다.",
        "5. 마지막으로 `tokens_per_sec_after_warmup`이 높은 쪽에 작은 보너스를 준다.",
        "",
        "## phase 순서",
        "",
        "1. `phase1_tokenizer`: vocab_size와 min_frequency를 먼저 고른다.",
        "2. `phase2_learning_rate`: tokenizer champion을 고정하고 learning rate를 고른다.",
        "3. `phase3_capacity`: optimizer/tokenizer를 고정하고 모델 폭과 깊이를 고른다.",
        "4. `phase4_regularization`: dropout과 weight decay로 과적합을 제어한다.",
        "5. `phase5_confirm`: 최종 후보를 여러 seed로 반복해서 우연인지 확인한다.",
        "",
        "## 한글 tokenizer에 대한 규칙",
        "",
        "현재 실행 루프는 GPT 계열과 같은 byte-level BPE를 쓴다. 따라서 `영화`와 ` 영화`는 둘 다 유효한 후보가 될 수 있다.",
        "공백 포함 토큰은 언어모델 효율에는 유리할 수 있으므로 금지하지 않는다.",
        "대신 `min_frequency`와 `vocab_size`를 같이 보고, 문장 전체가 거대한 토큰으로 커지는 후보는 bits/char와 일반화 gap에서 걸러낸다.",
        "",
        "## 누적 그래프 규칙",
        "",
        "- 모든 완료 실행은 `all_run_results.jsonl`에 원장으로 남긴다.",
        "- 그래프는 원장을 `run_number` 순서로 읽어 누적 선형 그래프로 다시 그린다.",
        "- 핵심 그래프는 quality, tokenizer efficiency, compute, generalization, training history, hyperparameter path를 나눠 본다.",
        "- 가장 큰 전체 대시보드는 `00_all_metrics_linear_dashboard.png`다.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision_report(path: Path, plan: dict[str, Any], rows: list[dict[str, str]], results: list[dict[str, Any]]) -> None:
    next_row = plan.get("next_run") or {}
    champion = plan.get("current_champion") or {}
    lines = [
        "# NSMC Best-Fit Decision Report",
        "",
        f"- updated_at: `{plan['updated_at']}`",
        f"- action: `{plan['action']}`",
        f"- reason: `{plan['reason']}`",
        f"- planned_runs: `{plan['planned_runs']}`",
        f"- completed_runs: `{plan['completed_runs']}`",
        f"- pending_runs: `{plan['pending_runs']}`",
        "",
        "## 다음 실행",
        "",
    ]
    if next_row:
        lines.extend(
            [
                f"- run_number: `{next_row['run_number']}`",
                f"- condition_id: `{next_row['condition_id']}`",
                f"- phase: `{next_row['phase']}`",
                f"- purpose: `{next_row['purpose']}`",
                f"- seed: `{next_row['seed']}`",
                f"- tokenizer: `vocab={next_row['vocab_size']}, min_frequency={next_row.get('tokenizer_min_frequency', 2)}`",
                f"- model: `emb={next_row['emb_dim']}, heads={next_row['n_heads']}, layers={next_row['n_layers']}, ffn_mult={next_row['ffn_mult']}`",
                f"- optimization: `lr={next_row['learning_rate']}, drop={next_row['drop_rate']}, wd={next_row['weight_decay']}`",
            ]
        )
    else:
        lines.append("- pending run 없음")
    lines.extend(["", "## 현재 champion", ""])
    if champion:
        for key, value in champion.items():
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- 완료된 결과가 아직 없음")
    lines.extend(["", "## Phase 상태", "", "| phase | label | planned | completed | best_score | mean_score |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in phase_table(rows, results):
        best = "" if row["best_score"] is None else f"{row['best_score']:.6g}"
        avg = "" if row["mean_score"] is None else f"{row['mean_score']:.6g}"
        lines.append(f"| {row['phase']} | {row['label']} | {row['planned']} | {row['completed']} | {best} | {avg} |")
    lines.extend(
        [
            "",
            "## 누적 선형 그래프",
            "",
            "- figure index: `docs/nsmc_bestfit/linear_graphs/figure_index.md`",
            "- dashboard: `docs/nsmc_bestfit/linear_graphs/00_all_metrics_linear_dashboard.png`",
            "",
            "",
            "## 실행 명령",
            "",
            "한 번만 실행:",
            "",
            "```bash",
            "python scripts/nsmc_bestfit_step.py --max-runs 1 --device auto --eval-batches 20",
            "```",
            "",
            "가벼운 smoke 실행:",
            "",
            "```bash",
            "python scripts/nsmc_bestfit_step.py --max-runs 1 --max-steps-per-run 5 --device cpu --eval-batches 2",
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_rules(matrix_path: Path = MATRIX_PATH, runs_dir: Path = RUNS_DIR, docs_dir: Path = DOCS_DIR, write: bool = True) -> dict[str, Any]:
    rows = read_matrix(matrix_path)
    results = load_completed_results(runs_dir)
    rows, plan = build_decision(rows, results, runs_dir, matrix_path=matrix_path)
    if write:
        write_matrix(matrix_path, rows)
        write_json(docs_dir / NEXT_PLAN_PATH.name, plan)
        write_rules(docs_dir / RULES_PATH.name)
        write_decision_report(docs_dir / DECISION_REPORT_PATH.name, plan, rows, results)
    return plan


def main() -> None:
    args = parse_args()
    plan = apply_rules(matrix_path=args.matrix, runs_dir=args.runs_dir, docs_dir=args.docs_dir, write=args.write)
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
