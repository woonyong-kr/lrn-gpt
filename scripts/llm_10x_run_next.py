#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one pending LLM 10x long-run experiment from the matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from llm_10x_common import CACHE_DIR, MATRIX_PATH, RUNS_DIR, cache_dir_for, ensure_src_import, parse_bool, parse_float, parse_int, read_json, read_matrix, read_u32, run_dir_for, write_json


ensure_src_import()
from src.config import GPTConfig, set_seed  # noqa: E402
from src.dataset import create_dataloader  # noqa: E402
from src.experiments import count_parameters, estimate_steps_per_epoch  # noqa: E402
from src.model import GPTModel  # noqa: E402
from src.train import calc_loss_batch, calc_loss_loader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--run-number", type=int, default=None, help="Run a specific matrix row. Defaults to first pending row.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--epochs", type=float, default=None, help="Override matrix epochs for smoke tests or resumed plans.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override total optimizer steps for smoke tests.")
    parser.add_argument("--queue-status-json", type=Path, default=None, help="Optional queue-level status file to update during training.")
    parser.add_argument("--progress-every-steps", type=int, default=25)
    parser.add_argument("--checkpoint-every-epochs", type=float, default=5.0)
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if result.json exists.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def completed_run_numbers(runs_dir: Path) -> set[int]:
    completed: set[int] = set()
    if not runs_dir.exists():
        return completed
    for result_path in runs_dir.glob("run_*/result.json"):
        try:
            payload = read_json(result_path)
        except Exception:
            continue
        if payload.get("status") == "completed":
            completed.add(parse_int(payload.get("run_number", result_path.parent.name.split("_")[-1])))
    return completed


def select_row(rows: list[dict[str, str]], args: argparse.Namespace) -> dict[str, str]:
    if args.run_number is not None:
        for row in rows:
            if parse_int(row["run_number"]) == args.run_number:
                return row
        raise ValueError(f"run_number not found in matrix: {args.run_number}")

    completed = completed_run_numbers(args.runs_dir)
    for row in rows:
        run_number = parse_int(row["run_number"])
        result_path = run_dir_for(run_number, args.runs_dir) / "result.json"
        if run_number not in completed and (args.force or not result_path.exists()):
            return row
    raise ValueError("no pending run found")


def model_config_from_row(row: dict[str, str]) -> GPTConfig:
    return GPTConfig(
        vocab_size=parse_int(row["vocab_size"]),
        context_length=parse_int(row["context_length"]),
        emb_dim=parse_int(row["emb_dim"]),
        n_heads=parse_int(row["n_heads"]),
        n_layers=parse_int(row["n_layers"]),
        drop_rate=parse_float(row["drop_rate"]),
        qkv_bias=parse_bool(row["qkv_bias"]),
        ffn_mult=parse_int(row["ffn_mult"]),
        norm_first=parse_bool(row["norm_first"]),
        activation_name=row["activation_name"],
        attention_impl=row["attention_impl"],
        tie_embeddings=parse_bool(row["tie_embeddings"]),
        init_std=parse_float(row["init_std"]),
        seed=parse_int(row["seed"]),
    )


def config_record(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    record: dict[str, Any] = dict(row)
    for key in ("vocab_size", "tokenizer_min_frequency", "context_length", "batch_size", "emb_dim", "n_heads", "n_layers", "ffn_mult", "run_number", "repeat_index", "seed"):
        record[key] = parse_int(record[key])
    for key in ("drop_rate", "init_std", "learning_rate", "weight_decay", "grad_clip", "epochs"):
        record[key] = parse_float(record[key])
    for key in ("norm_first", "qkv_bias", "tie_embeddings"):
        record[key] = parse_bool(record[key])
    if args.epochs is not None:
        record["epochs"] = args.epochs
    return record


def load_cache(vocab_size: int, min_frequency: int, cache_root: Path) -> tuple[Any, Any, dict[str, Any]]:
    cache_path = cache_dir_for(vocab_size, cache_root, min_frequency)
    manifest_path = cache_path / "manifest.json"
    train_path = cache_path / "train_ids.u32"
    val_path = cache_path / "val_ids.u32"
    missing = [path for path in (manifest_path, train_path, val_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing cache files for vocab={vocab_size}, min_frequency={min_frequency}: {missing}")
    return read_u32(train_path), read_u32(val_path), read_json(manifest_path)


def save_checkpoint(model: GPTModel, optimizer: torch.optim.Optimizer, path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **payload,
        },
        path,
    )


def compute_overfit_metrics(initial_train_loss: float, initial_val_loss: float, final_train_loss: float, final_val_loss: float) -> dict[str, Any]:
    train_loss_delta = initial_train_loss - final_train_loss
    val_loss_delta = initial_val_loss - final_val_loss
    initial_generalization_gap = initial_val_loss - initial_train_loss
    final_generalization_gap = final_val_loss - final_train_loss
    generalization_gap_delta = final_generalization_gap - initial_generalization_gap
    train_val_improvement_gap = train_loss_delta - val_loss_delta
    overfit_score = max(0.0, final_generalization_gap) + max(0.0, generalization_gap_delta) + max(0.0, train_val_improvement_gap)

    if val_loss_delta < -0.01:
        fit_status = "val_regressed"
    elif train_loss_delta < 0.005 and val_loss_delta < 0.005:
        fit_status = "underfit_or_too_short"
    elif generalization_gap_delta > 0.05 and train_val_improvement_gap > 0.02:
        fit_status = "overfit_risk"
    elif train_loss_delta > 0.01 and val_loss_delta > 0.01:
        fit_status = "generalizing"
    else:
        fit_status = "mixed"

    return {
        "train_loss_delta": train_loss_delta,
        "val_loss_delta": val_loss_delta,
        "initial_generalization_gap": initial_generalization_gap,
        "final_generalization_gap": final_generalization_gap,
        "generalization_gap_delta": generalization_gap_delta,
        "train_val_improvement_gap": train_val_improvement_gap,
        "overfit_score": overfit_score,
        "fit_status": fit_status,
    }


def write_progress_status(run_dir: Path, args: argparse.Namespace, payload: dict[str, Any]) -> None:
    updated = {"updated_at": datetime.now(timezone.utc).isoformat(), **payload}
    write_json(run_dir / "status.json", updated)
    if args.queue_status_json is not None:
        queue_payload: dict[str, Any] = {}
        if args.queue_status_json.exists():
            try:
                queue_payload = read_json(args.queue_status_json)
            except Exception:
                queue_payload = {}
        queue_payload.update(updated)
        write_json(args.queue_status_json, queue_payload)


def should_checkpoint(step: int, steps_per_epoch: int, checkpoint_every_epochs: float) -> bool:
    if checkpoint_every_epochs <= 0:
        return False
    interval = max(1, int(math.ceil(steps_per_epoch * checkpoint_every_epochs)))
    return step > 0 and step % interval == 0


def run_one(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    record = config_record(row, args)
    run_number = parse_int(record["run_number"])
    run_dir = run_dir_for(run_number, args.runs_dir)
    result_path = run_dir / "result.json"
    if result_path.exists() and not args.force:
        return {"status": "exists", "run_number": run_number, "result_path": str(result_path)}

    train_ids, val_ids, cache_manifest = load_cache(parse_int(record["vocab_size"]), parse_int(record.get("tokenizer_min_frequency", 2)), args.cache_dir)
    context_length = parse_int(record["context_length"])
    batch_size = parse_int(record.get("batch_size", 4)) if "batch_size" in record else 4
    stride = context_length
    steps_per_epoch = estimate_steps_per_epoch(len(train_ids), context_length, batch_size, stride=stride)
    epochs = parse_float(record["epochs"])
    total_steps = max(1, int(math.ceil(epochs * steps_per_epoch)))
    if args.max_steps is not None:
        total_steps = args.max_steps
        epochs = total_steps / steps_per_epoch

    if args.dry_run:
        return {
            "status": "dry_run",
            "run_number": run_number,
            "condition_id": record["condition_id"],
            "seed": record["seed"],
            "vocab_size": record["vocab_size"],
            "tokenizer_min_frequency": record.get("tokenizer_min_frequency", 2),
            "train_tokens": len(train_ids),
            "val_tokens": len(val_ids),
            "steps_per_epoch": steps_per_epoch,
            "epochs": epochs,
            "total_steps": total_steps,
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", record)
    write_json(run_dir / "cache_manifest.json", cache_manifest)
    started_at = datetime.now(timezone.utc).isoformat()
    write_progress_status(
        run_dir,
        args,
        {
            "status": "running",
            "stage": "initializing",
            "run_number": run_number,
            "current_run_number": run_number,
            "condition_id": record["condition_id"],
            "current_condition_id": record["condition_id"],
            "seed": record["seed"],
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "current_step": 0,
            "remaining_steps": total_steps,
            "progress_percent": 0.0,
            "epochs": epochs,
            "max_steps": total_steps,
            "started_at": started_at,
        },
    )

    set_seed(parse_int(record["seed"]))
    device = resolve_device(args.device)
    train_loader = create_dataloader(train_ids, context_length=context_length, batch_size=batch_size, stride=stride, shuffle=True, drop_last=False, seed=parse_int(record["seed"]))
    val_loader = create_dataloader(val_ids, context_length=context_length, batch_size=batch_size, stride=stride, shuffle=False, drop_last=False)
    model_cfg = model_config_from_row(row)
    model = GPTModel(model_cfg.to_dict()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=parse_float(record["learning_rate"]), weight_decay=parse_float(record["weight_decay"]))

    initial_train_loss = calc_loss_loader(train_loader, model, device, num_batches=args.eval_batches)
    initial_val_loss = calc_loss_loader(val_loader, model, device, num_batches=args.eval_batches)
    history_path = run_dir / "history.jsonl"
    best_val_loss = float("inf")
    best_step = 0
    tokens_seen = 0
    last_loss = initial_train_loss
    start = time.perf_counter()
    train_iter = iter(train_loader)
    model.train()
    write_progress_status(
        run_dir,
        args,
        {
            "status": "running",
            "stage": "training",
            "run_number": run_number,
            "current_run_number": run_number,
            "condition_id": record["condition_id"],
            "current_condition_id": record["condition_id"],
            "seed": record["seed"],
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "current_step": 0,
            "remaining_steps": total_steps,
            "progress_percent": 0.0,
            "epochs": epochs,
            "max_steps": total_steps,
            "initial_train_loss": initial_train_loss,
            "initial_val_loss": initial_val_loss,
            "started_at": started_at,
        },
    )

    with history_path.open("w", encoding="utf-8") as history:
        for step in range(1, total_steps + 1):
            try:
                input_batch, target_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_batch, target_batch = next(train_iter)

            optimizer.zero_grad(set_to_none=True)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            grad_clip = parse_float(record["grad_clip"])
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            last_loss = float(loss.item())
            tokens_seen += int(input_batch.numel())
            elapsed_so_far = time.perf_counter() - start

            end_of_epoch = step % steps_per_epoch == 0
            final_step = step == total_steps
            if step % max(1, args.progress_every_steps) == 0 or final_step:
                write_progress_status(
                    run_dir,
                    args,
                    {
                        "status": "running",
                        "stage": "training",
                        "run_number": run_number,
                        "current_run_number": run_number,
                        "condition_id": record["condition_id"],
                        "current_condition_id": record["condition_id"],
                        "seed": record["seed"],
                        "steps_per_epoch": steps_per_epoch,
                        "total_steps": total_steps,
                        "current_step": step,
                        "remaining_steps": max(0, total_steps - step),
                        "progress_percent": 100.0 * step / total_steps,
                        "epochs": epochs,
                        "current_epoch": step / steps_per_epoch,
                        "max_steps": total_steps,
                        "last_step_loss": last_loss,
                        "tokens_seen": tokens_seen,
                        "tokens_per_sec": 0.0 if elapsed_so_far == 0 else tokens_seen / elapsed_so_far,
                        "started_at": started_at,
                    },
                )
            if end_of_epoch or final_step:
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=args.eval_batches)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=args.eval_batches)
                epoch = step / steps_per_epoch
                event_elapsed = time.perf_counter() - start
                event_overfit = compute_overfit_metrics(initial_train_loss, initial_val_loss, train_loss, val_loss)
                event = {
                    "step": step,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "last_step_loss": last_loss,
                    "elapsed_sec": event_elapsed,
                    "tokens_seen": tokens_seen,
                    "tokens_per_sec": 0.0 if event_elapsed == 0 else tokens_seen / event_elapsed,
                    **event_overfit,
                }
                history.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                history.flush()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    if not args.no_checkpoints:
                        save_checkpoint(model, optimizer, run_dir / "checkpoints" / "best.pt", {"step": step, "epoch": epoch, "val_loss": val_loss, "config": record})
            if not args.no_checkpoints and should_checkpoint(step, steps_per_epoch, args.checkpoint_every_epochs):
                save_checkpoint(model, optimizer, run_dir / "checkpoints" / f"step_{step}.pt", {"step": step, "epoch": step / steps_per_epoch, "config": record})

    elapsed = time.perf_counter() - start
    final_train_loss = calc_loss_loader(train_loader, model, device, num_batches=args.eval_batches)
    final_val_loss = calc_loss_loader(val_loader, model, device, num_batches=args.eval_batches)
    overfit_metrics = compute_overfit_metrics(initial_train_loss, initial_val_loss, final_train_loss, final_val_loss)
    train_tokens_per_char = cache_manifest["train_tokens_per_char"]
    val_tokens_per_char = cache_manifest["val_tokens_per_char"]
    result = {
        "status": "completed",
        **record,
        "device": str(device),
        "steps_per_epoch": steps_per_epoch,
        "max_steps": total_steps,
        "resolved_epochs": epochs,
        "parameter_count": count_parameters(model),
        "initial_train_loss": initial_train_loss,
        "initial_val_loss": initial_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        **overfit_metrics,
        "final_train_bits_per_char": final_train_loss * train_tokens_per_char / math.log(2),
        "final_val_bits_per_char": final_val_loss * val_tokens_per_char / math.log(2),
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "best_epoch": 0.0 if steps_per_epoch == 0 else best_step / steps_per_epoch,
        "tokens_seen": tokens_seen,
        "tokens_per_sec": 0.0 if elapsed == 0 else tokens_seen / elapsed,
        "seconds_per_epoch": 0.0 if epochs == 0 else elapsed / epochs,
        "elapsed_sec": elapsed,
        "cache_vocab_manifest": cache_manifest,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(result_path, result)
    write_progress_status(run_dir, args, {"status": "completed", "stage": "completed", "run_number": run_number, "current_run_number": run_number, "condition_id": record["condition_id"], "current_condition_id": record["condition_id"], "completed_at": result["completed_at"], "current_step": total_steps, "total_steps": total_steps, "remaining_steps": 0, "progress_percent": 100.0})
    return result


def main() -> None:
    args = parse_args()
    rows = read_matrix(args.matrix)
    row = select_row(rows, args)
    result = run_one(row, args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
