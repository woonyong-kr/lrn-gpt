#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the LLM 10x broad growth-sweep condition x seed matrix."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path


EXPLORATORY_SEEDS = (123, 321, 777)
CONFIRMATORY_SEEDS = (123, 321, 777, 2026, 3407, 42, 9001, 2718, 31415, 1618)
DEFAULT_OUTPUT = Path("docs/llm_10x/run_matrix.csv")


@dataclass(frozen=True)
class Condition:
    phase: str
    condition_id: str
    purpose: str
    sweep_axis: str
    axis_value: str
    stage: str = "exploratory"
    vocab_size: int = 12000
    tokenizer_min_frequency: int = 2
    context_length: int = 512
    batch_size: int = 4
    emb_dim: int = 512
    n_heads: int = 8
    n_layers: int = 8
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
    epochs: int = 50
    epoch_milestones: str = ""


BASE = Condition(
    phase="phase1_lr",
    condition_id="LR0300",
    purpose="learning-rate baseline lr=0.0003",
    sweep_axis="learning_rate",
    axis_value="0.0003",
)


def compact_float(value: float) -> str:
    return f"{value:g}"


def learning_rate_conditions() -> list[Condition]:
    values = (5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3, 3e-3)
    return [
        replace(
            BASE,
            condition_id=f"LR{int(round(learning_rate * 1_000_000)):04d}",
            purpose=f"learning-rate sweep lr={compact_float(learning_rate)}",
            learning_rate=learning_rate,
            axis_value=compact_float(learning_rate),
        )
        for learning_rate in values
    ]


def epoch_conditions() -> list[Condition]:
    values = (25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1500)
    milestones = "|".join(str(value) for value in values)
    return [
        replace(
            BASE,
            phase="phase2_epoch",
            condition_id="EH1500",
            purpose="epoch horizon long-run with 25..1500 milestone extraction",
            sweep_axis="epoch_milestones",
            axis_value=milestones,
            epochs=max(values),
            epoch_milestones=milestones,
        )
    ]


def vocab_conditions() -> list[Condition]:
    values = (4000, 6000, 8000, 10000, 12000, 16000, 20000, 24000, 32000, 40000, 48000, 56000, 64000)
    return [
        replace(
            BASE,
            phase="phase3_vocab",
            condition_id=f"V{vocab_size:05d}",
            purpose=f"tokenizer vocab-size sweep vocab={vocab_size}",
            sweep_axis="vocab_size",
            axis_value=str(vocab_size),
            vocab_size=vocab_size,
            tokenizer_min_frequency=2,
        )
        for vocab_size in values
    ]


def tokenizer_granularity_conditions() -> list[Condition]:
    values = (1, 2, 3, 4, 5, 8, 13, 21, 34, 55)
    return [
        replace(
            BASE,
            phase="phase4_tokenizer_granularity",
            condition_id=f"MF{min_frequency:03d}",
            purpose=f"BPE merge min-frequency sweep min_frequency={min_frequency}",
            sweep_axis="tokenizer_min_frequency",
            axis_value=str(min_frequency),
            vocab_size=20000,
            tokenizer_min_frequency=min_frequency,
        )
        for min_frequency in values
    ]


def capacity_conditions() -> list[Condition]:
    values = (
        ("M031", "31M baseline", 512, 8, 8),
        ("M044", "44M deeper", 512, 8, 12),
        ("M057", "57M deepest 512-width", 512, 8, 16),
        ("M066", "66M wider", 768, 12, 8),
        ("M098", "98M wide-deep", 768, 12, 12),
        ("M130", "130M widest-deep 768-width", 768, 12, 16),
        ("M110", "110M 1024-width", 1024, 16, 8),
        ("M160", "160M 1024-width deep", 1024, 16, 12),
        ("M210", "210M 1024-width deepest", 1024, 16, 16),
        ("M240", "240M 1280-width deep", 1280, 20, 12),
    )
    return [
        replace(
            BASE,
            phase="phase5_capacity",
            condition_id=condition_id,
            purpose=f"capacity growth sweep {label}",
            sweep_axis="capacity",
            axis_value=label,
            emb_dim=emb_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )
        for condition_id, label, emb_dim, n_heads, n_layers in values
    ]


def context_conditions() -> list[Condition]:
    values = (256, 384, 512, 640, 768, 1024, 1280, 1536, 2048)
    return [
        replace(
            BASE,
            phase="phase6_context",
            condition_id=f"CTX{context_length:04d}",
            purpose=f"context-length growth sweep context={context_length}",
            sweep_axis="context_length",
            axis_value=str(context_length),
            context_length=context_length,
        )
        for context_length in values
    ]


def batch_size_conditions() -> list[Condition]:
    values = (1, 2, 4, 8, 16)
    return [
        replace(
            BASE,
            phase="phase7_batch_size",
            condition_id=f"BS{batch_size:03d}",
            purpose=f"batch-size sweep batch_size={batch_size}",
            sweep_axis="batch_size",
            axis_value=str(batch_size),
            batch_size=batch_size,
        )
        for batch_size in values
    ]


def dropout_conditions() -> list[Condition]:
    values = (0.00, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40)
    return [
        replace(
            BASE,
            phase="phase8_dropout",
            condition_id=f"DO{int(round(dropout * 1000)):03d}",
            purpose=f"dropout sweep dropout={compact_float(dropout)}",
            sweep_axis="drop_rate",
            axis_value=compact_float(dropout),
            drop_rate=dropout,
        )
        for dropout in values
    ]


def weight_decay_conditions() -> list[Condition]:
    values = (0.00, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50)
    return [
        replace(
            BASE,
            phase="phase9_weight_decay",
            condition_id=f"WD{int(round(weight_decay * 1000)):03d}",
            purpose=f"weight-decay sweep weight_decay={compact_float(weight_decay)}",
            sweep_axis="weight_decay",
            axis_value=compact_float(weight_decay),
            weight_decay=weight_decay,
        )
        for weight_decay in values
    ]


def grad_clip_conditions() -> list[Condition]:
    values = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0)
    return [
        replace(
            BASE,
            phase="phase10_grad_clip",
            condition_id=f"GC{int(round(grad_clip * 100)):03d}",
            purpose=f"gradient clipping sweep grad_clip={compact_float(grad_clip)}",
            sweep_axis="grad_clip",
            axis_value=compact_float(grad_clip),
            grad_clip=grad_clip,
        )
        for grad_clip in values
    ]


def ffn_multiplier_conditions() -> list[Condition]:
    values = (2, 3, 4, 5, 6, 8)
    return [
        replace(
            BASE,
            phase="phase11_ffn_mult",
            condition_id=f"FFN{ffn_mult:02d}",
            purpose=f"FFN multiplier sweep ffn_mult={ffn_mult}",
            sweep_axis="ffn_mult",
            axis_value=str(ffn_mult),
            ffn_mult=ffn_mult,
        )
        for ffn_mult in values
    ]


def activation_conditions() -> list[Condition]:
    values = ("gelu", "gelu_exact", "quick_gelu", "relu", "silu", "swish", "mish", "squared_relu", "identity", "swiglu", "geglu")
    return [
        replace(
            BASE,
            phase="phase12_activation",
            condition_id=f"ACT_{activation.upper().replace('_', '')}",
            purpose=f"activation sweep activation={activation}",
            sweep_axis="activation_name",
            axis_value=activation,
            activation_name=activation,
        )
        for activation in values
    ]


def init_std_conditions() -> list[Condition]:
    values = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04)
    return [
        replace(
            BASE,
            phase="phase13_init_std",
            condition_id=f"INIT{int(round(init_std * 1000)):03d}",
            purpose=f"initialization std sweep init_std={compact_float(init_std)}",
            sweep_axis="init_std",
            axis_value=compact_float(init_std),
            init_std=init_std,
        )
        for init_std in values
    ]


def structure_conditions() -> list[Condition]:
    return [
        replace(
            BASE,
            phase="phase14_structure",
            condition_id="STRUCT_BASE",
            purpose="structure baseline pre-LN tied embeddings SDPA no QKV bias",
            sweep_axis="structure",
            axis_value="baseline",
        ),
        replace(
            BASE,
            phase="phase14_structure",
            condition_id="STRUCT_QKVB",
            purpose="structure sweep QKV bias enabled",
            sweep_axis="structure",
            axis_value="qkv_bias=True",
            qkv_bias=True,
        ),
        replace(
            BASE,
            phase="phase14_structure",
            condition_id="STRUCT_UNTIED",
            purpose="structure sweep untied input/output embeddings",
            sweep_axis="structure",
            axis_value="tie_embeddings=False",
            tie_embeddings=False,
        ),
        replace(
            BASE,
            phase="phase14_structure",
            condition_id="STRUCT_POSTLN",
            purpose="structure sweep post-LN transformer blocks",
            sweep_axis="structure",
            axis_value="norm_first=False",
            norm_first=False,
        ),
        replace(
            BASE,
            phase="phase14_structure",
            condition_id="STRUCT_MANUALATT",
            purpose="structure sweep manual attention implementation",
            sweep_axis="structure",
            axis_value="attention_impl=manual",
            attention_impl="manual",
        ),
        replace(
            BASE,
            phase="phase14_structure",
            condition_id="STRUCT_QKVB_UNTIED",
            purpose="structure sweep QKV bias plus untied embeddings",
            sweep_axis="structure",
            axis_value="qkv_bias=True,tie_embeddings=False",
            qkv_bias=True,
            tie_embeddings=False,
        ),
    ]


def conditions() -> list[Condition]:
    rows: list[Condition] = []
    rows.extend(learning_rate_conditions())
    rows.extend(epoch_conditions())
    rows.extend(vocab_conditions())
    rows.extend(tokenizer_granularity_conditions())
    rows.extend(capacity_conditions())
    rows.extend(context_conditions())
    rows.extend(batch_size_conditions())
    rows.extend(dropout_conditions())
    rows.extend(weight_decay_conditions())
    rows.extend(grad_clip_conditions())
    rows.extend(ffn_multiplier_conditions())
    rows.extend(activation_conditions())
    rows.extend(init_std_conditions())
    rows.extend(structure_conditions())
    return rows


def row_for(condition: Condition, repeat_index: int, seed: int, run_number: int) -> dict[str, object]:
    row = asdict(condition)
    row["run_number"] = run_number
    row["repeat_index"] = repeat_index
    row["seed"] = seed
    return row


def write_matrix(path: Path, seeds: tuple[int, ...]) -> int:
    rows: list[dict[str, object]] = []
    run_number = 1
    for condition in conditions():
        for repeat_index, seed in enumerate(seeds, start=1):
            rows.append(row_for(condition, repeat_index, seed, run_number))
            run_number += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-mode", choices=["explore", "confirm"], default="explore")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = EXPLORATORY_SEEDS if args.seed_mode == "explore" else CONFIRMATORY_SEEDS
    count = write_matrix(args.output, seeds)
    print(f"wrote {count} planned {args.seed_mode} runs to {args.output}")


if __name__ == "__main__":
    main()
