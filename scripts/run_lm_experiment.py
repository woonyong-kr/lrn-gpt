from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bpe import BPETokenizer
from config import set_seed
from dataset import GPTDataset
from model import GPTModel
from train import calc_loss_batch, generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one LM experiment and write one markdown result file.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--purpose", default="")
    parser.add_argument("--change", default="")
    parser.add_argument("--train-path", default="data/nsmc_lm_train.txt")
    parser.add_argument("--val-path", default="data/nsmc_lm_val.txt")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--vocab-size", type=int, default=3000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--activation", choices=["gelu", "relu", "silu"], default="gelu")
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--qkv-bias", action="store_true")
    parser.add_argument("--weight-tying", action="store_true")
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--eval-iter", type=int, default=20)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prompt", default="이 영화는")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--corpus-limit", type=int, default=None)
    parser.add_argument("--train-token-limit", type=int, default=None)
    parser.add_argument("--val-token-limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_text(path: Path, limit: int | None = None) -> str:
    text = path.read_text(encoding="utf-8")
    return text if limit is None else text[:limit]


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_or_load_tokenizer(args: argparse.Namespace, train_text: str) -> tuple[BPETokenizer, Path, float, bool]:
    tokenizer_path = resolve_path(args.tokenizer_path or f"data/nsmc_bpe_vocab_{args.vocab_size}.json")
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    started = time.perf_counter()
    loaded = False
    if tokenizer_path.exists():
        tokenizer.load(tokenizer_path)
        loaded = True
    else:
        tokenizer.train(train_text)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tokenizer_path)
    return tokenizer, tokenizer_path, time.perf_counter() - started, loaded


def make_loader(token_ids: list[int], args: argparse.Namespace, shuffle: bool) -> torch.utils.data.DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    dataset = GPTDataset(token_ids, context_length=args.context_length, stride=args.stride or args.context_length)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=args.num_workers,
        generator=generator if shuffle else None,
    )


def grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        norm = param.grad.detach().data.norm(2).item()
        total += norm * norm
    return total**0.5


def calc_loss_loader_safe(data_loader, model: GPTModel, device: torch.device, num_batches: int | None = None) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
            total_batches += 1
    if was_training:
        model.train()
    return 0.0 if total_batches == 0 else total_loss / total_batches


def maybe_autocast(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def eval_snapshot(model, train_loader, val_loader, device, args, epoch: int, step: int, tokens_seen: int, elapsed: float, last_grad_norm: float | None) -> dict:
    train_loss = calc_loss_loader_safe(train_loader, model, device, num_batches=args.eval_iter)
    val_loss = calc_loss_loader_safe(val_loader, model, device, num_batches=args.eval_iter)
    return {
        "epoch": epoch,
        "step": step,
        "tokens_seen": tokens_seen,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": val_loss,
        "loss_gap": val_loss - train_loss,
        "lr": args.lr,
        "grad_norm": "" if last_grad_norm is None else last_grad_norm,
        "elapsed_sec": elapsed,
        "tokens_per_sec": 0.0 if elapsed <= 0 else tokens_seen / elapsed,
    }


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = get_device(args.device)

    train_path = resolve_path(args.train_path)
    val_path = resolve_path(args.val_path)
    train_text = load_text(train_path, args.corpus_limit)
    val_text = load_text(val_path, args.corpus_limit)
    tokenizer, tokenizer_path, tokenizer_elapsed, tokenizer_loaded = build_or_load_tokenizer(args, train_text)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    if args.train_token_limit is not None:
        train_ids = train_ids[: args.train_token_limit]
    if args.val_token_limit is not None:
        val_ids = val_ids[: args.val_token_limit]

    train_loader = make_loader(train_ids, args, shuffle=True)
    val_loader = make_loader(val_ids, args, shuffle=False)

    config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": args.qkv_bias,
        "ffn_mult": args.ffn_mult,
        "activation": args.activation,
        "norm_first": args.norm_first,
        "seed": args.seed,
        "debug": False,
    }
    model = GPTModel(config)
    if args.weight_tying:
        model.lm_head.weight = model.embedding.token_embedding.weight
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics = []
    tokens_seen = 0
    global_step = 0
    last_grad_norm = None
    started = time.perf_counter()

    metrics.append(eval_snapshot(model, train_loader, val_loader, device, args, 0, 0, 0, 0.0, None))

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        for input_batch, target_batch in train_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with maybe_autocast(device, args.amp):
                loss, _ = model(input_batch, targets=target_batch)
            loss.backward()
            last_grad_norm = grad_norm(model.parameters())
            optimizer.step()

            batch_tokens = int(input_batch.numel())
            tokens_seen += batch_tokens
            global_step += 1

            should_eval = args.eval_freq > 0 and global_step % args.eval_freq == 0
            if should_eval:
                elapsed = time.perf_counter() - started
                metrics.append(eval_snapshot(model, train_loader, val_loader, device, args, epoch, global_step, tokens_seen, elapsed, last_grad_norm))

        if not metrics or metrics[-1]["step"] != global_step:
            elapsed = time.perf_counter() - started
            metrics.append(eval_snapshot(model, train_loader, val_loader, device, args, epoch, global_step, tokens_seen, elapsed, last_grad_norm))

    elapsed_total = time.perf_counter() - started
    prompt_ids = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long, device=device).unsqueeze(0)
    generated_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        context_size=args.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.get_eos_id(),
    )[0].detach().cpu().tolist()
    sample = tokenizer.decode(generated_ids)
    parameter_count = sum(param.numel() for param in model.parameters())

    return {
        "args": args,
        "config": config,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "tokenizer_path": tokenizer_path,
        "tokenizer_elapsed": tokenizer_elapsed,
        "tokenizer_loaded": tokenizer_loaded,
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "parameter_count": parameter_count,
        "metrics": metrics,
        "elapsed_total": elapsed_total,
        "sample": sample,
    }


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def metrics_csv(metrics: list[dict]) -> str:
    output = io.StringIO()
    fieldnames = ["epoch", "step", "tokens_seen", "train_loss", "val_loss", "test_loss", "loss_gap", "lr", "grad_norm", "elapsed_sec", "tokens_per_sec"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in metrics:
        writer.writerow({key: fmt(row[key]) for key in fieldnames})
    return output.getvalue().strip()


def write_markdown(result: dict) -> None:
    args = result["args"]
    metrics = result["metrics"]
    final = metrics[-1]
    best = min(metrics, key=lambda row: row["val_loss"])
    perplexity = math.exp(final["val_loss"]) if final["val_loss"] < 20 else float("inf")

    lines = [
        f"# {args.experiment_id} {args.experiment_name}",
        "",
        "## 1. 실험 목적",
        "",
        args.purpose or "기준 모델의 사전학습 손실 곡선과 생성 샘플을 확보한다.",
        "",
        "## 2. 실행 설정",
        "",
        "| 항목 | 값 |",
        "| --- | --- |",
        f"| 변경점 | {args.change or 'baseline'} |",
        f"| seed | {args.seed} |",
        f"| vocab_size | {args.vocab_size} |",
        f"| context_length | {args.context_length} |",
        f"| emb_dim | {args.emb_dim} |",
        f"| n_heads | {args.n_heads} |",
        f"| n_layers | {args.n_layers} |",
        f"| ffn_mult | {args.ffn_mult} |",
        f"| activation | {args.activation} |",
        f"| drop_rate | {args.drop_rate} |",
        f"| qkv_bias | {args.qkv_bias} |",
        f"| weight_tying | {args.weight_tying} |",
        f"| norm_first | {args.norm_first} |",
        f"| batch_size | {args.batch_size} |",
        f"| num_epochs | {args.num_epochs} |",
        f"| lr | {args.lr} |",
        f"| weight_decay | {args.weight_decay} |",
        f"| eval_freq | {args.eval_freq} |",
        f"| eval_iter | {args.eval_iter} |",
        f"| stride | {args.stride or args.context_length} |",
        f"| amp | {args.amp} |",
        f"| device | {result['device_name']} |",
        f"| torch | {result['torch_version']} |",
        f"| parameter_count | {result['parameter_count']} |",
        f"| train_chars | {result['train_chars']} |",
        f"| val_chars | {result['val_chars']} |",
        f"| train_tokens | {result['train_tokens']} |",
        f"| val_tokens | {result['val_tokens']} |",
        f"| train_batches | {result['train_batches']} |",
        f"| val_batches | {result['val_batches']} |",
        f"| tokenizer_path | {result['tokenizer_path']} |",
        f"| tokenizer_loaded | {result['tokenizer_loaded']} |",
        f"| tokenizer_elapsed_sec | {result['tokenizer_elapsed']:.3f} |",
        "",
        "## 3. 결과 요약",
        "",
        "| 지표 | 값 |",
        "| --- | ---: |",
        f"| final_train_loss | {final['train_loss']:.6f} |",
        f"| final_val_loss | {final['val_loss']:.6f} |",
        f"| final_test_loss | {final['test_loss']:.6f} |",
        f"| final_loss_gap | {final['loss_gap']:.6f} |",
        f"| best_val_loss | {best['val_loss']:.6f} |",
        f"| best_step | {best['step']} |",
        f"| final_perplexity | {perplexity:.6f} |",
        f"| elapsed_sec | {result['elapsed_total']:.3f} |",
        f"| tokens_per_sec | {final['tokens_per_sec']:.3f} |",
        "",
        "## 4. Step별 지표",
        "",
        "`test_loss`는 별도 LM test split이 없기 때문에 held-out validation split의 loss를 같은 값으로 기록한 것이다. dropout 과적합 분석에서는 `loss_gap = test_loss - train_loss`를 중심으로 본다.",
        "",
        "| epoch | step | tokens_seen | train_loss | val_loss | test_loss | loss_gap | lr | grad_norm | elapsed_sec | tokens_per_sec |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in metrics:
        lines.append(
            "| {epoch} | {step} | {tokens_seen} | {train_loss:.6f} | {val_loss:.6f} | {test_loss:.6f} | {loss_gap:.6f} | {lr:.6g} | {grad_norm} | {elapsed_sec:.3f} | {tokens_per_sec:.3f} |".format(
                **{
                    **row,
                    "grad_norm": "" if row["grad_norm"] == "" else f"{row['grad_norm']:.6f}",
                }
            )
        )

    lines.extend(
        [
            "",
            "## 4-1. 그래프용 CSV",
            "",
            "아래 CSV 블록만 파싱하면 loss curve와 throughput curve를 그릴 수 있다.",
            "",
            "```csv",
            metrics_csv(metrics),
            "```",
            "",
            "## 5. 생성 샘플",
            "",
            f"prompt: `{args.prompt}`",
            "",
            "```text",
            result["sample"],
            "```",
            "",
            "## 6. Checkpoint",
            "",
            "저장하지 않음",
            "",
            "## 7. 해석",
            "",
            "baseline 결과다. 이후 실험은 이 md의 `final_val_loss`, `best_val_loss`, `tokens_per_sec`, 생성 샘플을 기준으로 비교한다.",
            "",
        ]
    )

    output_path = resolve_path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = train(args)
    write_markdown(result)
    print(resolve_path(args.output_md))


if __name__ == "__main__":
    main()
