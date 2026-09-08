"""CPU training/generation path with complete, reproducible restart state."""

from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .bpe import BPETokenizer
from .config import GPTConfig, set_seed
from .model import GPTModel
from .generation import generate

ROOT = Path(__file__).resolve().parents[1]


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


class TrainingRun:
    @classmethod
    def create(cls, train_text, validation_text, settings):
        settings = dict(settings)
        set_seed(settings.get("seed", 42), deterministic=True)
        tokenizer = BPETokenizer(vocab_size=settings["vocab_size"]).train(train_text)
        run = cls(settings, tokenizer, train_text, validation_text)
        return run

    def __init__(self, settings, tokenizer, train_text, validation_text):
        self.settings = dict(settings)
        self.tokenizer = tokenizer
        self.global_step = 0
        fields = GPTConfig.__dataclass_fields__
        config = {k: v for k, v in settings.items() if k in fields}
        config["vocab_size"] = len(tokenizer.id_to_token)
        self.model = GPTModel(GPTConfig(**config).to_dict())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=settings.get("learning_rate", 4e-4), weight_decay=settings.get("weight_decay", 0.1))
        self.train_tokens = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
        self.val_tokens = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)
        self.hashes = {"train": digest(train_text), "validation": digest(validation_text)}
        if min(len(self.train_tokens), len(self.val_tokens)) <= self.settings["context_length"]:
            raise ValueError("corpus must contain more tokens than context_length")
        if self.settings.get("batch_size", 8) < 1:
            raise ValueError("batch_size must be positive")

    def step(self):
        self.model.train()
        context = self.settings["context_length"]
        batch = self.settings.get("batch_size", 8)
        starts = torch.randint(len(self.train_tokens) - context, (batch,))
        x = torch.stack([self.train_tokens[s : s + context] for s in starts])
        y = torch.stack([self.train_tokens[s + 1 : s + context + 1] for s in starts])
        self.optimizer.zero_grad(set_to_none=True)
        loss, _ = self.model(x, targets=y)
        if not torch.isfinite(loss):
            raise ValueError("non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.global_step += 1
        return float(loss.detach())

    def validate(self, max_batches=20):
        was_training = self.model.training
        self.model.eval()
        total = 0.0
        tokens = 0
        context = self.settings["context_length"]
        batch = self.settings.get("batch_size", 8)
        try:
            with torch.no_grad():
                starts = list(range(0, len(self.val_tokens) - context, context))[: max_batches * batch]
                for offset in range(0, len(starts), batch):
                    group = starts[offset : offset + batch]
                    x = torch.stack([self.val_tokens[s : s + context] for s in group])
                    y = torch.stack([self.val_tokens[s + 1 : s + context + 1] for s in group])
                    loss, _ = self.model(x, targets=y)
                    total += float(loss) * y.numel()
                    tokens += y.numel()
        finally:
            self.model.train(was_training)
        if not tokens:
            raise ValueError("empty validation token set")
        return {"validation_loss": total / tokens, "evaluated_tokens": tokens}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        numpy_state = np.random.get_state()
        payload = {"format": 1, "settings": self.settings, "model_config": self.model.config, "tokenizer": self.tokenizer.to_dict(), "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "global_step": self.global_step, "hashes": self.hashes, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(), "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])}
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        temp.replace(path)

    @classmethod
    def load(cls, path, train_text, validation_text):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != 1:
            raise ValueError("unsupported checkpoint format")
        if payload["hashes"] != {"train": digest(train_text), "validation": digest(validation_text)}:
            raise ValueError("corpus changed since checkpoint")
        tokenizer = BPETokenizer().load_dict(payload["tokenizer"])
        run = cls(payload["settings"], tokenizer, train_text, validation_text)
        run.model.load_state_dict(payload["model"])
        run.optimizer.load_state_dict(payload["optimizer"])
        run.global_step = payload["global_step"]
        torch.set_rng_state(payload["torch_rng"])
        random.setstate(payload["python_rng"])
        state = payload["numpy_rng"]
        np.random.set_state((state[0], np.array(state[1], dtype="uint32"), *state[2:]))
        return run


def load_generator(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != 1:
        raise ValueError("unsupported checkpoint format")
    model = GPTModel(saved["model_config"])
    model.load_state_dict(saved["model"])
    model.eval()
    return model, BPETokenizer().load_dict(saved["tokenizer"]), saved


def sample(path, prompt, temperature, top_k, length, seed):
    if not prompt:
        raise ValueError("prompt must not be empty")
    if length < 0 or length > 2048:
        raise ValueError("length must be between 0 and 2048")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    model, tokenizer, saved = load_generator(path)
    set_seed(seed)
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    output = generate(model, ids, length, model.config["context_length"], temperature, top_k)
    return {"prompt": prompt, "text": tokenizer.decode(output[0].tolist()), "global_step": saved["global_step"], "seed": seed, "temperature": temperature, "top_k": top_k}


def main():
    parser = argparse.ArgumentParser(description="Small GPT: byte BPE → manual Transformer → train/resume/generate")
    parser.add_argument("--threads", type=int, default=2)
    subs = parser.add_subparsers(dest="command", required=True)
    train = subs.add_parser("train")
    train.add_argument("--config", default="configs/cpu.json")
    train.add_argument("--train", default="examples/train.txt")
    train.add_argument("--validation", default="examples/validation.txt")
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--checkpoint", default=".artifacts/checkpoint.pt")
    train.add_argument("--resume")
    train.add_argument("--save-every", type=int, default=25)
    gen = subs.add_parser("generate")
    gen.add_argument("--model", default="models/reference.pt")
    gen.add_argument("--prompt", default="A small model")
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--top-k", type=int, default=20)
    gen.add_argument("--length", type=int, default=60)
    gen.add_argument("--seed", type=int, default=42)
    subs.add_parser("demo")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    try:
        if args.command == "generate":
            print(json.dumps(sample(args.model, args.prompt, args.temperature, args.top_k, args.length, args.seed), ensure_ascii=False))
            return
        if args.command == "demo":
            for name in ("untrained", "reference"):
                for temperature, top_k in ((0.5, 5), (1.0, 20)):
                    print(json.dumps({"model": name, **sample(ROOT / f"models/{name}.pt", "A small model", temperature, top_k, 40, 42)}, ensure_ascii=False))
            return
        if args.steps < 0 or args.save_every < 1:
            raise ValueError("steps >= 0 and save-every >= 1 required")
        train_text = Path(args.train).read_text()
        validation = Path(args.validation).read_text()
        preparation_start = time.monotonic()
        print(json.dumps({"phase": "prepare", "resume": bool(args.resume), "train_bytes": len(train_text.encode("utf-8")), "note": "BPE training and corpus encoding may take minutes for NSMC."}), flush=True)
        if args.resume:
            run = TrainingRun.load(args.resume, train_text, validation)
        else:
            run = TrainingRun.create(train_text, validation, json.loads(Path(args.config).read_text()))
        print(json.dumps({"phase": "prepared", "preparation_seconds": time.monotonic() - preparation_start, "train_tokens": len(run.train_tokens), "validation_tokens": len(run.val_tokens)}), flush=True)
        if args.steps < run.global_step:
            raise ValueError("--steps is the final global step, not additional steps")
        start = time.monotonic()
        print(json.dumps({"step": run.global_step, **run.validate()}), flush=True)
        try:
            while run.global_step < args.steps:
                loss = run.step()
                if run.global_step % args.save_every == 0:
                    run.save(args.checkpoint)
                    print(json.dumps({"step": run.global_step, "train_batch_loss": loss, **run.validate()}), flush=True)
        except KeyboardInterrupt:
            run.save(args.checkpoint)
            print("Interrupted: checkpoint saved.", flush=True)
            return
        run.save(args.checkpoint)
        print(json.dumps({"checkpoint": args.checkpoint, "step": run.global_step, "seconds": time.monotonic() - start, **run.validate()}))
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"{error}\nPrepare models with make train. NSMC data: make data.\n")


if __name__ == "__main__":
    main()
