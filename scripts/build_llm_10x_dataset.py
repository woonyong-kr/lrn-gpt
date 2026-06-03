#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a duplicate-free, LLM-heavy 10x pretraining corpus.

This extends the Obsidian OS/PintOS/AI corpus with LLM implementation material
from local project files and explicitly licensed public source repositories.
The resulting LM text is chunk-based so exact duplicate chunks cannot enter the
training stream.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from build_obsidian_dataset import (
    DEFAULT_MIN_CHARS,
    Document,
    chunk_document,
    collect_documents,
    redact_local_path,
    redact_sensitive,
    split_train_val,
)


DEFAULT_DATASET_NAME = "obsidian_llm_10x"
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_BASE_DATASET = (
    Path("data/nsmc_lm_train.txt"),
    Path("data/nsmc_lm_val.txt"),
)
DEFAULT_TARGET_MULTIPLIER = 10.0
DEFAULT_TARGET_CHARS_FALLBACK = 15_000_000
DEFAULT_MAX_CHARS = 4_000
DEFAULT_VAL_RATIO = 0.08
DEFAULT_SEED = 42

VAULT_ROOT = Path("/Users/woonyong/workspace/vault")
PROJECT_ROOT = Path("/Users/woonyong/workspace/Krafton-Jungle/SW_AI-W13-gpt")
EXTERNAL_ROOT = PROJECT_ROOT / "local" / "llm_sources"

ALLOWED_EXTENSIONS = {".md", ".mdx", ".py", ".ipynb"}
PROJECT_ALLOWED_EXTENSIONS = {".md", ".py"}

SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    ".ipynb_checkpoints",
    "data",
    "experiments",
    "output",
    "outputs",
    "images",
    "img",
    "assets",
    "media",
    "translations",
    "nbs",
}

PROJECT_INCLUDE_PREFIXES = (
    "README.md",
    "REPORT.md",
    "docs/woonyong/",
    "src/",
    "tests/",
)

COOKBOOK_KEYWORDS = (
    "rag",
    "retrieval",
    "embedding",
    "embeddings",
    "fine",
    "finetun",
    "eval",
    "chat",
    "responses",
    "structured",
    "function",
    "tool",
    "agents",
    "reasoning",
    "prompt",
    "gpt",
    "llm",
    "model",
)

TOPIC_KEYWORDS = {
    "llm": ("llm", "gpt", "language model", "transformer", "chat", "responses"),
    "from-scratch": ("from-scratch", "scratch", "nanogpt", "build", "gpt.py"),
    "tokenizer": ("token", "tokenizer", "bpe", "vocab", "encoding"),
    "attention": ("attention", "transformer", "kv-cache", "gqa", "mha", "causal"),
    "pretraining": ("pretrain", "training", "train", "dataloader", "dataset"),
    "finetuning": ("fine", "finetune", "sft", "instruction", "classification", "dpo", "preference"),
    "rag": ("rag", "retrieval", "embedding", "vector", "rerank"),
    "evaluation": ("eval", "benchmark", "judge", "metric"),
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    root: Path
    license: str
    include_prefixes: tuple[str, ...] = ()
    path_keywords: tuple[str, ...] = ()
    allowed_extensions: frozenset[str] = frozenset(ALLOWED_EXTENSIONS)
    max_files: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-multiplier", type=float, default=DEFAULT_TARGET_MULTIPLIER)
    parser.add_argument("--target-chars", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalized_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def normalized_sha(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def char_count(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        if path.exists():
            total += len(path.read_text(encoding="utf-8"))
    return total


def target_chars(args: argparse.Namespace) -> int:
    if args.target_chars is not None:
        return args.target_chars
    base_chars = char_count(DEFAULT_BASE_DATASET)
    if base_chars == 0:
        return DEFAULT_TARGET_CHARS_FALLBACK
    return int(base_chars * args.target_multiplier)


def should_skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def project_docs() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in PROJECT_ALLOWED_EXTENSIONS:
            continue
        if should_skip(path):
            continue
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if not any(rel == prefix or rel.startswith(prefix) for prefix in PROJECT_INCLUDE_PREFIXES):
            continue
        text = read_source_file(path)
        if len(text) < DEFAULT_MIN_CHARS:
            continue
        title = rel
        body = f"# {title}\n\n{text}".strip()
        digest = normalized_sha(body)
        topics = infer_topics(rel, body)
        documents.append(
            Document(
                path=path,
                rel_path=f"project/{rel}",
                title=title,
                topics=topics,
                text=body,
                source="project",
                sha256=digest,
                redactions=0,
            )
        )
    return documents


def external_specs() -> list[SourceSpec]:
    return [
        SourceSpec(
            name="rasbt-llms-from-scratch",
            root=EXTERNAL_ROOT / "LLMs-from-scratch",
            license="Apache-2.0 source code/docs only; book text and images excluded",
        ),
        SourceSpec(
            name="karpathy-nanogpt",
            root=EXTERNAL_ROOT / "nanoGPT",
            license="MIT",
        ),
        SourceSpec(
            name="openai-cookbook",
            root=EXTERNAL_ROOT / "openai-cookbook",
            license="MIT",
            path_keywords=COOKBOOK_KEYWORDS,
        ),
        SourceSpec(
            name="huggingface-course-en",
            root=EXTERNAL_ROOT / "huggingface-course",
            license="Apache-2.0",
            include_prefixes=("chapters/en/", "README.md"),
        ),
    ]


def external_docs(spec: SourceSpec) -> list[Document]:
    if not spec.root.exists():
        return []
    documents: list[Document] = []
    files_seen = 0
    for path in sorted(spec.root.rglob("*")):
        if spec.max_files is not None and files_seen >= spec.max_files:
            break
        if not path.is_file() or path.suffix.lower() not in spec.allowed_extensions:
            continue
        if should_skip(path):
            continue
        rel = path.relative_to(spec.root).as_posix()
        if spec.include_prefixes and not any(rel == prefix or rel.startswith(prefix) for prefix in spec.include_prefixes):
            continue
        if spec.path_keywords and not any(keyword in rel.lower() for keyword in spec.path_keywords):
            continue
        text = read_source_file(path)
        text, redactions, skip_secret = redact_sensitive(text)
        if skip_secret or len(text) < DEFAULT_MIN_CHARS:
            continue
        files_seen += 1
        title = f"{spec.name}/{rel}"
        body = f"# {title}\n\n{text}".strip()
        digest = normalized_sha(body)
        topics = infer_topics(rel, body)
        documents.append(
            Document(
                path=path,
                rel_path=f"external/{spec.name}/{rel}",
                title=title,
                topics=topics,
                text=body,
                source=spec.name,
                sha256=digest,
                redactions=redactions,
            )
        )
    return documents


def read_source_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".ipynb":
        return read_notebook(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".mdx":
        text = strip_mdx_noise(text)
    return text.strip()


def read_notebook(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return ""
    pieces: list[str] = []
    for idx, cell in enumerate(payload.get("cells", [])):
        cell_type = cell.get("cell_type", "cell")
        source = cell.get("source", [])
        if isinstance(source, list):
            text = "".join(str(part) for part in source)
        else:
            text = str(source)
        text = text.strip()
        if not text:
            continue
        if cell_type == "code":
            pieces.append(f"```python\n{text}\n```")
        else:
            pieces.append(text)
    return "\n\n".join(pieces)


def strip_mdx_noise(text: str) -> str:
    text = re.sub(r"import\s+.*?from\s+['\"][^'\"]+['\"];?", "", text)
    text = re.sub(r"<Youtube[^>]*>", "", text)
    text = re.sub(r"</?Tip[^>]*>", "", text)
    text = re.sub(r"</?FrameworkSwitchCourse[^>]*>", "", text)
    text = re.sub(r"</?FrameworkContent[^>]*>", "", text)
    return text.strip()


def infer_topics(path: str, text: str) -> tuple[str, ...]:
    sample = f"{path}\n{text[:8000]}".lower()
    topics = {"llm"}
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in sample for keyword in keywords):
            topics.add(topic)
    return tuple(sorted(topics))


def dedupe_documents(documents: Iterable[Document]) -> tuple[list[Document], Counter[str]]:
    seen: set[str] = set()
    unique: list[Document] = []
    skips: Counter[str] = Counter()
    for document in documents:
        digest = normalized_sha(document.text)
        if digest in seen:
            skips["duplicate_document"] += 1
            continue
        seen.add(digest)
        unique.append(document)
    return unique, skips


def unique_chunks(
    documents: Iterable[Document],
    *,
    max_chars: int,
    min_chars: int,
    target_chars: int,
) -> tuple[list[dict[str, object]], Counter[str]]:
    seen: set[str] = set()
    chunks: list[dict[str, object]] = []
    skips: Counter[str] = Counter()
    total_chars = 0
    for document in documents:
        for chunk in chunk_document(document, max_chars=max_chars, min_chars=min_chars):
            digest = normalized_sha(str(chunk["text"]))
            if digest in seen:
                skips["duplicate_chunk"] += 1
                continue
            seen.add(digest)
            chunk = dict(chunk)
            chunk["normalized_sha256"] = digest
            chunk["license"] = license_for_source(str(chunk["source"]))
            chunks.append(chunk)
            total_chars += len(str(chunk["text"]))
            if total_chars >= target_chars:
                return chunks, skips
    return chunks, skips


def license_for_source(source: str) -> str:
    if source == "project":
        return "local project"
    if source == "wiki" or source == "map" or source == "ai-reference" or source == "vault" or source == "legacy-backup":
        return "local Obsidian vault"
    for spec in external_specs():
        if source == spec.name:
            return spec.license
    return "unknown"


def format_lm_chunk(chunk: dict[str, object]) -> str:
    topics = ", ".join(str(topic) for topic in chunk.get("topics", []))
    return (
        "<|chunk_start|>\n"
        f"source_path: {chunk['source_path']}\n"
        f"title: {chunk['title']}\n"
        f"topics: {topics}\n"
        f"license: {chunk.get('license', '')}\n\n"
        f"{chunk['text']}\n"
        "<|chunk_end|>"
    )


def split_chunks(chunks: list[dict[str, object]], val_ratio: float, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pseudo_docs = [
        Document(
            path=Path(str(chunk["source_path"])),
            rel_path=str(chunk["id"]),
            title=str(chunk["title"]),
            topics=tuple(str(topic) for topic in chunk.get("topics", [])),
            text=str(chunk["text"]),
            source=str(chunk["source"]),
            sha256=str(chunk["normalized_sha256"]),
            redactions=0,
        )
        for chunk in chunks
    ]
    train_docs, val_docs = split_train_val(pseudo_docs, val_ratio=val_ratio, seed=seed)
    train_ids = {doc.rel_path for doc in train_docs}
    val_ids = {doc.rel_path for doc in val_docs}
    train = [chunk for chunk in chunks if str(chunk["id"]) in train_ids]
    val = [chunk for chunk in chunks if str(chunk["id"]) in val_ids]
    return train, val


def write_lm(path: Path, chunks: list[dict[str, object]]) -> int:
    content = "\n\n".join(format_lm_chunk(chunk) for chunk in chunks)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")
    return len(content)


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def document_row(document: Document) -> dict[str, object]:
    return {
        "source_path": document.rel_path,
        "title": document.title,
        "topics": list(document.topics),
        "source": document.source,
        "license": license_for_source(document.source),
        "char_count": len(document.text),
        "normalized_sha256": normalized_sha(document.text),
        "redactions": document.redactions,
        "text": document.text,
    }


def collect_all_documents(min_chars: int) -> tuple[list[Document], Counter[str]]:
    obsidian_docs, obsidian_stats = collect_documents(VAULT_ROOT, min_chars=min_chars)
    all_documents: list[Document] = []
    all_documents.extend(obsidian_docs)
    all_documents.extend(project_docs())
    for spec in external_specs():
        all_documents.extend(external_docs(spec))
    unique_documents, skips = dedupe_documents(all_documents)
    skips.update(obsidian_stats.get("skip_reasons", {}))
    return unique_documents, skips


def build_manifest(
    *,
    args: argparse.Namespace,
    target: int,
    documents: list[Document],
    chunks: list[dict[str, object]],
    train_chunks: list[dict[str, object]],
    val_chunks: list[dict[str, object]],
    skips: Counter[str],
    outputs: dict[str, str],
) -> dict[str, object]:
    source_counts: Counter[str] = Counter(document.source for document in documents)
    topic_counts: Counter[str] = Counter()
    for chunk in chunks:
        topic_counts.update(str(topic) for topic in chunk.get("topics", []))
    return {
        "dataset_name": args.dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_chars": target,
        "counts": {
            "documents": len(documents),
            "chunks": len(chunks),
            "train_chunks": len(train_chunks),
            "val_chunks": len(val_chunks),
            "chunk_text_chars": sum(len(str(chunk["text"])) for chunk in chunks),
            "train_text_chars": sum(len(str(chunk["text"])) for chunk in train_chunks),
            "val_text_chars": sum(len(str(chunk["text"])) for chunk in val_chunks),
            "source_counts": dict(sorted(source_counts.items())),
            "topic_counts": dict(sorted(topic_counts.items())),
        },
        "dedupe_policy": {
            "document": "normalized full document text SHA-256 must be unique",
            "chunk": "normalized chunk text SHA-256 must be unique; duplicate chunks are skipped before LM text writing",
            "duplicates_allowed": False,
        },
        "sources": [
            {
                "name": "local Obsidian vault",
                "path": redact_local_path(str(VAULT_ROOT)),
                "license": "local user notes",
            },
            {
                "name": "current GPT project",
                "path": redact_local_path(str(PROJECT_ROOT)),
                "license": "local project",
            },
            *[
                {
                    "name": spec.name,
                    "path": redact_local_path(str(spec.root)),
                    "license": spec.license,
                }
                for spec in external_specs()
            ],
        ],
        "skips": dict(sorted(skips.items())),
        "outputs": outputs,
    }


def main() -> None:
    args = parse_args()
    target = target_chars(args)
    output_dir = args.output_dir
    paths = {
        "lm_train": output_dir / f"{args.dataset_name}_lm_train.txt",
        "lm_val": output_dir / f"{args.dataset_name}_lm_val.txt",
        "documents_jsonl": output_dir / f"{args.dataset_name}_documents.jsonl",
        "chunks_jsonl": output_dir / f"{args.dataset_name}_chunks.jsonl",
        "manifest": output_dir / f"{args.dataset_name}_manifest.json",
    }
    outputs = {key: str(value) for key, value in paths.items()}

    documents, skips = collect_all_documents(min_chars=args.min_chars)
    chunks, chunk_skips = unique_chunks(documents, max_chars=args.max_chars, min_chars=args.min_chars, target_chars=target)
    skips.update(chunk_skips)
    train_chunks, val_chunks = split_chunks(chunks, val_ratio=args.val_ratio, seed=args.seed)
    manifest = build_manifest(
        args=args,
        target=target,
        documents=documents,
        chunks=chunks,
        train_chunks=train_chunks,
        val_chunks=val_chunks,
        skips=skips,
        outputs=outputs,
    )

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_lm(paths["lm_train"], train_chunks)
        write_lm(paths["lm_val"], val_chunks)
        write_jsonl(paths["documents_jsonl"], (document_row(document) for document in documents))
        write_jsonl(paths["chunks_jsonl"], chunks)
        paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps({"target_chars": target, "dedupe_policy": manifest["dedupe_policy"], "skips": manifest["skips"]}, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.dry_run:
        print("Wrote:")
        for key, value in outputs.items():
            print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
