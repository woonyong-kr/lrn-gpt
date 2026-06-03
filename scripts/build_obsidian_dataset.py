#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a local pretraining corpus from an Obsidian vault.

The project pretraining path consumes plain text files. This script creates
train/validation text files plus JSONL sidecars for inspection and reuse.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable


DEFAULT_VAULT = Path("/Users/woonyong/workspace/vault")
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_DATASET_NAME = "obsidian_os_pintos_ai"
DEFAULT_VAL_RATIO = 0.08
DEFAULT_SEED = 42
DEFAULT_MAX_CHARS = 4_000
DEFAULT_MIN_CHARS = 240

INCLUDE_ROOTS = (
    "wiki/ai",
    "wiki/os",
    "wiki/pintos",
    ".legacy-backup/notes-ai",
    ".legacy-backup/notes-llm",
    ".legacy-backup/notes-os",
    ".legacy-backup/labs-os",
    ".legacy-backup/traces-os",
    ".legacy-backup/questions-os",
)

OPTIONAL_RELEVANT_ROOTS = (
    "maps",
    "ai-reference",
)

EXCLUDE_PARTS = {
    ".git",
    ".obsidian",
    ".trash",
    "node_modules",
    "attachments",
    "templates",
    "types",
}

EXCLUDE_PREFIXES = (
    "quartz/",
    "projects/",
    "inbox/",
    "sources/",
    "wiki/career/",
    "wiki/tools/",
)

PERSONAL_PATH_TERMS = {
    "career",
    "personal",
    "portfolio",
    "novel",
    "writing",
    "daily",
    "capture",
    "marine-bio",
    "data-collection",
    "데이터-개발자",
    "커리어",
    "개인",
    "소설",
    "포트폴리오",
}

MAP_NAME_TERMS = {
    "ai",
    "llm",
    "rag",
    "agent",
    "deep-learning",
    "neural",
    "transformer",
    "attention",
    "cnn",
    "rnn",
    "sequence",
    "os",
    "pintos",
    "thread",
    "process",
    "virtual-memory",
    "memory",
    "cpu",
    "qemu",
    "syscall",
    "user-program",
    "filesystem",
    "concept-to-code",
}

AI_KEYWORDS = (
    "ai",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "딥러닝",
    "머신러닝",
    "인공지능",
    "llm",
    "large language model",
    "gpt",
    "transformer",
    "attention",
    "self-attention",
    "tokenizer",
    "tokenization",
    "bpe",
    "embedding",
    "fine-tuning",
    "finetuning",
    "pretraining",
    "rag",
    "retrieval augmented generation",
    "agent",
    "vector database",
    "rlhf",
    "dpo",
    "lora",
    "sft",
    "cnn",
    "rnn",
    "vision transformer",
)

OS_KEYWORDS = (
    "os",
    "operating system",
    "운영체제",
    "kernel",
    "커널",
    "pintos",
    "qemu",
    "thread",
    "스레드",
    "scheduler",
    "스케줄러",
    "process",
    "프로세스",
    "interrupt",
    "인터럽트",
    "syscall",
    "system call",
    "page fault",
    "페이지 폴트",
    "virtual memory",
    "가상 메모리",
    "page table",
    "페이지 테이블",
    "pml4",
    "mmu",
    "tlb",
    "fork",
    "exec",
    "mmap",
    "swap",
    "inode",
    "file descriptor",
    "파일 디스크립터",
    "semaphore",
    "lock",
    "condition variable",
)

OBSIDIAN_LINK_RE = re.compile(r"!?\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]")
IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
MULTI_BLANK_RE = re.compile(r"\n{3,}")
SPACE_RE = re.compile(r"[ \t]+")

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?82[-.\s]?)?0?1[016789][-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)")
LOCAL_PATH_RE = re.compile(r"/Users/[A-Za-z0-9._-]+/[^\s)`,'\"]+")
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|secret|password|passwd|access[_ -]?token|refresh[_ -]?token|auth[_ -]?token|client[_ -]?secret)\b"
    r"\s*[:=]\s*['\"]?[^'\"\s`]{8,}"
)
BEARER_RE = re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]{10,}")
SECRET_VALUE_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{20,})\b"
)
PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
KOREAN_PASSWORD_RE = re.compile(r"(비밀번호|패스워드)\s*[:=]\s*[^\s`]{4,}")
ID_NUMBER_RE = re.compile(r"\b\d{6}-[1-4]\d{6}\b")


@dataclass(frozen=True)
class Document:
    path: Path
    rel_path: str
    title: str
    topics: tuple[str, ...]
    text: str
    source: str
    sha256: str
    redactions: int


@dataclass(frozen=True)
class CandidateResult:
    path: Path
    include: bool
    reason: str
    topics: tuple[str, ...]
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def starts_with_any(value: str, prefixes: Iterable[str]) -> bool:
    return any(value == prefix.rstrip("/") or value.startswith(prefix) for prefix in prefixes)


def has_excluded_part(path: Path) -> bool:
    return any(part in EXCLUDE_PARTS for part in path.parts)


def has_personal_path_term(rel_path: str) -> bool:
    lower = rel_path.lower()
    return any(term.lower() in lower for term in PERSONAL_PATH_TERMS)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    if not text.startswith("---\n"):
        return {}, text
    lines = text.splitlines()
    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return {}, text

    raw_meta = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    meta: dict[str, object] = {}
    current_key: str | None = None

    for line in raw_meta:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- ") and current_key:
            meta.setdefault(current_key, [])
            value = stripped[2:].strip().strip('"')
            if isinstance(meta[current_key], list):
                meta[current_key].append(value)
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            current_key = key.strip()
            value = value.strip()
            if value:
                meta[current_key] = value.strip('"')
            else:
                meta[current_key] = []

    return meta, body


def title_from(path: Path, metadata: dict[str, object], body: str) -> str:
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    for line in body.splitlines():
        if line.startswith("# "):
            return line.lstrip("# ").strip()
    return path.stem.replace("-", " ")


def metadata_blob(metadata: dict[str, object]) -> str:
    pieces: list[str] = []
    for key in ("type", "title", "parent_moc", "status", "tags", "related_to"):
        value = metadata.get(key)
        if isinstance(value, list):
            pieces.extend(str(item) for item in value)
        elif value is not None:
            pieces.append(str(value))
    return "\n".join(pieces)


def topic_hits(text: str, keywords: tuple[str, ...]) -> int:
    lower = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lower)


def topics_for(rel_path: str, metadata: dict[str, object], body: str) -> tuple[str, ...]:
    lower_path = rel_path.lower()
    sample = f"{lower_path}\n{metadata_blob(metadata)}\n{body[:6000]}"
    topics: set[str] = set()

    if "pintos" in lower_path or "pintos" in sample.lower():
        topics.add("pintos")
    if lower_path.startswith("wiki/os/") or topic_hits(sample, OS_KEYWORDS) >= 2:
        topics.add("os")
    if lower_path.startswith("wiki/ai/") or topic_hits(sample, AI_KEYWORDS) >= 2:
        topics.add("ai")
    if "llm" in lower_path or any(word in sample.lower() for word in ("llm", "gpt", "transformer", "tokenizer", "rag")):
        topics.add("llm")
    if "deep-learning" in lower_path or "deep learning" in sample.lower() or "딥러닝" in sample:
        topics.add("deep-learning")

    return tuple(sorted(topics))


def source_for(rel_path: str) -> str:
    if rel_path.startswith(".legacy-backup/"):
        return "legacy-backup"
    if rel_path.startswith("wiki/"):
        return "wiki"
    if rel_path.startswith("maps/"):
        return "map"
    if rel_path.startswith("ai-reference/"):
        return "ai-reference"
    return "vault"


def candidate_for(path: Path, vault: Path) -> CandidateResult:
    rel_path = rel_posix(path, vault)
    if path.suffix.lower() not in {".md", ".markdown"}:
        return CandidateResult(path, False, "not_markdown", (), source_for(rel_path))
    if has_excluded_part(path):
        return CandidateResult(path, False, "excluded_part", (), source_for(rel_path))
    if starts_with_any(rel_path, EXCLUDE_PREFIXES):
        return CandidateResult(path, False, "excluded_prefix", (), source_for(rel_path))
    if has_personal_path_term(rel_path):
        return CandidateResult(path, False, "personal_path", (), source_for(rel_path))

    raw = read_text(path)
    metadata, body = split_frontmatter(raw)
    topics = topics_for(rel_path, metadata, body)
    source = source_for(rel_path)

    if starts_with_any(rel_path, INCLUDE_ROOTS):
        return CandidateResult(path, bool(topics), "included_root" if topics else "root_without_topic", topics, source)

    if starts_with_any(rel_path, OPTIONAL_RELEVANT_ROOTS):
        name = path.stem.lower()
        if any(term in name for term in MAP_NAME_TERMS) or len(topics) >= 1:
            return CandidateResult(path, bool(topics), "relevant_optional_root" if topics else "optional_without_topic", topics, source)
        return CandidateResult(path, False, "optional_not_relevant", topics, source)

    if len(topics) >= 2:
        return CandidateResult(path, True, "keyword_match", topics, source)

    return CandidateResult(path, False, "not_relevant", topics, source)


def clean_obsidian_markdown(text: str) -> str:
    text = HTML_COMMENT_RE.sub("", text)
    text = IMAGE_LINK_RE.sub("", text)

    def replace_link(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        alias = match.group(2)
        if match.group(0).startswith("!"):
            return ""
        return alias.strip() if alias else target

    text = OBSIDIAN_LINK_RE.sub(replace_link, text)
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("%%") and stripped.endswith("%%"):
            continue
        if stripped.startswith("> [!"):
            line = re.sub(r"^>\s*\[![^\]]+\]\s*", "> ", line)
        cleaned_lines.append(SPACE_RE.sub(" ", line).rstrip())
    text = "\n".join(cleaned_lines).strip()
    return MULTI_BLANK_RE.sub("\n\n", text)


def redact_sensitive(text: str) -> tuple[str, int, bool]:
    if PRIVATE_KEY_RE.search(text):
        return text, 0, True

    redactions = 0

    def replace(pattern: re.Pattern[str], repl: str, value: str) -> str:
        nonlocal redactions
        value, count = pattern.subn(repl, value)
        redactions += count
        return value

    text = replace(SECRET_ASSIGNMENT_RE, r"\1=[REDACTED_SECRET]", text)
    text = replace(BEARER_RE, "authorization: Bearer [REDACTED_SECRET]", text)
    text = replace(SECRET_VALUE_RE, "[REDACTED_SECRET]", text)
    text = replace(KOREAN_PASSWORD_RE, r"\1=[REDACTED_SECRET]", text)
    text = replace(ID_NUMBER_RE, "[REDACTED_ID_NUMBER]", text)
    text = replace(EMAIL_RE, "[REDACTED_EMAIL]", text)
    text = replace(PHONE_RE, "[REDACTED_PHONE]", text)
    text = replace(LOCAL_PATH_RE, "[REDACTED_LOCAL_PATH]", text)
    return text, redactions, False


def redact_local_path(text: str) -> str:
    return LOCAL_PATH_RE.sub("[REDACTED_LOCAL_PATH]", text)


def normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_document(result: CandidateResult, vault: Path, min_chars: int) -> tuple[Document | None, str, int]:
    rel_path = rel_posix(result.path, vault)
    raw = read_text(result.path)
    metadata, body = split_frontmatter(raw)
    title = title_from(result.path, metadata, body)
    cleaned = clean_obsidian_markdown(body)
    cleaned, redactions, skip_secret = redact_sensitive(cleaned)
    if skip_secret:
        return None, "private_key", redactions
    if len(cleaned) < min_chars:
        return None, "too_short", redactions

    doc_text = f"# {title}\n\n{cleaned}".strip()
    digest = normalized_hash(doc_text)
    return (
        Document(
            path=result.path,
            rel_path=rel_path,
            title=title,
            topics=result.topics,
            text=doc_text,
            source=result.source,
            sha256=digest,
            redactions=redactions,
        ),
        "included",
        redactions,
    )


def chunk_document(document: Document, max_chars: int, min_chars: int) -> list[dict[str, object]]:
    sections = split_sections(document.text)
    chunks: list[str] = []
    buffer = ""

    for section in sections:
        if len(section) > max_chars:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(split_long_text(section, max_chars))
            continue
        if buffer and len(buffer) + len(section) + 2 > max_chars:
            chunks.append(buffer.strip())
            buffer = section
        else:
            buffer = section if not buffer else f"{buffer}\n\n{section}"

    if buffer:
        chunks.append(buffer.strip())

    rows: list[dict[str, object]] = []
    for idx, chunk in enumerate(chunks):
        if len(chunk) < min_chars and idx > 0:
            continue
        rows.append(
            {
                "id": f"{document.sha256[:12]}-{idx:03d}",
                "source_path": document.rel_path,
                "title": document.title,
                "topics": list(document.topics),
                "source": document.source,
                "chunk_index": idx,
                "text": chunk,
                "char_count": len(chunk),
                "sha256": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
            }
        )
    return rows


def split_sections(text: str) -> list[str]:
    sections: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [section for section in sections if section]


def split_long_text(text: str, max_chars: int) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                pieces.append(current.strip())
                current = ""
            pieces.extend(paragraph[i : i + max_chars].strip() for i in range(0, len(paragraph), max_chars))
            continue
        if current and len(current) + len(paragraph) + 2 > max_chars:
            pieces.append(current.strip())
            current = paragraph
        else:
            current = paragraph if not current else f"{current}\n\n{paragraph}"
    if current:
        pieces.append(current.strip())
    return [piece for piece in pieces if piece]


def format_lm_document(document: Document) -> str:
    topics = ", ".join(document.topics)
    return (
        "<|document_start|>\n"
        f"source_path: {document.rel_path}\n"
        f"title: {document.title}\n"
        f"topics: {topics}\n\n"
        f"{document.text}\n"
        "<|document_end|>"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_text(path: Path, documents: list[Document]) -> int:
    content = "\n\n".join(format_lm_document(document) for document in documents)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")
    return len(content)


def document_row(document: Document) -> dict[str, object]:
    return {
        "source_path": document.rel_path,
        "title": document.title,
        "topics": list(document.topics),
        "source": document.source,
        "char_count": len(document.text),
        "sha256": document.sha256,
        "redactions": document.redactions,
        "text": document.text,
    }


def collect_documents(vault: Path, min_chars: int) -> tuple[list[Document], dict[str, object]]:
    skip_reasons: Counter[str] = Counter()
    candidate_reasons: Counter[str] = Counter()
    redactions_total = 0
    seen: dict[str, str] = {}
    documents: list[Document] = []

    markdown_files = sorted(vault.rglob("*.md")) + sorted(vault.rglob("*.markdown"))
    for path in markdown_files:
        result = candidate_for(path, vault)
        candidate_reasons[result.reason] += 1
        if not result.include:
            skip_reasons[result.reason] += 1
            continue

        document, reason, redactions = build_document(result, vault, min_chars)
        redactions_total += redactions
        if document is None:
            skip_reasons[reason] += 1
            continue
        duplicate_of = seen.get(document.sha256)
        if duplicate_of is not None:
            skip_reasons["duplicate"] += 1
            continue
        seen[document.sha256] = document.rel_path
        documents.append(document)

    stats = {
        "candidates_seen": len(markdown_files),
        "candidate_reasons": dict(sorted(candidate_reasons.items())),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "redactions": redactions_total,
    }
    return documents, stats


def split_train_val(documents: list[Document], val_ratio: float, seed: int) -> tuple[list[Document], list[Document]]:
    docs = list(documents)
    rng = random.Random(seed)
    rng.shuffle(docs)
    if not docs:
        return [], []
    val_count = max(1, int(len(docs) * val_ratio))
    val_docs = sorted(docs[:val_count], key=lambda document: document.rel_path)
    train_docs = sorted(docs[val_count:], key=lambda document: document.rel_path)
    return train_docs, val_docs


def build_manifest(
    *,
    vault: Path,
    dataset_name: str,
    documents: list[Document],
    train_docs: list[Document],
    val_docs: list[Document],
    chunks: list[dict[str, object]],
    stats: dict[str, object],
    outputs: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, object]:
    topic_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for document in documents:
        topic_counts.update(document.topics)
        source_counts[document.source] += 1

    chars_by_split = {
        "train": sum(len(document.text) for document in train_docs),
        "val": sum(len(document.text) for document in val_docs),
    }
    docs_by_topic_source: dict[str, dict[str, int]] = defaultdict(dict)
    for document in documents:
        for topic in document.topics:
            docs_by_topic_source[topic][document.source] = docs_by_topic_source[topic].get(document.source, 0) + 1

    return {
        "dataset_name": dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vault": redact_local_path(str(vault)),
        "selection": {
            "include_roots": list(INCLUDE_ROOTS),
            "optional_relevant_roots": list(OPTIONAL_RELEVANT_ROOTS),
            "excluded_prefixes": list(EXCLUDE_PREFIXES),
            "excluded_parts": sorted(EXCLUDE_PARTS),
            "personal_path_terms": sorted(PERSONAL_PATH_TERMS),
        },
        "parameters": {
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "max_chars": args.max_chars,
            "min_chars": args.min_chars,
        },
        "counts": {
            "documents": len(documents),
            "train_documents": len(train_docs),
            "val_documents": len(val_docs),
            "chunks": len(chunks),
            "topic_counts": dict(sorted(topic_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "docs_by_topic_source": {key: dict(sorted(value.items())) for key, value in sorted(docs_by_topic_source.items())},
            "chars_by_split": chars_by_split,
            "total_chars": sum(chars_by_split.values()),
        },
        "privacy": {
            "redactions": stats["redactions"],
            "secret_policy": "Redact high-confidence credentials and local user paths; skip private-key files.",
            "excluded_personal_areas": [
                "career",
                "daily/inbox",
                "fiction/writing projects",
                "portfolio/novel maps",
                "raw sources/clippings",
                "Quartz build/output tree",
            ],
        },
        "skips": stats["skip_reasons"],
        "candidate_reasons": stats["candidate_reasons"],
        "outputs": outputs,
    }


def main() -> None:
    args = parse_args()
    vault = args.vault.expanduser().resolve()
    output_dir = args.output_dir
    dataset_name = args.dataset_name

    if not vault.exists():
        raise SystemExit(f"Vault not found: {vault}")
    if not 0.0 < args.val_ratio < 0.5:
        raise SystemExit("--val-ratio must be between 0 and 0.5")
    if args.max_chars < args.min_chars:
        raise SystemExit("--max-chars must be >= --min-chars")

    documents, stats = collect_documents(vault, min_chars=args.min_chars)
    train_docs, val_docs = split_train_val(documents, args.val_ratio, args.seed)
    chunks = [chunk for document in documents for chunk in chunk_document(document, args.max_chars, args.min_chars)]

    output_paths = {
        "lm_train": output_dir / f"{dataset_name}_lm_train.txt",
        "lm_val": output_dir / f"{dataset_name}_lm_val.txt",
        "documents_jsonl": output_dir / f"{dataset_name}_documents.jsonl",
        "chunks_jsonl": output_dir / f"{dataset_name}_chunks.jsonl",
        "manifest": output_dir / f"{dataset_name}_manifest.json",
    }
    outputs = {key: str(path) for key, path in output_paths.items()}
    manifest = build_manifest(
        vault=vault,
        dataset_name=dataset_name,
        documents=documents,
        train_docs=train_docs,
        val_docs=val_docs,
        chunks=chunks,
        stats=stats,
        outputs=outputs,
        args=args,
    )

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_text(output_paths["lm_train"], train_docs)
        write_text(output_paths["lm_val"], val_docs)
        write_jsonl(output_paths["documents_jsonl"], (document_row(document) for document in documents))
        write_jsonl(output_paths["chunks_jsonl"], chunks)
        output_paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps({"privacy": manifest["privacy"], "skips": manifest["skips"]}, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        print("Dry run complete. No files written.")
    else:
        print("Wrote:")
        for key, path in outputs.items():
            print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
