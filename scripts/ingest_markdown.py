"""Migrate an existing Markdown memory scaffold into the engine.

Reads a directory of `.md` files (e.g. the agent-memory/ folder from the
Markdown scaffold convention in scaffold/) and writes one memory per `##` section,
guessing the memory type from the filename.

    python scripts/ingest_markdown.py path/to/agent-memory/ [--path store.json]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_memory import MEMORY_TYPES, MemoryStore, count_tokens  # noqa: E402

# Map common scaffold filenames to memory types.
_FILE_TYPE = {
    "project": "project",
    "decisions": "decision",
    "known_issues": "issue",
    "state": "state",
    "handoff": "handoff",
    "worklog": "worklog",
}


def type_for(filename: str) -> str:
    stem = filename.lower().removesuffix(".md")
    return _FILE_TYPE.get(stem, "fact")


def sections(text: str, max_tokens: int = 120) -> list[str]:
    """Split a Markdown file into `##` sections, dropping headings and blanks.

    Long sections are split further on blank lines. A memory bigger than a
    typical recall budget can never be packed into one, so it would sit in the
    store permanently unreachable — chunking keeps every ingested memory
    retrievable.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    chunks = [c.strip() for c in re.split(r"^##\s+.*$", text, flags=re.MULTILINE)]
    out: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        if count_tokens(chunk) <= max_tokens:
            out.append(chunk)
            continue
        buffer: list[str] = []
        for para in re.split(r"\n\s*\n", chunk):
            para = para.strip()
            if not para:
                continue
            candidate = "\n\n".join(buffer + [para])
            if buffer and count_tokens(candidate) > max_tokens:
                out.append("\n\n".join(buffer))
                buffer = [para]
            else:
                buffer.append(para)
        if buffer:
            out.append("\n\n".join(buffer))
    # A single paragraph (including a long code block) can itself exceed the
    # cap. Split at Python character boundaries, preferring whitespace, so we
    # neither drop content nor cut UTF-8 bytes in half.
    bounded = []
    for chunk in out:
        while count_tokens(chunk) > max_tokens:
            low, high = 0, len(chunk)
            while low < high:
                middle = (low + high + 1) // 2
                if count_tokens(chunk[:middle]) <= max_tokens:
                    low = middle
                else:
                    high = middle - 1
            if low == 0:
                raise ValueError("max_tokens is too small for a source character")
            boundary = max(chunk.rfind(" ", 0, low), chunk.rfind("\n", 0, low))
            end = boundary if boundary > low // 2 else low
            bounded.append(chunk[:end].strip())
            chunk = chunk[end:].strip()
        if chunk:
            bounded.append(chunk)
    return bounded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory of .md memory files")
    parser.add_argument("--path", type=Path, default=None, help="output store.json")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=120,
        help="split sections larger than this so they stay recallable",
    )
    args = parser.parse_args()

    if args.path is None:
        parser.error(
            "--path is required: without it the ingested memories would be built "
            "in memory and thrown away. Example: --path ~/.agent_memory/store.json"
        )
    if not args.source.is_dir():
        parser.error("source must be an existing directory")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    store = MemoryStore(path=args.path)
    written = skipped = 0
    for md in sorted(args.source.glob("*.md")):
        mem_type = type_for(md.name)
        if mem_type not in MEMORY_TYPES:
            mem_type = "fact"
        for body in sections(md.read_text(encoding="utf-8"), max_tokens=args.max_tokens):
            _, stored = store.write_with_status(
                body, type=mem_type, metadata={"source": md.name},
                source={"document": md.name, "event": "markdown_import"},
            )
            written += stored
            skipped += not stored
    note = f" ({skipped} skipped as exact duplicates)" if skipped else ""
    print(f"Ingested {written} memories from {args.source} -> {args.path}{note}")


if __name__ == "__main__":
    main()
