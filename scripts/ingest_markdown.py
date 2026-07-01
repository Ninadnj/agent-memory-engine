"""Migrate an existing Markdown memory scaffold into the engine.

Reads a directory of `.md` files (e.g. the agent-memory/ folder from the
ai-agent-memory-scaffold convention) and writes one memory per `##` section,
guessing the memory type from the filename.

    python scripts/ingest_markdown.py path/to/agent-memory/ [--path store.json]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_memory import MEMORY_TYPES, MemoryStore  # noqa: E402

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


def sections(text: str) -> list[str]:
    """Split a Markdown file into `##` sections, dropping headings and blanks."""
    chunks = re.split(r"^##\s+.*$", text, flags=re.MULTILINE)
    return [c.strip() for c in chunks if c.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory of .md memory files")
    parser.add_argument("--path", type=Path, default=None, help="output store.json")
    args = parser.parse_args()

    store = MemoryStore(path=args.path)
    written = 0
    for md in sorted(args.source.glob("*.md")):
        mem_type = type_for(md.name)
        if mem_type not in MEMORY_TYPES:
            mem_type = "fact"
        for body in sections(md.read_text()):
            store.write(body, type=mem_type, metadata={"source": md.name})
            written += 1
    dest = args.path or "(in-memory, not saved — pass --path)"
    print(f"Ingested {written} memories from {args.source} -> {dest}")


if __name__ == "__main__":
    main()
