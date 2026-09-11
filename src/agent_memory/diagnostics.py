"""Local diagnostics and evidence inspection; never evaluates memory text."""

from dataclasses import asdict
import os
from pathlib import Path
import re
import subprocess

from .embeddings import embedding_config
from .rendering import render_entry
from .store import MemoryStore, _validate_limits, find_project_root, startup_fresh
from .tokens import count_tokens, using_exact_tokenizer


def inspect_memory(store: MemoryStore, entry_id: str) -> dict | None:
    entry = store.get(entry_id)
    if entry is None:
        return None
    result = asdict(entry)
    result["source_check"] = source_check(store, entry.source)
    return result


def source_check(store: MemoryStore, source: dict) -> str:
    """A changed source calls for review; it does not prove the fact is false."""
    if not source.get("path"):
        return "no file source supplied"
    root = find_project_root(store.path.parent) if store.path else None
    if root is None:
        return "source cannot be checked outside a Git project"
    path = source["path"]
    if not isinstance(path, str) or Path(path).is_absolute():
        return "source path must be relative to the project"
    target = (root / path).resolve()
    if root != target and root not in target.parents:
        return "source path is outside the project"
    if not target.is_file():
        return "source file is missing"
    commit = source.get("commit", "")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{7,64}", commit):
        return "file exists; no verifiable source commit supplied"
    relative = target.relative_to(root).as_posix()
    try:
        exists = subprocess.run(["git", "cat-file", "-e", f"{commit}:{relative}"],
                                cwd=root, capture_output=True, timeout=5)
        if exists.returncode:
            return "source commit or path cannot be verified"
        diff = subprocess.run(["git", "diff", "--quiet", commit, "--", relative],
                              cwd=root, capture_output=True, timeout=5)
        if diff.returncode == 0:
            return "file unchanged since source commit; claim still requires verification"
        if diff.returncode == 1:
            return "source changed; review this memory"
        return "source check failed"
    except (OSError, subprocess.SubprocessError):
        return "Git source check unavailable"


def doctor(path: Path) -> dict:
    report = {"store": str(path.resolve()), "exists": path.exists(), "ok": False,
              "tokenizer": "cl100k_base" if using_exact_tokenizer() else "approximate",
              "locking": "OS advisory lock (local filesystems only)"}
    try:
        store = MemoryStore(path)
        report.update(store.stats())
        report["embedding_config"] = embedding_config(store.embedder)
        report["expired_startup_notes"] = sum(
            entry.type in ("handoff", "worklog") and not startup_fresh(entry) for entry in store.all())
        directory = path.parent
        while not directory.exists() and directory != directory.parent:
            directory = directory.parent
        report["directory_writable"] = os.access(directory, os.W_OK)
        report["ok"] = report["directory_writable"]
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        report["error"] = str(exc)
    return report


def explain_recall(store: MemoryStore, query: str, *, k=5, budget=300, min_score=0.0, decay=True) -> dict:
    _validate_limits(k, budget, min_score)
    hits = store.recall(query, k=max(1, len(store.all())), min_score=-1, decay=decay)
    rows, parts = [], []
    for hit in hits:
        block = render_entry(hit.entry)
        candidate = "\n".join(parts + [block])
        if hit.score < min_score:
            reason = "below relevance floor"
        elif len(parts) >= k:
            reason = "result limit"
        elif budget is not None and count_tokens(candidate) > budget:
            reason = "rendered text exceeds remaining budget"
        else:
            reason = "selected"
            parts.append(block)
        rows.append({"id": hit.entry.id, "revision": hit.entry.revision,
                     "score": round(hit.score, 6), "reason": reason})
    rows.extend({"id": entry.id, "revision": entry.revision, "reason": "superseded"}
                for entry in store.all() if entry.status != "active")
    return {"query": query, "rendered_tokens": count_tokens("\n".join(parts)),
            "budget": budget, "tokenizer": "cl100k_base" if using_exact_tokenizer() else "approximate",
            "candidates": rows}
