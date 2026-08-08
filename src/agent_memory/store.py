"""The memory store: write durable facts, recall only what's relevant.

This is the engine behind the Markdown scaffold idea. Instead of an agent
reading whole memory files into context every session, it writes atomic
memories here and recalls the top-k relevant ones for the task at hand via
vector similarity. That is what cuts context tokens while keeping the facts the
agent actually needs.

Persistence is a single JSON file, but writes are careful: they take a lock,
re-read anything another process appended, and land atomically. Two agents
pointed at one store append to it instead of overwriting each other.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .embeddings import Embedder, default_embedder
from .tokens import count_tokens

# Memory categories mirror the original Markdown scaffold (PROJECT, DECISIONS,
# KNOWN_ISSUES, STATE, HANDOFF, WORKLOG) so migration is one-to-one.
MEMORY_TYPES = {
    "project",
    "decision",
    "issue",
    "state",
    "handoff",
    "worklog",
    "fact",
}

# Bumped when the on-disk layout changes. v2 stores embeddings as base64
# float16 instead of JSON float lists (~5x smaller, same ranking).
STORE_FORMAT = 2

_ID_RE = re.compile(r"^mem_(\d+)$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class MemoryEntry:
    id: str
    type: str
    text: str
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    # Which agent wrote this (e.g. "claude-code", "codex", "cursor"). Lets one
    # store be shared between agents while keeping provenance visible.
    agent: str = ""

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


@dataclass
class RecallHit:
    entry: MemoryEntry
    score: float


@contextmanager
def _file_lock(target: Path, timeout: float = 10.0, stale_after: float = 60.0) -> Iterator[None]:
    """Cross-process advisory lock for one store file.

    An exclusive-create lock file is portable (POSIX and Windows) and needs no
    extra dependency. A lock older than `stale_after` is assumed to belong to a
    crashed process and is broken, so a dead agent can't wedge the store.
    """
    lock = target.with_name(target.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale_after:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue  # released while we looked; retry immediately
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"could not lock {target} after {timeout}s; "
                    f"remove {lock} if no agent is running"
                )
            time.sleep(0.02)
    try:
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _encode_vector(vec: np.ndarray) -> str:
    """float16 + base64. Precision loss is ~1e-3 — far below what ranking needs."""
    return base64.b64encode(np.asarray(vec, dtype=np.float16).tobytes()).decode("ascii")


def _decode_vector(raw: str | list[float]) -> np.ndarray:
    if isinstance(raw, str):
        vec = np.frombuffer(base64.b64decode(raw), dtype=np.float16).astype(np.float32)
    else:  # v1 stores kept a plain JSON list of floats
        vec = np.asarray(raw, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec  # re-normalise after the float16 round-trip


class MemoryStore:
    """Vector store over a JSON file.

    Small by design: an agent's durable memory for one project is hundreds of
    entries, not millions, so a brute-force cosine search over a numpy matrix
    is both exact and instant. Swap in FAISS/Chroma behind the same API only if
    a project ever outgrows this.
    """

    def __init__(
        self, path: Optional[str | Path] = None, embedder: Optional[Embedder] = None
    ) -> None:
        self.path = Path(path).expanduser() if path else None
        self.embedder = embedder or default_embedder()
        self._entries: list[MemoryEntry] = []
        self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._stamp: Optional[tuple[int, int]] = None
        if self.path and self.path.exists():
            self.load()

    # ---- writing -------------------------------------------------------
    def write(
        self,
        text: str,
        type: str = "fact",
        metadata: Optional[dict] = None,
        id: Optional[str] = None,
        dedup_threshold: float = 0.97,
        agent: str = "",
    ) -> MemoryEntry:
        """Save one memory. Returns the entry — the existing one if this text
        near-duplicates something already stored."""
        entry, _ = self.write_with_status(
            text,
            type=type,
            metadata=metadata,
            id=id,
            dedup_threshold=dedup_threshold,
            agent=agent,
        )
        return entry

    def write_with_status(
        self,
        text: str,
        type: str = "fact",
        metadata: Optional[dict] = None,
        id: Optional[str] = None,
        dedup_threshold: float = 0.97,
        agent: str = "",
    ) -> tuple[MemoryEntry, bool]:
        """Like `write`, but also reports whether the text was actually stored.

        Returns `(entry, stored)`. `stored=False` means the write was dropped as
        a near-duplicate and `entry` is the memory already on file — callers
        that report back to an agent must not claim a save happened.
        """
        if type not in MEMORY_TYPES:
            raise ValueError(f"unknown memory type {type!r}; use one of {MEMORY_TYPES}")

        if self.path is None:
            return self._append(text, type, metadata, id, dedup_threshold, agent)

        # Under the lock: pick up anything another agent appended, then write.
        with _file_lock(self.path):
            self._reload_if_changed()
            entry, stored = self._append(
                text, type, metadata, id, dedup_threshold, agent
            )
            if stored:
                self._save_unlocked()
            return entry, stored

    def _append(
        self,
        text: str,
        type: str,
        metadata: Optional[dict],
        id: Optional[str],
        dedup_threshold: float,
        agent: str,
    ) -> tuple[MemoryEntry, bool]:
        vec = self.embedder.embed([text])[0]

        # Skip near-duplicates so repeated handoffs don't bloat the store.
        if len(self._entries):
            sims = self._matrix @ vec
            best = int(np.argmax(sims))
            if sims[best] >= dedup_threshold:
                return self._entries[best], False

        entry = MemoryEntry(
            id=id or self._next_id(),
            type=type,
            text=text,
            metadata=metadata or {},
            agent=agent,
        )
        self._entries.append(entry)
        self._matrix = np.vstack([self._matrix, vec[None, :]])
        return entry, True

    def _next_id(self) -> str:
        """Smallest unused `mem_NNNN`. Derived from the ids actually present, so
        it survives explicit ids, deletions and concurrent appends."""
        used = {e.id for e in self._entries}
        highest = 0
        for entry_id in used:
            match = _ID_RE.match(entry_id)
            if match:
                highest = max(highest, int(match.group(1)))
        candidate = highest + 1
        while f"mem_{candidate:04d}" in used:
            candidate += 1
        return f"mem_{candidate:04d}"

    def forget(self, entry_id: str) -> bool:
        """Delete one memory. Returns False if that id isn't in the store.

        Memory that can't be corrected is worse than no memory: a stale `state`
        entry keeps being recalled and quietly misleads every later session.
        """
        if self.path is None:
            return self._remove(entry_id)
        with _file_lock(self.path):
            self._reload_if_changed()
            removed = self._remove(entry_id)
            if removed:
                self._save_unlocked()
            return removed

    def _remove(self, entry_id: str) -> bool:
        for i, entry in enumerate(self._entries):
            if entry.id == entry_id:
                del self._entries[i]
                self._matrix = np.delete(self._matrix, i, axis=0)
                return True
        return False

    def update(
        self,
        entry_id: str,
        text: Optional[str] = None,
        type: Optional[str] = None,
    ) -> Optional[MemoryEntry]:
        """Revise a memory in place, re-embedding when the text changes.

        Use this when a fact changes rather than writing a second, contradictory
        memory — both would otherwise be recalled together.
        """
        if type is not None and type not in MEMORY_TYPES:
            raise ValueError(f"unknown memory type {type!r}; use one of {MEMORY_TYPES}")
        if self.path is None:
            return self._revise(entry_id, text, type)
        with _file_lock(self.path):
            self._reload_if_changed()
            entry = self._revise(entry_id, text, type)
            if entry is not None:
                self._save_unlocked()
            return entry

    def _revise(
        self, entry_id: str, text: Optional[str], type: Optional[str]
    ) -> Optional[MemoryEntry]:
        for i, entry in enumerate(self._entries):
            if entry.id != entry_id:
                continue
            if text is not None and text != entry.text:
                entry.text = text
                self._matrix[i] = self.embedder.embed([text])[0]
            if type is not None:
                entry.type = type
            return entry
        return None

    # ---- reading -------------------------------------------------------
    def recall(
        self,
        query: str,
        k: int = 5,
        type_filter: Optional[str] = None,
        budget_tokens: Optional[int] = None,
        exclude_ids: Optional[set[str]] = None,
        min_score: float = 0.0,
    ) -> list[RecallHit]:
        """Top-k most relevant memories, optionally under a hard token budget.

        With `budget_tokens` set, memories are packed greedily in relevance
        order: an entry that would overflow the remaining budget is skipped and
        the next-best one is tried. The result never costs more than the budget
        — the caller controls exactly how much context this loads.

        `min_score` drops weak matches entirely. Without it a query unrelated to
        anything in the store still returns k memories, and the agent reading
        them has no way to tell they are noise.
        """
        self._reload_if_changed()
        if not self._entries:
            return []
        qvec = self.embedder.embed([query])[0]
        sims = self._matrix @ qvec  # cosine: both sides are unit-norm
        order = np.argsort(-sims)
        hits: list[RecallHit] = []
        remaining = budget_tokens
        for idx in order:
            score = float(sims[idx])
            if score < min_score:
                break  # sorted by score, so nothing further can qualify
            entry = self._entries[idx]
            if exclude_ids and entry.id in exclude_ids:
                continue
            if type_filter and entry.type != type_filter:
                continue
            if remaining is not None:
                cost = entry.tokens
                if cost > remaining:
                    continue  # doesn't fit; a smaller lower-ranked one may
                remaining -= cost
            hits.append(RecallHit(entry=entry, score=score))
            if len(hits) >= k:
                break
        return hits

    def boot(
        self,
        task: str,
        k: int = 5,
        budget_tokens: Optional[int] = 300,
        min_score: float = 0.0,
    ) -> tuple[Optional[MemoryEntry], list[RecallHit]]:
        """Return the latest handoff plus relevant memories for a new session.

        The budget applies to memory content across both parts. If the latest
        handoff is too large to fit, it is skipped and the full budget remains
        available for relevant memories.
        """
        remaining = budget_tokens
        latest_handoff = self.latest("handoff")
        included_handoff: Optional[MemoryEntry] = None
        excluded_ids: set[str] = set()

        if latest_handoff is not None:
            excluded_ids.add(latest_handoff.id)
            if remaining is None or latest_handoff.tokens <= remaining:
                included_handoff = latest_handoff
                if remaining is not None:
                    remaining -= latest_handoff.tokens

        hits = self.recall(
            task,
            k=k,
            budget_tokens=remaining,
            exclude_ids=excluded_ids,
            min_score=min_score,
        )
        return included_handoff, hits

    def latest(self, type: str) -> Optional[MemoryEntry]:
        """Most recently written entry of a type (e.g. the last handoff)."""
        self._reload_if_changed()
        for entry in reversed(self._entries):
            if entry.type == type:
                return entry
        return None

    def all(self) -> list[MemoryEntry]:
        self._reload_if_changed()
        return list(self._entries)

    def stats(self) -> dict:
        self._reload_if_changed()
        by_type: dict[str, int] = {}
        for e in self._entries:
            by_type[e.type] = by_type.get(e.type, 0) + 1
        return {
            "count": len(self._entries),
            "by_type": by_type,
            "total_tokens": sum(e.tokens for e in self._entries),
            "embedding_dim": self.embedder.dim,
            "embedder": type(self.embedder).__name__,
        }

    # ---- persistence ---------------------------------------------------
    def save(self, path: Optional[str | Path] = None) -> None:
        target = Path(path).expanduser() if path else self.path
        if target is None:
            raise ValueError("no path set for this store")
        with _file_lock(target):
            self._save_unlocked(target)

    def _save_unlocked(self, path: Optional[Path] = None) -> None:
        """Serialise atomically: a crash mid-write must not truncate the store."""
        target = path or self.path
        assert target is not None
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": STORE_FORMAT,
            "embedder": type(self.embedder).__name__,
            "dim": self.embedder.dim,
            "entries": [
                {**asdict(e), "embedding": _encode_vector(self._matrix[i])}
                for i, e in enumerate(self._entries)
            ],
        }
        tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        os.replace(tmp, target)  # atomic on POSIX and Windows
        if target == self.path:
            self._stamp = self._read_stamp()

    def load(self, path: Optional[str | Path] = None) -> None:
        target = Path(path).expanduser() if path else self.path
        if target is None or not target.exists():
            return
        # Stamp BEFORE reading. Reads are not locked (a running server reloads on
        # every recall), so another agent can replace the file mid-read. Stamping
        # afterwards would pair the new stamp with the content we already read,
        # and every later freshness check would wrongly conclude we were current.
        # Stamping first can only cause a redundant reload, never a skipped one.
        stamp = self._read_stamp() if target == self.path else None
        payload = json.loads(target.read_text())

        # A store written by a different embedder holds vectors that are not
        # comparable with ours — different dimension (a hard crash on the first
        # matmul) or, worse, the same dimension from a different model (silently
        # meaningless scores). Re-embed from the text instead.
        stored_dim = payload.get("dim")
        stored_embedder = payload.get("embedder")
        reembed = (
            stored_dim != self.embedder.dim
            or stored_embedder != type(self.embedder).__name__
        )

        entries: list[MemoryEntry] = []
        vectors: list[Optional[np.ndarray]] = []
        known = {f.name for f in MemoryEntry.__dataclass_fields__.values()}
        for raw in payload.get("entries", []):
            embedding = raw.pop("embedding", None)
            entries.append(MemoryEntry(**{k: v for k, v in raw.items() if k in known}))
            if embedding is None or reembed:
                vectors.append(None)  # filled in below, in one batch
            else:
                vectors.append(_decode_vector(embedding))

        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:
            fresh = self.embedder.embed([entries[i].text for i in missing])
            for slot, i in enumerate(missing):
                vectors[i] = fresh[slot]

        self._entries = entries
        self._matrix = (
            np.array(vectors, dtype=np.float32)
            if vectors
            else np.zeros((0, self.embedder.dim), dtype=np.float32)
        )
        if target == self.path:
            self._stamp = stamp

    def _read_stamp(self) -> Optional[tuple[int, int]]:
        try:
            st = self.path.stat()  # type: ignore[union-attr]
        except (OSError, AttributeError):
            return None
        return (st.st_mtime_ns, st.st_size)

    def _reload_if_changed(self) -> None:
        """Pick up writes made by another process since we last read the file."""
        if self.path is None or not self.path.exists():
            return
        if self._read_stamp() != self._stamp:
            self.load()
