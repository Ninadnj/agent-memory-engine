"""The memory store: write durable facts, recall only what's relevant.

This is the engine behind the Markdown scaffold idea. Instead of an agent
reading whole memory files into context every session, it writes atomic
memories here and recalls the top-k relevant ones for the task at hand via
vector similarity. That is what cuts context tokens while keeping the facts the
agent actually needs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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


class MemoryStore:
    """In-memory vector store with JSON persistence.

    Small by design: an agent's durable memory for one project is hundreds of
    entries, not millions, so a brute-force cosine search over a numpy matrix
    is both exact and instant. Swap in FAISS/Chroma behind the same API only if
    a project ever outgrows this.
    """

    def __init__(
        self, path: Optional[str | Path] = None, embedder: Optional[Embedder] = None
    ) -> None:
        self.path = Path(path) if path else None
        self.embedder = embedder or default_embedder()
        self._entries: list[MemoryEntry] = []
        self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
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
        if type not in MEMORY_TYPES:
            raise ValueError(f"unknown memory type {type!r}; use one of {MEMORY_TYPES}")
        vec = self.embedder.embed([text])[0]

        # Skip near-duplicates so repeated handoffs don't bloat the store.
        if len(self._entries):
            sims = self._matrix @ vec
            best = int(np.argmax(sims))
            if sims[best] >= dedup_threshold:
                return self._entries[best]

        entry = MemoryEntry(
            id=id or f"mem_{len(self._entries) + 1:04d}",
            type=type,
            text=text,
            metadata=metadata or {},
            agent=agent,
        )
        self._entries.append(entry)
        self._matrix = np.vstack([self._matrix, vec[None, :]])
        if self.path:
            self.save()
        return entry

    # ---- reading -------------------------------------------------------
    def recall(
        self,
        query: str,
        k: int = 5,
        type_filter: Optional[str] = None,
        budget_tokens: Optional[int] = None,
    ) -> list[RecallHit]:
        """Top-k most relevant memories, optionally under a hard token budget.

        With `budget_tokens` set, memories are packed greedily in relevance
        order: an entry that would overflow the remaining budget is skipped and
        the next-best one is tried. The result never costs more than the budget
        — the caller controls exactly how much context this loads.
        """
        if not self._entries:
            return []
        qvec = self.embedder.embed([query])[0]
        sims = self._matrix @ qvec  # cosine: both sides are unit-norm
        order = np.argsort(-sims)
        hits: list[RecallHit] = []
        remaining = budget_tokens
        for idx in order:
            entry = self._entries[idx]
            if type_filter and entry.type != type_filter:
                continue
            if remaining is not None:
                cost = entry.tokens
                if cost > remaining:
                    continue  # doesn't fit; a smaller lower-ranked one may
                remaining -= cost
            hits.append(RecallHit(entry=entry, score=float(sims[idx])))
            if len(hits) >= k:
                break
        return hits

    def latest(self, type: str) -> Optional[MemoryEntry]:
        """Most recently written entry of a type (e.g. the last handoff)."""
        for entry in reversed(self._entries):
            if entry.type == type:
                return entry
        return None

    def all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def stats(self) -> dict:
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
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("no path set for this store")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedder": type(self.embedder).__name__,
            "dim": self.embedder.dim,
            "entries": [
                {**asdict(e), "embedding": self._matrix[i].tolist()}
                for i, e in enumerate(self._entries)
            ],
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def load(self, path: Optional[str | Path] = None) -> None:
        target = Path(path) if path else self.path
        if target is None or not target.exists():
            return
        payload = json.loads(target.read_text())
        self._entries = []
        vectors: list[list[float]] = []
        for raw in payload.get("entries", []):
            embedding = raw.pop("embedding", None)
            self._entries.append(MemoryEntry(**raw))
            if embedding is None:  # re-embed if the file predates stored vectors
                embedding = self.embedder.embed([raw["text"]])[0].tolist()
            vectors.append(embedding)
        self._matrix = (
            np.array(vectors, dtype=np.float32)
            if vectors
            else np.zeros((0, self.embedder.dim), dtype=np.float32)
        )
