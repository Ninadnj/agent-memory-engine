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

import json
import os
import uuid
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Optional

import numpy as np

from ._locking import _file_lock
from .embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    default_embedder,
    validated_embed,
)
from .models import (
    HALF_LIFE_DAYS as HALF_LIFE_DAYS,
    MAX_HISTORY,
    MAX_TEXT_CHARS,
    MEMORY_TYPES,
    MemoryConflictError,
    MemoryEntry,
    RecallHit,
    _now_iso,
    _validate_limits,
    _validate_mapping,
    _validate_text,
    age_in_days as age_in_days,
    decay_factor,
    startup_fresh,
)
from .persistence import (
    STORE_FORMAT as STORE_FORMAT,
    StoreFormatError,
    file_stamp,
    read_payload,
    read_snapshot,
    write_snapshot,
)

# Where memories live when nothing is configured. One store per project, not one
# store for everything you have ever worked on: recall matches on similarity
# alone, so a single global file lets one project's deploy notes surface while
# you are working on another.
PROJECT_STORE_DIR = ".agent_memory"
STORE_FILENAME = "store.json"
GLOBAL_STORE = Path.home() / PROJECT_STORE_DIR / STORE_FILENAME


def find_project_root(start: Optional[str | Path] = None) -> Optional[Path]:
    """Nearest ancestor directory containing `.git`, or None outside a repo."""
    current = Path(start).expanduser().resolve() if start else Path.cwd().resolve()
    for candidate in (current, *current.parents):
        # `.git` is a directory in a normal clone and a file in a worktree or
        # submodule, so test for existence rather than for a directory.
        if (candidate / ".git").exists():
            return candidate
    return None


def default_store_path(start: Optional[str | Path] = None) -> Path:
    """Resolve which store to use.

    In precedence order: an explicit ``AGENT_MEMORY_PATH``, then the current
    project's ``.agent_memory/store.json``, then a global store for work that
    isn't in a repository.
    """
    configured = os.environ.get("AGENT_MEMORY_PATH")
    if configured:
        return Path(configured).expanduser()
    root = find_project_root(start)
    if root is not None:
        return root / PROJECT_STORE_DIR / STORE_FILENAME
    return GLOBAL_STORE


def relocation_notice(path: Path) -> Optional[str]:
    """Warn once when a fresh project store is used but a global one has data.

    Stores used to default to a single global file. Without this, upgrading
    looks like every memory was deleted. Callers must print it to stderr — on
    the MCP server stdout carries the protocol.
    """
    if path == GLOBAL_STORE or path.exists() or not GLOBAL_STORE.exists():
        return None
    try:
        count = len(json.loads(GLOBAL_STORE.read_text()).get("entries", []))
    except (OSError, ValueError):
        return None
    if not count:
        return None
    return (
        f"[agent-memory] Starting an empty store for this project at {path}. "
        f"Your global store still holds {count} memories — use them here with "
        f"AGENT_MEMORY_PATH={GLOBAL_STORE}"
    )


def _synchronized(method):
    """Keep each public operation consistent across threads sharing one store."""

    @wraps(method)
    def call(self, *args, **kwargs):
        with self._mutex:
            return method(self, *args, **kwargs)

    return call


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
        self._mutex = RLock()
        self.path = Path(path).expanduser() if path else None
        self.embedder = (
            embedder if embedder is not None else self._configured_embedder()
        )
        self._entries: list[MemoryEntry] = []
        self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._stamp: Optional[tuple[int, int, int]] = None
        if self.path and self.path.exists():
            self.load()

    def _configured_embedder(self) -> Embedder:
        """An existing store pins its backend unless the caller overrides it."""
        if (
            self.path
            and self.path.exists()
            and os.environ.get("AGENT_MEMORY_EMBEDDER", "auto").strip().lower()
            == "auto"
        ):
            try:
                payload = read_payload(self.path)
                config = payload.get("embedding_config", {})
                if config.get("backend") == "hashing":
                    return HashingEmbedder(dim=config["dim"])
                if config.get("backend") == "sentence-transformers":
                    return SentenceTransformerEmbedder(
                        config["model"], revision=config.get("revision")
                    )
            except (ValueError, KeyError, AttributeError, TypeError) as exc:
                raise StoreFormatError(
                    f"cannot read embedding configuration in {self.path}: {exc}"
                ) from exc
        return default_embedder()

    # ---- writing -------------------------------------------------------
    @contextmanager
    def _transaction(self):
        """Reload under the lock and roll back local state if persistence fails."""
        with _file_lock(self.path) if self.path else nullcontext():
            self._reload_if_changed()
            entries, matrix, stamp = (
                deepcopy(self._entries),
                self._matrix.copy(),
                self._stamp,
            )
            try:
                yield
            except BaseException:
                self._entries, self._matrix, self._stamp = entries, matrix, stamp
                raise

    @_synchronized
    def write(
        self,
        text: str,
        type: str = "fact",
        metadata: Optional[dict] = None,
        id: Optional[str] = None,
        *,
        agent: str = "",
        source: Optional[dict] = None,
        deduplicate: bool = True,
    ) -> MemoryEntry:
        """Save a memory, or return its exact duplicate of the same type/source.

        Caller-supplied IDs are preserved; a duplicate ID raises ValueError.
        Semantic similarity never establishes identity. Set ``deduplicate=False``
        to store a separate occurrence. The returned entry is a detached snapshot.
        """
        return self.write_with_status(
            text,
            type,
            metadata,
            id,
            agent=agent,
            source=source,
            deduplicate=deduplicate,
        )[0]

    @_synchronized
    def write_with_status(
        self,
        text: str,
        type: str = "fact",
        metadata: Optional[dict] = None,
        id: Optional[str] = None,
        *,
        agent: str = "",
        source: Optional[dict] = None,
        deduplicate: bool = True,
    ) -> tuple[MemoryEntry, bool]:
        """Like write; stored=False means an exact duplicate was found."""
        _validate_text(text)
        if type not in MEMORY_TYPES:
            raise ValueError(f"unknown memory type {type!r}; use one of {MEMORY_TYPES}")
        _validate_mapping(metadata if metadata is not None else {}, "metadata")
        _validate_mapping(source if source is not None else {}, "source")
        if id is not None and not isinstance(id, str):
            raise ValueError("id must be a string")
        if not isinstance(agent, str) or len(agent) > 200:
            raise ValueError("agent must be a string of at most 200 characters")
        if not isinstance(deduplicate, bool):
            raise ValueError("deduplicate must be a boolean")
        with self._transaction():
            entry, stored = self._append(
                text, type, metadata, id, deduplicate, agent, source
            )
            if stored and self.path:
                self._save_unlocked()
            return deepcopy(entry), stored

    def _append(self, text, type, metadata, id, deduplicate, agent, source=None):
        if id is not None and any(entry.id == id for entry in self._entries):
            raise ValueError(f"duplicate memory id {id!r}; use update to revise it")
        # Whitespace-only differences can be ignored; case, numbers and
        # negation can change a fact or identifier and must be preserved.
        normalized = " ".join(text.split())
        if id is None and deduplicate:
            for entry in self._entries:
                if (
                    entry.status == "active"
                    and entry.type == type
                    and entry.source == (source or {})
                    and " ".join(entry.text.split()) == normalized
                ):
                    return entry, False
        vec = validated_embed(self.embedder, [text])[0]
        entry = MemoryEntry(
            id=id if id is not None else self._next_id(),
            type=type,
            text=text,
            created_at=_now_iso(),
            metadata=deepcopy(metadata or {}),
            agent=agent,
            source=deepcopy(source or {}),
        )
        self._entries.append(entry)
        self._matrix = np.vstack([self._matrix, vec[None, :]])
        return entry, True

    def _next_id(self) -> str:
        """Generate an identity independent of deletions and store lifetimes."""
        used = {e.id for e in self._entries}
        candidate = f"mem_{uuid.uuid4().hex}"
        while candidate in used:
            candidate = f"mem_{uuid.uuid4().hex}"
        return candidate

    @staticmethod
    def _check_revision(entry: MemoryEntry, expected: Optional[int]) -> None:
        if expected is not None:
            if (
                isinstance(expected, bool)
                or not isinstance(expected, int)
                or expected < 1
            ):
                raise ValueError("expected_revision must be a positive integer")
            if entry.revision != expected:
                raise MemoryConflictError(
                    f"memory {entry.id} changed: expected revision {expected}, current {entry.revision}; reload before editing"
                )

    @_synchronized
    def forget(self, entry_id: str, *, expected_revision: Optional[int] = None) -> bool:
        """Delete one memory; optionally reject a stale caller revision."""
        with self._transaction():
            for i, entry in enumerate(self._entries):
                if entry.id == entry_id:
                    self._check_revision(entry, expected_revision)
                    del self._entries[i]
                    self._matrix = np.delete(self._matrix, i, axis=0)
                    if self.path:
                        self._save_unlocked()
                    return True
            return False

    @_synchronized
    def update(
        self,
        entry_id: str,
        text: Optional[str] = None,
        type: Optional[str] = None,
        *,
        expected_revision: Optional[int] = None,
        agent: str = "",
        source: Optional[dict] = None,
    ) -> Optional[MemoryEntry]:
        """Revise an active memory, retaining creation time and revision history.

        No-op edits do not refresh stale information. The original author is
        preserved; updated_by identifies the correcting agent.
        """
        if text is not None:
            _validate_text(text)
        if type is not None and type not in MEMORY_TYPES:
            raise ValueError(f"unknown memory type {type!r}")
        if source is not None:
            _validate_mapping(source, "source")
        if not isinstance(agent, str) or len(agent) > 200:
            raise ValueError("agent must be a string of at most 200 characters")
        with self._transaction():
            for i, entry in enumerate(self._entries):
                if entry.id != entry_id:
                    continue
                self._check_revision(entry, expected_revision)
                if entry.status != "active":
                    raise MemoryConflictError(
                        f"memory {entry_id} is superseded by {entry.superseded_by}"
                    )
                new_text = entry.text if text is None else text
                new_type = entry.type if type is None else type
                new_source = entry.source if source is None else source
                if (new_text, new_type, new_source) == (
                    entry.text,
                    entry.type,
                    entry.source,
                ):
                    return deepcopy(entry)
                vec = (
                    validated_embed(self.embedder, [new_text])[0]
                    if new_text != entry.text
                    else self._matrix[i]
                )
                self._record_revision(entry, agent)
                entry.text, entry.type, entry.source = (
                    new_text,
                    new_type,
                    deepcopy(new_source),
                )
                self._matrix[i] = vec
                if self.path:
                    self._save_unlocked()
                return deepcopy(entry)
            return None

    @staticmethod
    def _record_revision(entry: MemoryEntry, agent: str) -> None:
        snapshot = asdict(entry)
        snapshot.pop("history")
        entry.history = (entry.history + [snapshot])[-MAX_HISTORY:]
        entry.revision += 1
        entry.updated_at = _now_iso()
        entry.updated_by = agent

    @_synchronized
    def supersede(
        self,
        entry_id: str,
        text: str,
        *,
        expected_revision: int,
        agent: str = "",
        source: Optional[dict] = None,
    ) -> MemoryEntry:
        """Atomically replace an active decision with a new identity.

        The old memory remains inspectable but is excluded from normal recall.
        """
        _validate_text(text)
        _validate_mapping(source if source is not None else {}, "source")
        if expected_revision is None:
            raise ValueError("expected_revision must be a positive integer")
        if not isinstance(agent, str) or len(agent) > 200:
            raise ValueError("agent must be a string of at most 200 characters")
        with self._transaction():
            old = next((entry for entry in self._entries if entry.id == entry_id), None)
            if old is None:
                raise ValueError(f"No memory with id {entry_id}.")
            self._check_revision(old, expected_revision)
            if old.status != "active":
                raise MemoryConflictError(f"memory {entry_id} is already superseded")
            replacement, _ = self._append(
                text, old.type, old.metadata, None, False, agent, source
            )
            self._record_revision(old, agent)
            old.status, old.superseded_by = "superseded", replacement.id
            if self.path:
                self._save_unlocked()
            return deepcopy(replacement)

    @_synchronized
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Return a detached snapshot, including superseded entries and history."""
        self._reload_if_changed()
        return deepcopy(
            next((entry for entry in self._entries if entry.id == entry_id), None)
        )

    # ---- reading -------------------------------------------------------
    @_synchronized
    def recall(
        self,
        query: str,
        k: int = 5,
        type_filter: Optional[str] = None,
        budget_tokens: Optional[int] = None,
        exclude_ids: Optional[set[str]] = None,
        min_score: float = 0.0,
        decay: bool = True,
    ) -> list[RecallHit]:
        """Top-k most relevant memories, optionally under a hard token budget.

        With `budget_tokens` set, memories are packed greedily in relevance
        order: an entry that would overflow the remaining budget is skipped and
        the next-best one is tried. The result never costs more than the budget
        — the caller controls exactly how much context this loads.

        `min_score` drops weak matches entirely. Without it a query unrelated to
        anything in the store still returns k memories, and the agent reading
        them has no way to tell they are noise.

        `decay` fades time-sensitive memories (see HALF_LIFE_DAYS) so that a
        month-old "currently implementing X" ranks below a fresh fact instead of
        alongside it. Durable types are unaffected. Combined with `min_score`,
        stale status notes eventually drop out of recall on their own.
        """
        self._reload_if_changed()
        return self._recall_snapshot(
            query,
            k=k,
            type_filter=type_filter,
            budget_tokens=budget_tokens,
            exclude_ids=exclude_ids,
            min_score=min_score,
            decay=decay,
        )

    def _recall_snapshot(
        self,
        query: str,
        *,
        k: int,
        type_filter: Optional[str] = None,
        budget_tokens: Optional[int] = None,
        exclude_ids: Optional[set[str]] = None,
        min_score: float = 0.0,
        decay: bool = True,
    ) -> list[RecallHit]:
        """Score the current snapshot without reloading; caller holds the mutex."""
        _validate_limits(k, budget_tokens, min_score)
        if not isinstance(query, str) or len(query) > MAX_TEXT_CHARS:
            raise ValueError(
                f"query must be a string of at most {MAX_TEXT_CHARS} characters"
            )
        if type_filter is not None and type_filter not in MEMORY_TYPES:
            raise ValueError(f"unknown memory type {type_filter!r}")
        if not self._entries or k == 0 or budget_tokens == 0 or not query.strip():
            return []
        qvec = validated_embed(self.embedder, [query])[0]
        sims = self._matrix @ qvec  # cosine: both sides are unit-norm
        if decay:
            factors = np.array(
                [decay_factor(e) for e in self._entries], dtype=np.float32
            )
            # Only fade positive scores: scaling a negative one moves it toward
            # zero, which would promote an unrelated old memory rather than bury it.
            sims = np.where(sims > 0, sims * factors, sims)
        order = np.argsort(-sims)
        hits: list[RecallHit] = []
        remaining = budget_tokens
        for idx in order:
            score = float(sims[idx])
            if score < min_score:
                break  # sorted by score, so nothing further can qualify
            entry = self._entries[idx]
            if entry.status != "active":
                continue
            if exclude_ids and entry.id in exclude_ids:
                continue
            if type_filter and entry.type != type_filter:
                continue
            if remaining is not None:
                cost = entry.tokens
                if cost > remaining:
                    continue  # doesn't fit; a smaller lower-ranked one may
                remaining -= cost
            hits.append(RecallHit(entry=deepcopy(entry), score=score))
            if len(hits) >= k:
                break
        return hits

    @_synchronized
    def boot(
        self,
        task: str,
        k: int = 5,
        budget_tokens: Optional[int] = 300,
        min_score: float = 0.0,
        decay: bool = True,
    ) -> tuple[Optional[MemoryEntry], list[RecallHit]]:
        """Return the latest handoff plus relevant memories for a new session.

        The budget applies to memory content across both parts. If the latest
        handoff is too large to fit, it is skipped and the full budget remains
        available for relevant memories. Both parts use one snapshot, even if
        another process commits a correction while this call is running.
        """
        _validate_limits(k, budget_tokens, min_score)
        remaining = budget_tokens
        latest_handoff = self.latest("handoff", fresh=True)
        included_handoff: Optional[MemoryEntry] = None
        excluded_ids = {
            entry.id
            for entry in self._entries
            if entry.type in ("handoff", "worklog") and not startup_fresh(entry)
        }

        if latest_handoff is not None:
            excluded_ids.add(latest_handoff.id)
            if remaining is None or latest_handoff.tokens <= remaining:
                included_handoff = latest_handoff
                if remaining is not None:
                    remaining -= latest_handoff.tokens

        # latest() already refreshed the snapshot. Reloading again here could
        # combine a retired handoff with its replacement from another writer.
        hits = self._recall_snapshot(
            task,
            k=k,
            budget_tokens=remaining,
            exclude_ids=excluded_ids,
            min_score=min_score,
            decay=decay,
        )
        return included_handoff, hits

    @_synchronized
    def latest(self, type: str, *, fresh: bool = False) -> Optional[MemoryEntry]:
        """Most recently written entry of a type (e.g. the last handoff)."""
        self._reload_if_changed()
        for entry in reversed(self._entries):
            if (
                entry.type == type
                and entry.status == "active"
                and (not fresh or startup_fresh(entry))
            ):
                return deepcopy(entry)
        return None

    @_synchronized
    def __len__(self) -> int:
        """Count entries without copying their text, metadata and histories."""
        self._reload_if_changed()
        return len(self._entries)

    @_synchronized
    def all(self) -> list[MemoryEntry]:
        """Return detached snapshots; use update/supersede to persist changes."""
        self._reload_if_changed()
        return deepcopy(self._entries)

    @_synchronized
    def stats(self) -> dict:
        self._reload_if_changed()
        by_type: dict[str, int] = {}
        for e in self._entries:
            by_type[e.type] = by_type.get(e.type, 0) + 1
        return {
            "count": len(self._entries),
            "active": sum(e.status == "active" for e in self._entries),
            "superseded": sum(e.status == "superseded" for e in self._entries),
            "by_type": by_type,
            "total_tokens": sum(e.tokens for e in self._entries),
            "embedding_dim": self.embedder.dim,
            "embedder": type(self.embedder).__name__,
        }

    # ---- persistence ---------------------------------------------------
    @_synchronized
    def save(self, path: Optional[str | Path] = None) -> None:
        """Save only a current snapshot; export to a new path for a backup."""
        target = Path(path).expanduser() if path else self.path
        if target is None:
            raise ValueError("no path set for this store")
        if self.path is None or target.resolve() != self.path.resolve():
            self.export(target)
            return
        with _file_lock(target):
            if self._read_stamp() != self._stamp:
                raise MemoryConflictError(
                    "store changed since this snapshot; reload before saving"
                )
            self._save_unlocked(target)

    @_synchronized
    def export(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Write an explicit snapshot to a different file, without rebinding."""
        target = Path(path).expanduser()
        if self.path and target.resolve() == self.path.resolve():
            raise ValueError("export destination must differ from the live store")
        self._reload_if_changed()
        with _file_lock(target):
            if target.exists() and not overwrite:
                raise FileExistsError(f"export destination already exists: {target}")
            self._save_unlocked(target)

    def _save_unlocked(self, path: Optional[Path] = None) -> None:
        target = path or self.path
        assert target is not None
        write_snapshot(target, self._entries, self._matrix, self.embedder)
        if self.path and target.resolve() == self.path.resolve():
            self._stamp = self._read_stamp()

    @_synchronized
    def load(self, path: Optional[str | Path] = None) -> None:
        target = Path(path).expanduser() if path else self.path
        if target is None or not target.exists():
            return
        if self.path and target.resolve() != self.path.resolve():
            raise ValueError("open a separate MemoryStore to load a different file")
        entries, matrix, stamp = read_snapshot(target, self.embedder)
        self._entries, self._matrix = entries, matrix
        if target == self.path:
            self._stamp = stamp

    def _read_stamp(self) -> Optional[tuple[int, int, int]]:
        return file_stamp(self.path) if self.path else None

    def _reload_if_changed(self) -> None:
        if self.path is None:
            return
        stamp = self._read_stamp()
        if stamp == self._stamp:
            return
        if stamp is None:
            self._entries = []
            self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
            self._stamp = None
            return
        self.load()
