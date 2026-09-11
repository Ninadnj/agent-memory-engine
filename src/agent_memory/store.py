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
from copy import deepcopy
import json
import os
import tempfile
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    default_embedder,
    embedding_config,
)
from ._locking import _file_lock
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
STORE_FORMAT = 3
MAX_TEXT_CHARS = 20_000
MAX_METADATA_BYTES = 16_384
MAX_HISTORY = 20
MAX_STORE_BYTES = 128 * 1024 * 1024


class MemoryConflictError(ValueError):
    """The caller's revision or store snapshot is no longer current."""


class StoreFormatError(ValueError):
    """An unreadable or invalid store was left untouched."""


def _validate_text(text: str) -> None:
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"memory text must contain 1..{MAX_TEXT_CHARS} characters")


def _validate_mapping(value: dict, name: str, *, limit: bool = True) -> None:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    try:
        encoded = json.dumps(value, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values") from exc
    if limit and len(encoded) > MAX_METADATA_BYTES:
        raise ValueError(f"{name} exceeds {MAX_METADATA_BYTES} bytes")


def _validate_limits(k: int, budget: Optional[int], min_score: float) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if budget is not None and (
        isinstance(budget, bool) or not isinstance(budget, int) or budget < 0
    ):
        raise ValueError("budget_tokens must be a nonnegative integer or None")
    if (
        not isinstance(min_score, (int, float))
        or not np.isfinite(min_score)
        or not -1 <= min_score <= 1
    ):
        raise ValueError("min_score must be finite and between -1 and 1")


# How fast a memory's relevance fades, in days, per type. A memory's similarity
# score is multiplied by 0.5 ** (age / half_life), so an entry at its half-life
# needs to be twice as good a match to rank where it did when fresh.
#
# Not everything should fade. "Bookings are stored in UTC" is as true in a year
# as it was on the day it was written, and decaying it would quietly lose the
# facts most worth keeping. What goes stale is the record of a moment:
# "currently implementing X" is usually false a fortnight later, and recalling
# it with full confidence actively misleads. Correct a decision with
# memory_update; let a status note fade on its own.
HALF_LIFE_DAYS: dict[str, Optional[float]] = {
    "state": 7.0,  # "currently working on…" — stale fastest
    "handoff": 7.0,  # next steps are usually done or abandoned by then
    "worklog": 21.0,  # what happened still orients, but fades
    "decision": None,  # durable until explicitly superseded
    "project": None,
    "issue": None,  # true until someone fixes it; forget it then
    "fact": None,
}

# Where memories live when nothing is configured. One store per project, not one
# store for everything you have ever worked on: recall matches on similarity
# alone, so a single global file lets one project's deploy notes surface while
# you are working on another.
PROJECT_STORE_DIR = ".agent_memory"
STORE_FILENAME = "store.json"
GLOBAL_STORE = Path.home() / PROJECT_STORE_DIR / STORE_FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    updated_at: Optional[str] = None
    updated_by: str = ""
    revision: int = 1
    status: str = "active"
    superseded_by: Optional[str] = None
    source: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


@dataclass
class RecallHit:
    entry: MemoryEntry
    score: float


def _entry_from_raw(raw: dict, *, history: bool = True) -> MemoryEntry:
    if not isinstance(raw, dict):
        raise ValueError("each memory must be an object")
    known = MemoryEntry.__dataclass_fields__
    entry = MemoryEntry(**{key: value for key, value in raw.items() if key in known})
    # New writes are bounded at the API boundary. Existing records may predate
    # those limits (including empty text); retain them so they can be corrected
    # or deleted without an unrelated migration changing their identity/data.
    if not isinstance(entry.text, str):
        raise ValueError("stored memory text must be a string")
    if entry.type not in MEMORY_TYPES:
        raise ValueError(f"unknown memory type {entry.type!r}")
    if not isinstance(entry.id, str):
        raise ValueError("invalid memory id")
    for name in ("created_at", "agent", "updated_by"):
        if not isinstance(getattr(entry, name), str):
            raise ValueError(f"{name} must be a string")
    if entry.updated_at is not None and not isinstance(entry.updated_at, str):
        raise ValueError("updated_at must be a string or null")
    if (
        isinstance(entry.revision, bool)
        or not isinstance(entry.revision, int)
        or entry.revision < 1
    ):
        raise ValueError("revision must be a positive integer")
    if entry.status not in ("active", "superseded"):
        raise ValueError("invalid memory status")
    if entry.superseded_by is not None and not isinstance(entry.superseded_by, str):
        raise ValueError("superseded_by must be an id or null")
    if entry.status == "superseded" and entry.superseded_by is None:
        raise ValueError("superseded memory must name its replacement")
    _validate_mapping(entry.metadata, "metadata", limit=False)
    _validate_mapping(entry.source, "source", limit=False)
    if not isinstance(entry.history, list) or len(entry.history) > MAX_HISTORY:
        raise ValueError(f"history must contain at most {MAX_HISTORY} revisions")
    if history:
        for snapshot in entry.history:
            if not isinstance(snapshot, dict) or "history" in snapshot:
                raise ValueError("invalid revision snapshot")
            previous = _entry_from_raw(snapshot, history=False)
            if previous.id != entry.id or previous.revision >= entry.revision:
                raise ValueError("invalid revision history identity or order")
    return entry


def age_in_days(entry: MemoryEntry, now: Optional[datetime] = None) -> float:
    """How old a memory is. 0.0 when the timestamp is unreadable or in the future."""
    try:
        written = datetime.fromisoformat(entry.updated_at or entry.created_at)
    except (TypeError, ValueError):
        return 0.0  # an unparseable timestamp must not silently bury the memory
    if written.tzinfo is None:
        written = written.replace(tzinfo=timezone.utc)
    delta = (now or datetime.now(timezone.utc)) - written
    return max(0.0, delta.total_seconds() / 86400.0)  # clock skew must not boost


def decay_factor(entry: MemoryEntry, now: Optional[datetime] = None) -> float:
    """Multiplier applied to a memory's similarity score, in (0, 1]."""
    half_life = HALF_LIFE_DAYS.get(entry.type)
    if not half_life:
        return 1.0
    return float(0.5 ** (age_in_days(entry, now) / half_life))


def startup_fresh(entry: MemoryEntry) -> bool:
    """Startup notes expire after two half-lives (handoff 14d, worklog 42d).

    Ordinary recall still supports explicit inspection of aged content.
    Unparseable startup timestamps are omitted rather than treated as current.
    """
    if entry.status != "active":
        return False
    half_life = HALF_LIFE_DAYS.get(entry.type)
    if half_life is None:
        return True
    try:
        datetime.fromisoformat(entry.updated_at or entry.created_at)
    except (TypeError, ValueError):
        return False
    return age_in_days(entry) <= 2 * half_life


def _encode_vector(vec: np.ndarray) -> str:
    """float16 + base64. Precision loss is ~1e-3 — far below what ranking needs."""
    return base64.b64encode(np.asarray(vec, dtype=np.float16).tobytes()).decode("ascii")


def _decode_vector(raw: str | list[float]) -> np.ndarray:
    if isinstance(raw, str):
        vec = np.frombuffer(
            base64.b64decode(raw, validate=True), dtype=np.float16
        ).astype(np.float32)
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
            and os.environ.get("AGENT_MEMORY_EMBEDDER", "auto") == "auto"
        ):
            try:
                if self.path.stat().st_size > MAX_STORE_BYTES:
                    raise ValueError("store exceeds the supported local file size")
                payload = json.loads(self.path.read_text(encoding="utf-8"))
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

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(self.embedder.embed(texts), dtype=np.float32)
        if (
            vectors.shape != (len(texts), self.embedder.dim)
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("embedder returned invalid vectors")
        return vectors

    def write(
        self,
        text: str,
        type: str = "fact",
        metadata: Optional[dict] = None,
        id: Optional[str] = None,
        dedup_threshold: float = 0.97,
        agent: str = "",
        *,
        source: Optional[dict] = None,
    ) -> MemoryEntry:
        """Save a memory, or return its exact duplicate of the same type/source.

        Caller-supplied IDs are preserved; a duplicate ID raises ValueError.
        Semantic similarity never establishes identity. ``dedup_threshold`` is
        retained for compatibility: values above 1 disable exact deduplication.
        """
        return self.write_with_status(
            text, type, metadata, id, dedup_threshold, agent, source=source
        )[0]

    def write_with_status(
        self,
        text: str,
        type: str = "fact",
        metadata: Optional[dict] = None,
        id: Optional[str] = None,
        dedup_threshold: float = 0.97,
        agent: str = "",
        *,
        source: Optional[dict] = None,
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
        if not isinstance(dedup_threshold, (int, float)) or not np.isfinite(
            dedup_threshold
        ):
            raise ValueError("dedup_threshold must be finite")
        with self._transaction():
            entry, stored = self._append(
                text, type, metadata, id, dedup_threshold, agent, source
            )
            if stored and self.path:
                self._save_unlocked()
            return entry, stored

    def _append(self, text, type, metadata, id, dedup_threshold, agent, source=None):
        if id is not None and any(entry.id == id for entry in self._entries):
            raise ValueError(f"duplicate memory id {id!r}; use update to revise it")
        # Whitespace-only differences can be ignored; case, numbers and
        # negation can change a fact or identifier and must be preserved.
        normalized = " ".join(text.split())
        if id is None and dedup_threshold <= 1:
            for entry in self._entries:
                if (
                    entry.status == "active"
                    and entry.type == type
                    and entry.source == (source or {})
                    and " ".join(entry.text.split()) == normalized
                ):
                    return entry, False
        vec = self._embed([text])[0]
        entry = MemoryEntry(
            id=id if id is not None else self._next_id(),
            type=type,
            text=text,
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
                    return entry
                vec = (
                    self._embed([new_text])[0]
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
                return entry
            return None

    @staticmethod
    def _record_revision(entry: MemoryEntry, agent: str) -> None:
        snapshot = asdict(entry)
        snapshot.pop("history")
        entry.history = (entry.history + [snapshot])[-MAX_HISTORY:]
        entry.revision += 1
        entry.updated_at = _now_iso()
        entry.updated_by = agent

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
                text, old.type, old.metadata, None, 2, agent, source
            )
            self._record_revision(old, agent)
            old.status, old.superseded_by = "superseded", replacement.id
            if self.path:
                self._save_unlocked()
            return replacement

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Inspect an entry, including superseded entries and recent history."""
        self._reload_if_changed()
        return next((entry for entry in self._entries if entry.id == entry_id), None)

    # ---- reading -------------------------------------------------------
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
        _validate_limits(k, budget_tokens, min_score)
        if not isinstance(query, str) or len(query) > MAX_TEXT_CHARS:
            raise ValueError(
                f"query must be a string of at most {MAX_TEXT_CHARS} characters"
            )
        if type_filter is not None and type_filter not in MEMORY_TYPES:
            raise ValueError(f"unknown memory type {type_filter!r}")
        self._reload_if_changed()
        if not self._entries or k == 0 or budget_tokens == 0 or not query.strip():
            return []
        qvec = self._embed([query])[0]
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
        decay: bool = True,
    ) -> tuple[Optional[MemoryEntry], list[RecallHit]]:
        """Return the latest handoff plus relevant memories for a new session.

        The budget applies to memory content across both parts. If the latest
        handoff is too large to fit, it is skipped and the full budget remains
        available for relevant memories.
        """
        _validate_limits(k, budget_tokens, min_score)
        remaining = budget_tokens
        latest_handoff = self.latest("handoff", fresh=True)
        included_handoff: Optional[MemoryEntry] = None
        excluded_ids = {
            entry.id
            for entry in self.all()
            if entry.type in ("handoff", "worklog") and not startup_fresh(entry)
        }

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
            decay=decay,
        )
        return included_handoff, hits

    def latest(self, type: str, *, fresh: bool = False) -> Optional[MemoryEntry]:
        """Most recently written entry of a type (e.g. the last handoff)."""
        self._reload_if_changed()
        for entry in reversed(self._entries):
            if (
                entry.type == type
                and entry.status == "active"
                and (not fresh or startup_fresh(entry))
            ):
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
            "active": sum(e.status == "active" for e in self._entries),
            "superseded": sum(e.status == "superseded" for e in self._entries),
            "by_type": by_type,
            "total_tokens": sum(e.tokens for e in self._entries),
            "embedding_dim": self.embedder.dim,
            "embedder": type(self.embedder).__name__,
        }

    # ---- persistence ---------------------------------------------------
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
        """Flush a unique temporary file before atomically replacing the store."""
        target = path or self.path
        assert target is not None
        target.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for i, entry in enumerate(self._entries):
            raw = asdict(entry)
            _entry_from_raw(raw)
            raw["embedding"] = _encode_vector(self._matrix[i])
            records.append(raw)
        payload = {
            "format": STORE_FORMAT,
            "embedder": type(self.embedder).__name__,
            "embedding_config": embedding_config(self.embedder),
            "dim": self.embedder.dim,
            "entries": records,
        }
        content = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        if len(content.encode("utf-8")) > MAX_STORE_BYTES:
            raise ValueError("store exceeds the supported local file size")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=target.name + ".",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if self.path and target.resolve() == self.path.resolve():
            self._stamp = self._read_stamp()

    def load(self, path: Optional[str | Path] = None) -> None:
        target = Path(path).expanduser() if path else self.path
        if target is None or not target.exists():
            return
        if self.path and target.resolve() != self.path.resolve():
            raise ValueError("open a separate MemoryStore to load a different file")
        # Stamp before reading: a concurrent replace must trigger another load.
        stamp = self._read_stamp() if target == self.path else None
        try:
            if target.stat().st_size > MAX_STORE_BYTES:
                raise ValueError("store exceeds the supported local file size")
            payload = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(
                payload.get("entries"), list
            ):
                raise ValueError("store must contain an entries array")
            version = payload.get("format", 1)
            if (
                isinstance(version, bool)
                or not isinstance(version, int)
                or not 1 <= version <= STORE_FORMAT
            ):
                raise ValueError(f"unsupported store format {version!r}")
            reembed = (
                payload.get("dim") != self.embedder.dim
                or payload.get("embedder") != type(self.embedder).__name__
                or payload.get("embedding_config") != embedding_config(self.embedder)
            )
            entries, vectors, seen = [], [], set()
            for raw in payload["entries"]:
                entry = _entry_from_raw(raw)
                if entry.id in seen:
                    raise ValueError(f"duplicate stored memory id {entry.id!r}")
                seen.add(entry.id)
                entries.append(entry)
                embedding = raw.get("embedding")
                # Validate stored vectors even when changing models. Invalid
                # files must not be silently repaired and overwritten.
                vector = None if embedding is None else _decode_vector(embedding)
                if vector is not None and (
                    vector.ndim != 1
                    or not np.isfinite(vector).all()
                    or len(vector) != payload.get("dim")
                ):
                    raise ValueError(f"invalid embedding for {entry.id}")
                vectors.append(None if reembed else vector)
            missing = [i for i, vector in enumerate(vectors) if vector is None]
            if missing:
                fresh = self._embed([entries[i].text for i in missing])
                for slot, i in enumerate(missing):
                    vectors[i] = fresh[slot]
            matrix = (
                np.array(vectors, dtype=np.float32)
                if vectors
                else np.zeros((0, self.embedder.dim), dtype=np.float32)
            )
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            raise StoreFormatError(
                f"cannot load {target}: {exc}; original file left untouched"
            ) from exc
        self._entries, self._matrix = entries, matrix
        if target == self.path:
            self._stamp = stamp

    def _read_stamp(self) -> Optional[tuple[int, int, int]]:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return None
        except AttributeError:
            return None
        return (st.st_mtime_ns, st.st_size, st.st_ino)

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
