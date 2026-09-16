"""Memory records, validation and freshness rules; no filesystem or model loading."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

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

MAX_TEXT_CHARS = 20_000
MAX_METADATA_BYTES = 16_384
MAX_HISTORY = 20


class MemoryConflictError(ValueError):
    """The caller's revision or store snapshot is no longer current."""


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
        or not math.isfinite(min_score)
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
