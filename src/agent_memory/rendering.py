"""Budget the actual returned text, including labels and provenance.

Accounting uses cl100k_base when available, otherwise the documented
approximation. It excludes protocol/tool schemas and client-added wrappers.
"""

from .store import MemoryStore, _validate_limits
from .tokens import count_tokens


def tag(entry) -> str:
    return f"{entry.type} · {entry.agent}" if entry.agent else entry.type


def reference(entry) -> str:
    parts = [entry.id, f"r{entry.revision}", (entry.updated_at or entry.created_at)[:10]]
    if entry.source.get("path"):
        parts.append(str(entry.source["path"]))
    if entry.source.get("commit"):
        parts.append(str(entry.source["commit"])[:12])
    return " · ".join(parts)


def render_entry(entry, *, identity=True) -> str:
    body = f"- [{tag(entry)}] {entry.text}"
    return f"{body} ({reference(entry)})" if identity else body


def pack_blocks(blocks, budget, *, limit=None) -> str:
    _validate_limits(0, budget, 0)
    parts = []
    for block in blocks:
        candidate = "\n".join(parts + [block])
        if budget is not None and count_tokens(candidate) > budget:
            continue
        parts.append(block)
        if limit is not None and len(parts) >= limit:
            break
    return "\n".join(parts)


def recall_context(store: MemoryStore, query: str, *, k=5, budget=300, min_score=0.0,
                   decay=True, identity=True) -> str:
    _validate_limits(k, budget, min_score)
    if k == 0:
        return ""
    hits = store.recall(query, k=max(1, len(store.all())), min_score=min_score, decay=decay)
    return pack_blocks((render_entry(hit.entry, identity=identity) for hit in hits), budget, limit=k)


def boot_context(store: MemoryStore, task: str, *, budget=300, min_score=0.0) -> str:
    handoff, hits = store.boot(task, k=max(1, len(store.all())), budget_tokens=None, min_score=min_score)
    blocks = []
    if handoff:
        blocks.append(f"Last handoff [{tag(handoff)}]: {handoff.text} ({reference(handoff)})")
    blocks.extend(render_entry(hit.entry) for hit in hits)
    return pack_blocks(blocks, budget, limit=5 + bool(handoff))


def empty_message(message: str, budget) -> str:
    return pack_blocks([message], budget)
