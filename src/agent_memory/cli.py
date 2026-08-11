"""Command-line interface for the memory engine.

    agent-memory write "Bookings are stored in UTC." --type decision
    agent-memory recall "how are timezones handled" -k 3 --budget 200
    agent-memory handoff --done "Fixed double emails." --next "Add rate limiting."
    agent-memory boot "continue the booking bug fix"
    agent-memory list --type decision
    agent-memory update mem_0003 "Bookings are stored in UTC; UI converts."
    agent-memory forget mem_0007
    agent-memory stats

Uses a JSON store at $AGENT_MEMORY_PATH (default ~/.agent_memory/store.json).
The writing agent is taken from --agent or $AGENT_MEMORY_AGENT, so several
agents (Claude Code, Codex, Cursor) can share one store with provenance.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .embeddings import default_min_score
from .store import (
    GLOBAL_STORE,
    MEMORY_TYPES,
    MemoryStore,
    default_store_path,
    relocation_notice,
)

DEFAULT_AGENT = os.environ.get("AGENT_MEMORY_AGENT", "")

# Sentinel: resolved per-embedder once the store is open (see default_min_score).
AUTO_MIN_SCORE = -1.0


def _resolve_path(args) -> Path:
    """Explicit --path, then --global, then the project/global default."""
    if args.path is not None:
        return Path(args.path).expanduser()
    if getattr(args, "use_global", False):
        return GLOBAL_STORE
    path = default_store_path()
    notice = relocation_notice(path)
    if notice:
        print(notice, file=sys.stderr)
    return path


def _store(args) -> MemoryStore:
    return MemoryStore(path=_resolve_path(args))


def _min_score(args, store: MemoryStore) -> float:
    if getattr(args, "min_score", AUTO_MIN_SCORE) != AUTO_MIN_SCORE:
        return args.min_score
    return default_min_score(store.embedder)


def _tag(entry) -> str:
    return f"{entry.type} · {entry.agent}" if entry.agent else entry.type


def cmd_write(args) -> None:
    entry, stored = _store(args).write_with_status(
        args.text, type=args.type, agent=args.agent
    )
    if stored:
        print(f"Saved {entry.id} ({entry.type}).")
    else:
        print(f"Not saved — near-duplicate of {entry.id}: {entry.text}")


def cmd_recall(args) -> None:
    store = _store(args)
    hits = store.recall(
        args.query,
        k=args.k,
        budget_tokens=args.budget,
        min_score=_min_score(args, store),
    )
    if not hits:
        print("No relevant memories.")
        return
    for h in hits:
        print(f"{h.score:.2f}  [{_tag(h.entry)}] {h.entry.text}")


def cmd_handoff(args) -> None:
    text = f"Done: {args.done} Next: {args.next}"
    if args.watch_out:
        text += f" Watch out: {args.watch_out}"
    entry, stored = _store(args).write_with_status(
        text, type="handoff", agent=args.agent
    )
    if stored:
        print(f"Handoff saved ({entry.id}).")
    else:
        print(f"Identical handoff already stored as {entry.id}; nothing written.")


def cmd_boot(args) -> None:
    store = _store(args)
    handoff, hits = store.boot(
        args.task, k=5, budget_tokens=args.budget, min_score=_min_score(args, store)
    )
    if handoff is not None:
        print(f"Last handoff [{_tag(handoff)}]: {handoff.text}")
    for h in hits:
        print(f"- [{_tag(h.entry)}] {h.entry.text}")


def cmd_list(args) -> None:
    entries = [e for e in reversed(_store(args).all()) if not args.type or e.type == args.type]
    if not entries:
        print("No memories stored.")
        return
    for e in entries[: args.limit]:
        print(f"{e.id}  [{_tag(e)}] {e.text}")


def cmd_update(args) -> None:
    entry = _store(args).update(args.id, text=args.text)
    print(f"Updated {entry.id}." if entry else f"No memory with id {args.id}.")


def cmd_forget(args) -> None:
    ok = _store(args).forget(args.id)
    print(f"Forgot {args.id}." if ok else f"No memory with id {args.id}.")


def cmd_stats(args) -> None:
    path = _resolve_path(args)
    s = MemoryStore(path=path).stats()
    print(f"store: {path}")
    print(f"{s['count']} memories | {s['total_tokens']} tokens | {s['embedder']}")
    for t, n in sorted(s["by_type"].items()):
        print(f"  {t}: {n}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-memory")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="store file (default: this project's .agent_memory/store.json)",
    )
    parser.add_argument(
        "--global",
        dest="use_global",
        action="store_true",
        help=f"use the cross-project store at {GLOBAL_STORE}",
    )
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="who is writing")
    sub = parser.add_subparsers(dest="command", required=True)

    w = sub.add_parser("write", help="save a memory")
    w.add_argument("text")
    w.add_argument("--type", default="fact", choices=sorted(MEMORY_TYPES))
    w.set_defaults(func=cmd_write)

    r = sub.add_parser("recall", help="recall relevant memories")
    r.add_argument("query")
    r.add_argument("-k", type=int, default=5)
    r.add_argument("--budget", type=int, default=None, help="max context tokens")
    r.add_argument(
        "--min-score",
        type=float,
        default=AUTO_MIN_SCORE,
        dest="min_score",
        help="drop matches weaker than this cosine score (0 disables; "
        "default is calibrated per embedder)",
    )
    r.set_defaults(func=cmd_recall)

    h = sub.add_parser("handoff", help="save a handoff for the next agent")
    h.add_argument("--done", required=True)
    h.add_argument("--next", required=True)
    h.add_argument("--watch-out", default="")
    h.set_defaults(func=cmd_handoff)

    b = sub.add_parser("boot", help="latest handoff + relevant memories")
    b.add_argument("task")
    b.add_argument("--budget", type=int, default=300, help="max context tokens")
    b.add_argument(
        "--min-score", type=float, default=AUTO_MIN_SCORE, dest="min_score"
    )
    b.set_defaults(func=cmd_boot)

    ls = sub.add_parser("list", help="list memories with their ids")
    ls.add_argument("--type", default="", choices=[""] + sorted(MEMORY_TYPES))
    ls.add_argument("--limit", type=int, default=20)
    ls.set_defaults(func=cmd_list)

    u = sub.add_parser("update", help="replace the text of a memory")
    u.add_argument("id")
    u.add_argument("text")
    u.set_defaults(func=cmd_update)

    f = sub.add_parser("forget", help="delete a memory that is wrong or stale")
    f.add_argument("id")
    f.set_defaults(func=cmd_forget)

    s = sub.add_parser("stats", help="show store stats")
    s.set_defaults(func=cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
