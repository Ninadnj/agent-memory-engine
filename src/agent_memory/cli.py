"""Command-line interface for the memory engine.

    agent-memory write "Bookings are stored in UTC." --type decision
    agent-memory recall "how are timezones handled" -k 3 --budget 200
    agent-memory handoff --done "Fixed double emails." --next "Add rate limiting."
    agent-memory boot "continue the booking bug fix"
    agent-memory stats

Uses a JSON store at $AGENT_MEMORY_PATH (default ~/.agent_memory/store.json).
The writing agent is taken from --agent or $AGENT_MEMORY_AGENT, so several
agents (Claude Code, Codex, Cursor) can share one store with provenance.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .store import MEMORY_TYPES, MemoryStore

DEFAULT_STORE = Path(
    os.environ.get("AGENT_MEMORY_PATH", "~/.agent_memory/store.json")
).expanduser()
DEFAULT_AGENT = os.environ.get("AGENT_MEMORY_AGENT", "")


def _store(args) -> MemoryStore:
    return MemoryStore(path=args.path)


def _tag(entry) -> str:
    return f"{entry.type} · {entry.agent}" if entry.agent else entry.type


def cmd_write(args) -> None:
    entry = _store(args).write(args.text, type=args.type, agent=args.agent)
    print(f"Saved {entry.id} ({entry.type}).")


def cmd_recall(args) -> None:
    hits = _store(args).recall(args.query, k=args.k, budget_tokens=args.budget)
    if not hits:
        print("No memories yet.")
        return
    for h in hits:
        print(f"{h.score:.2f}  [{_tag(h.entry)}] {h.entry.text}")


def cmd_handoff(args) -> None:
    text = f"Done: {args.done} Next: {args.next}"
    if args.watch_out:
        text += f" Watch out: {args.watch_out}"
    entry = _store(args).write(text, type="handoff", agent=args.agent)
    print(f"Handoff saved ({entry.id}).")


def cmd_boot(args) -> None:
    store = _store(args)
    handoff = store.latest("handoff")
    remaining = args.budget
    if handoff is not None:
        print(f"Last handoff [{_tag(handoff)}]: {handoff.text}")
        if remaining is not None:
            remaining = max(0, remaining - handoff.tokens)
    hits = store.recall(args.task, k=5, budget_tokens=remaining)
    for h in hits:
        if handoff is not None and h.entry.id == handoff.id:
            continue
        print(f"- [{_tag(h.entry)}] {h.entry.text}")


def cmd_stats(args) -> None:
    s = _store(args).stats()
    print(f"{s['count']} memories | {s['total_tokens']} tokens | {s['embedder']}")
    for t, n in sorted(s["by_type"].items()):
        print(f"  {t}: {n}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-memory")
    parser.add_argument("--path", type=Path, default=DEFAULT_STORE)
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
    r.set_defaults(func=cmd_recall)

    h = sub.add_parser("handoff", help="save a handoff for the next agent")
    h.add_argument("--done", required=True)
    h.add_argument("--next", required=True)
    h.add_argument("--watch-out", default="")
    h.set_defaults(func=cmd_handoff)

    b = sub.add_parser("boot", help="latest handoff + relevant memories")
    b.add_argument("task")
    b.add_argument("--budget", type=int, default=300, help="max context tokens")
    b.set_defaults(func=cmd_boot)

    s = sub.add_parser("stats", help="show store stats")
    s.set_defaults(func=cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
