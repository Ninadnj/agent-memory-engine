"""Command-line interface for the memory engine.

    agent-memory write "Bookings are stored in UTC." --type decision
    agent-memory recall "how are timezones handled" -k 3 --budget 200
    agent-memory handoff --done "Fixed double emails." --next "Add rate limiting."
    agent-memory boot "continue the booking bug fix"
    agent-memory list --type decision
    agent-memory update mem_0003 "Bookings are stored in UTC; UI converts." --expected-revision 1
    agent-memory forget mem_0007 --expected-revision 1
    agent-memory stats

Uses $AGENT_MEMORY_PATH or the project's .agent_memory/store.json; outside a
Git project, defaults to ~/.agent_memory/store.json.
The writing agent is taken from --agent or $AGENT_MEMORY_AGENT, so several
agents (Claude Code, Codex, Cursor) can share one store with provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .embeddings import default_min_score
from .rendering import boot_context, recall_context, empty_message
from .diagnostics import doctor, explain_recall, inspect_memory
from .store import (
    GLOBAL_STORE,
    MEMORY_TYPES,
    MemoryStore,
    default_store_path,
    find_project_root,
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
        args.text, type=args.type, agent=args.agent, source=_source(args)
    )
    if stored:
        print(f"Saved {entry.id} ({entry.type}).")
    else:
        print(f"Not saved — exact duplicate of {entry.id}: {entry.text}")


def _source(args) -> dict | None:
    source = {key: value for key, value in {
        "path": getattr(args, "source_path", None), "commit": getattr(args, "source_commit", None),
    }.items() if value is not None}
    return source or None


def cmd_recall(args) -> None:
    store = _store(args)
    options = dict(k=args.k, budget=args.budget, min_score=_min_score(args, store), decay=not args.no_decay)
    if args.explain:
        print(json.dumps(explain_recall(store, args.query, **options), indent=2))
        return
    result = recall_context(store, args.query, **options)
    result = result or empty_message("No relevant memories.", args.budget)
    if result:
        print(result, end="")


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
    result = boot_context(store, args.task, budget=args.budget, min_score=_min_score(args, store))
    if result:
        print(result, end="")


def cmd_list(args) -> None:
    from .store import age_in_days, decay_factor

    if args.limit < 0:
        raise ValueError("limit must be nonnegative")
    entries = [e for e in reversed(_store(args).all()) if not args.type or e.type == args.type]
    if not entries:
        print("No memories stored.")
        return
    for e in entries[: args.limit]:
        age = age_in_days(e)
        # Surface staleness here: this is the command you run to decide what to
        # correct or forget.
        faded = decay_factor(e)
        note = f"{age:.0f}d" + (f", faded to {faded:.0%}" if faded < 0.95 else "")
        print(f"{e.id}  [{_tag(e)} · r{e.revision} · {e.status} · {note}] {e.text}")


def cmd_update(args) -> None:
    entry = _store(args).update(args.id, text=args.text, expected_revision=args.expected_revision,
                                agent=args.agent, source=_source(args))
    print(f"Updated {entry.id}." if entry else f"No memory with id {args.id}.")


def cmd_forget(args) -> None:
    ok = _store(args).forget(args.id, expected_revision=args.expected_revision)
    print(f"Forgot {args.id}." if ok else f"No memory with id {args.id}.")


def cmd_inspect(args) -> None:
    result = inspect_memory(_store(args), args.id)
    if result is None:
        raise ValueError(f"No memory with id {args.id}.")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_supersede(args) -> None:
    entry = _store(args).supersede(args.id, args.text, expected_revision=args.expected_revision,
                                   agent=args.agent, source=_source(args))
    print(f"Saved {entry.id} (revision {entry.revision}); superseded {args.id}.")


def cmd_doctor(args) -> None:
    report = doctor(_resolve_path(args))
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")
    if not report["ok"]:
        raise SystemExit(1)


def cmd_export(args) -> None:
    _store(args).export(args.destination, overwrite=args.overwrite)
    print(f"Exported snapshot to {args.destination}")


def cmd_hook(args) -> None:
    from . import hooks

    raise SystemExit(hooks.run(args._events[args.event]))


def cmd_install_hooks(args) -> None:
    from . import hooks

    root = Path.home() if args.user else (find_project_root() or Path.cwd())
    settings = root / ".claude" / ("settings.json" if args.user else "settings.json")

    if args.uninstall:
        hooks.uninstall(settings)
        print(f"Removed agent-memory hooks from {settings}")
        return

    events = ["SessionStart", "SessionEnd"]
    if args.with_prompt_recall:
        events.append("UserPromptSubmit")
    for line in hooks.install(settings, events):
        print(f"  {line}")
    print(f"Installed into {settings}")
    if not args.with_prompt_recall:
        print(
            "Add --with-prompt-recall to also surface memories relevant to each "
            "prompt (costs a model load per message)."
        )


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
    w.add_argument("--source-path")
    w.add_argument("--source-commit")
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
    r.add_argument(
        "--no-decay",
        action="store_true",
        help="rank purely by similarity, without fading time-sensitive memories",
    )
    r.add_argument("--explain", action="store_true", help="show selection reasons (diagnostics are outside the context budget)")
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
    u.add_argument("--expected-revision", type=int, required=True)
    u.add_argument("--source-path")
    u.add_argument("--source-commit")
    u.set_defaults(func=cmd_update)

    f = sub.add_parser("forget", help="delete a memory that is wrong or stale")
    f.add_argument("id")
    f.add_argument("--expected-revision", type=int, required=True)
    f.set_defaults(func=cmd_forget)

    s = sub.add_parser("stats", help="show store stats")
    s.set_defaults(func=cmd_stats)

    inspect = sub.add_parser("inspect", help="show a memory, its source and revision history")
    inspect.add_argument("id")
    inspect.set_defaults(func=cmd_inspect)

    replace = sub.add_parser("supersede", help="replace a decision while retaining its history")
    replace.add_argument("id")
    replace.add_argument("text")
    replace.add_argument("--expected-revision", type=int, required=True)
    replace.add_argument("--source-path")
    replace.add_argument("--source-commit")
    replace.set_defaults(func=cmd_supersede)

    check = sub.add_parser("doctor", help="diagnose the active store and configuration")
    check.add_argument("--json", action="store_true")
    check.set_defaults(func=cmd_doctor)

    export = sub.add_parser("export", help="export an explicit snapshot to a different file")
    export.add_argument("destination", type=Path)
    export.add_argument("--overwrite", action="store_true")
    export.set_defaults(func=cmd_export)

    hook = sub.add_parser(
        "hook", help="internal: run a Claude Code hook (reads JSON on stdin)"
    )
    hook.add_argument(
        "event", choices=["session-start", "session-end", "user-prompt"]
    )
    hook.set_defaults(
        func=cmd_hook,
        _events={
            "session-start": "SessionStart",
            "session-end": "SessionEnd",
            "user-prompt": "UserPromptSubmit",
        },
    )

    ih = sub.add_parser(
        "install-hooks",
        help="wire memory into Claude Code so it runs without being asked",
    )
    ih.add_argument(
        "--with-prompt-recall",
        action="store_true",
        help="also inject memories relevant to each prompt (adds latency per message)",
    )
    ih.add_argument("--uninstall", action="store_true", help="remove the hooks again")
    ih.add_argument(
        "--user",
        action="store_true",
        help="install for every project (~/.claude/settings.json)",
    )
    ih.set_defaults(func=cmd_install_hooks)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"agent-memory: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
