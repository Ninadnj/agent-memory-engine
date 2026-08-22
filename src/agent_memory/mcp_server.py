"""MCP server exposing the memory engine as tools an agent can call.

One store, many agents: register this same server (stdio) in Claude Code,
Codex CLI and Cursor, and they all read and write the same memory. Each agent
identifies itself via the AGENT_MEMORY_AGENT env var, so every memory carries
its origin and a handoff written by one agent is picked up by the next.

The tool outputs are deliberately compact — recall is token-budgeted and
scores/timestamps are omitted — because everything a memory tool returns is
paid for again in the calling agent's context window. The flip side is that the
agent cannot judge relevance itself, so weak matches are filtered out here
rather than passed along unlabelled.

Requires the optional `mcp` dependency (`pip install "agent-memory-engine[mcp]"`).
The imports are deferred so the rest of the package works without it.

Run:  python -m agent_memory.mcp_server
Or register it (stdio) in your MCP client config — see the README.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from .embeddings import default_min_score
from .store import MEMORY_TYPES, MemoryStore, default_store_path, relocation_notice
from .tokens import count_tokens

# Who is talking to the store — "claude-code", "codex", "cursor", ...
DEFAULT_AGENT = os.environ.get("AGENT_MEMORY_AGENT", "")


def _load_server_class():
    """Return the FastMCP-style server class across `mcp` major versions.

    `mcp` 1.x exposes it as `mcp.server.fastmcp.FastMCP`; 2.x renamed the module
    and the class to `mcp.server.mcpserver.MCPServer`. Both take a name and
    provide `.tool()` and `.run()`, so the rest of this module is unchanged.
    """
    try:
        from mcp.server.fastmcp import FastMCP  # mcp 1.x

        return FastMCP
    except ImportError:
        pass
    try:
        from mcp.server.mcpserver import MCPServer  # mcp 2.x

        return MCPServer
    except ImportError as exc:
        raise SystemExit(
            "Could not load an MCP server class from the installed 'mcp' package "
            f"({exc}). Install a supported version with:  "
            'pip install "agent-memory-engine[mcp]"'
        ) from exc


# Sent to the client at initialize, so the model learns the workflow without
# anyone editing a CLAUDE.md. This is the vendor-neutral half of "memory should
# not depend on the model remembering": Claude Code can enforce the same habits
# with hooks, but Codex and Cursor have no equivalent, and they read this.
SERVER_INSTRUCTIONS = """\
Durable project memory shared across coding agents and sessions.

Use it like this:

1. At the start of a session, call memory_boot with a one-line description of
   the task. It returns the previous session's handoff plus relevant memories.
   Do this before exploring the codebase — it often answers the question first.
2. While working, call memory_write the moment you learn something that will
   still be true next week: an architectural decision, a non-obvious constraint,
   a bug's root cause. Write one fact per call, in one or two sentences. Do not
   write things that are already obvious from reading the code.
3. Before the session ends, call memory_handoff with what you finished, what
   comes next, and anything the next agent should watch out for.

If a memory turns out to be wrong or stale, fix it with memory_update or delete
it with memory_forget rather than writing a second, contradicting memory — both
would be recalled together. Find ids with memory_list.
"""


def _tag(entry) -> str:
    return f"{entry.type} · {entry.agent}" if entry.agent else entry.type


def _pack_lines(lines, budget_tokens: int, limit: int) -> str:
    """Pack complete rendered lines so protocol output respects the real budget."""
    if limit <= 0:
        return ""
    selected: list[str] = []
    for line in lines:
        if len(selected) >= limit:
            break
        candidate = "\n".join([*selected, line])
        if budget_tokens != 0 and count_tokens(candidate) > budget_tokens:
            continue
        selected.append(line)
    return "\n".join(selected)


def _budgeted_message(message: str, budget_tokens: int) -> str:
    if budget_tokens == 0 or count_tokens(message) <= budget_tokens:
        return message
    return ""


def build_server(
    store_path: Optional[Path] = None,
    agent: str = DEFAULT_AGENT,
    min_score: Optional[float] = None,
):
    """Construct the MCP server. Imports `mcp` lazily so importing this module
    never hard-fails when the optional dependency is absent.

    With no `store_path`, the store is resolved per project: the agent launches
    this server from the working directory, so `.agent_memory/store.json` in the
    enclosing repository is the store its memories belong to.
    """
    server_class = _load_server_class()

    if store_path is None:
        store_path = default_store_path()
        notice = relocation_notice(store_path)
        if notice:
            # stdout is the JSON-RPC channel; anything printed there corrupts it.
            print(notice, file=sys.stderr)

    store = MemoryStore(path=store_path)
    if min_score is None:
        min_score = default_min_score(store.embedder)
    try:
        server = server_class("agent-memory", instructions=SERVER_INSTRUCTIONS)
    except TypeError:  # older mcp releases have no `instructions` parameter
        server = server_class("agent-memory")

    @server.tool()
    def memory_write(text: str, type: str = "fact") -> str:
        """Save one durable memory. `type` is one of: project, decision, issue,
        state, handoff, worklog, fact. Near-duplicates are skipped."""
        if type not in MEMORY_TYPES:
            return f"Error: type must be one of {sorted(MEMORY_TYPES)}."
        entry, stored = store.write_with_status(text, type=type, agent=agent)
        if not stored:
            return (
                f"Not saved — near-duplicate of {entry.id}: {entry.text!r} "
                "Use memory_update to revise it if this supersedes it."
            )
        return f"Saved {entry.id} ({entry.type})."

    @server.tool()
    def memory_recall(query: str, k: int = 5, budget_tokens: int = 300) -> str:
        """Recall the most relevant memories for `query`, never exceeding
        `budget_tokens` of context. Set budget_tokens=0 for no cap."""
        candidate_count = len(store.all())
        hits = store.recall(
            query,
            k=max(k, candidate_count),
            budget_tokens=None,
            min_score=min_score,
        )
        rendered = _pack_lines(
            (f"- [{_tag(hit.entry)}] {hit.entry.text}" for hit in hits),
            budget_tokens,
            k,
        )
        return rendered or _budgeted_message("No relevant memories.", budget_tokens)

    @server.tool()
    def memory_boot(task: str, budget_tokens: int = 300) -> str:
        """Call once at the start of a session: returns the latest handoff from
        the previous agent plus the memories most relevant to `task`, packed
        under one rendered-output token budget. Set budget_tokens=0 for no cap."""
        candidate_count = len(store.all())
        handoff, hits = store.boot(
            task,
            k=max(5, candidate_count),
            budget_tokens=None,
            min_score=min_score,
        )
        parts: list[str] = []
        if handoff is not None:
            handoff_line = f"Last handoff [{_tag(handoff)}]: {handoff.text}"
            packed = _pack_lines([handoff_line], budget_tokens, 1)
            if packed:
                parts.append(packed)
        remaining_hits = 5
        for hit in hits:
            if remaining_hits <= 0:
                break
            line = f"- [{_tag(hit.entry)}] {hit.entry.text}"
            candidate = "\n".join([*parts, line])
            if budget_tokens != 0 and count_tokens(candidate) > budget_tokens:
                continue
            parts.append(line)
            remaining_hits -= 1
        if parts:
            return "\n".join(parts)
        return _budgeted_message("Empty store — start fresh.", budget_tokens)

    @server.tool()
    def memory_handoff(done: str, next_steps: str, warnings: str = "") -> str:
        """Call at the end of a session so the next agent (any tool, any
        vendor) can continue. Keep each part to one or two sentences."""
        text = f"Done: {done} Next: {next_steps}"
        if warnings:
            text += f" Watch out: {warnings}"
        entry, stored = store.write_with_status(text, type="handoff", agent=agent)
        if not stored:
            return f"Identical handoff already stored as {entry.id}; nothing written."
        return f"Handoff saved ({entry.id}). The next agent gets it via memory_boot."

    @server.tool()
    def memory_update(id: str, text: str) -> str:
        """Replace the text of an existing memory. Use this when a fact changes,
        instead of writing a second memory that contradicts the first."""
        entry = store.update(id, text=text)
        if entry is None:
            return f"No memory with id {id}."
        return f"Updated {entry.id} ({entry.type})."

    @server.tool()
    def memory_forget(id: str) -> str:
        """Delete a memory that is wrong or has gone stale. Find ids with
        memory_list."""
        return (
            f"Forgot {id}." if store.forget(id) else f"No memory with id {id}."
        )

    @server.tool()
    def memory_list(type: str = "", limit: int = 20) -> str:
        """List stored memories with their ids, newest first, so they can be
        updated or forgotten. Optionally filter by `type`."""
        entries = [e for e in reversed(store.all()) if not type or e.type == type]
        if not entries:
            return "No memories stored."
        shown = entries[:limit]
        lines = [f"- {e.id} [{_tag(e)}] {e.text}" for e in shown]
        if len(entries) > len(shown):
            lines.append(f"... and {len(entries) - len(shown)} more.")
        return "\n".join(lines)

    @server.tool()
    def memory_stats() -> str:
        """Summarize what is in the memory store."""
        s = store.stats()
        by_type = ", ".join(f"{k}={v}" for k, v in sorted(s["by_type"].items()))
        return (
            f"{s['count']} memories ({by_type or 'none'}); "
            f"{s['total_tokens']} tokens total; embedder {s['embedder']}."
        )

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
