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
from pathlib import Path
from typing import Optional

from .embeddings import default_min_score
from .store import MEMORY_TYPES, MemoryStore

DEFAULT_STORE = Path(
    os.environ.get("AGENT_MEMORY_PATH", "~/.agent_memory/store.json")
).expanduser()

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


def _tag(entry) -> str:
    return f"{entry.type} · {entry.agent}" if entry.agent else entry.type


def _render(hits) -> str:
    return "\n".join(f"- [{_tag(h.entry)}] {h.entry.text}" for h in hits)


def build_server(
    store_path: Path = DEFAULT_STORE,
    agent: str = DEFAULT_AGENT,
    min_score: Optional[float] = None,
):
    """Construct the MCP server. Imports `mcp` lazily so importing this module
    never hard-fails when the optional dependency is absent."""
    server_class = _load_server_class()

    store = MemoryStore(path=store_path)
    if min_score is None:
        min_score = default_min_score(store.embedder)
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
        hits = store.recall(
            query, k=k, budget_tokens=budget_tokens or None, min_score=min_score
        )
        return _render(hits) if hits else "No relevant memories."

    @server.tool()
    def memory_boot(task: str, budget_tokens: int = 300) -> str:
        """Call once at the start of a session: returns the latest handoff from
        the previous agent plus the memories most relevant to `task`, packed
        under one memory-content token budget."""
        parts: list[str] = []
        handoff, hits = store.boot(
            task, k=5, budget_tokens=budget_tokens, min_score=min_score
        )
        if handoff is not None:
            parts.append(f"Last handoff [{_tag(handoff)}]: {handoff.text}")
        if hits:
            parts.append(_render(hits))
        return "\n".join(parts) if parts else "Empty store — start fresh."

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
