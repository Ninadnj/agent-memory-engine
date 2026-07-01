"""MCP server exposing the memory engine as tools an agent can call.

One store, many agents: register this same server (stdio) in Claude Code,
Codex CLI and Cursor, and they all read and write the same memory. Each agent
identifies itself via the AGENT_MEMORY_AGENT env var, so every memory carries
its origin and a handoff written by one agent is picked up by the next.

The tool outputs are deliberately compact — recall is token-budgeted and
scores/timestamps are omitted — because everything a memory tool returns is
paid for again in the calling agent's context window.

Requires the optional `mcp` dependency (`pip install "agent-memory-engine[mcp]"`).
The imports are deferred so the rest of the package works without it.

Run:  python -m agent_memory.mcp_server
Or register it (stdio) in your MCP client config — see the README.
"""

from __future__ import annotations

import os
from pathlib import Path

from .store import MEMORY_TYPES, MemoryStore

DEFAULT_STORE = Path(
    os.environ.get("AGENT_MEMORY_PATH", "~/.agent_memory/store.json")
).expanduser()

# Who is talking to the store — "claude-code", "codex", "cursor", ...
DEFAULT_AGENT = os.environ.get("AGENT_MEMORY_AGENT", "")


def _tag(entry) -> str:
    return f"{entry.type} · {entry.agent}" if entry.agent else entry.type


def _render(hits) -> str:
    return "\n".join(f"- [{_tag(h.entry)}] {h.entry.text}" for h in hits)


def build_server(store_path: Path = DEFAULT_STORE, agent: str = DEFAULT_AGENT):
    """Construct the FastMCP server. Imports `mcp` lazily so importing this
    module never hard-fails when the optional dependency is absent."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "The MCP server needs the 'mcp' package. "
            'Install it with:  pip install "agent-memory-engine[mcp]"'
        ) from exc

    store = MemoryStore(path=store_path)
    server = FastMCP("agent-memory")

    @server.tool()
    def memory_write(text: str, type: str = "fact") -> str:
        """Save one durable memory. `type` is one of: project, decision, issue,
        state, handoff, worklog, fact. Near-duplicates are skipped."""
        if type not in MEMORY_TYPES:
            return f"Error: type must be one of {sorted(MEMORY_TYPES)}."
        entry = store.write(text, type=type, agent=agent)
        return f"Saved {entry.id} ({entry.type})."

    @server.tool()
    def memory_recall(query: str, k: int = 5, budget_tokens: int = 300) -> str:
        """Recall the most relevant memories for `query`, never exceeding
        `budget_tokens` of context. Set budget_tokens=0 for no cap."""
        hits = store.recall(query, k=k, budget_tokens=budget_tokens or None)
        return _render(hits) if hits else "No relevant memories."

    @server.tool()
    def memory_boot(task: str, budget_tokens: int = 300) -> str:
        """Call once at the start of a session: returns the latest handoff from
        the previous agent plus the memories most relevant to `task`, packed
        under one token budget."""
        parts: list[str] = []
        remaining = budget_tokens
        handoff = store.latest("handoff")
        if handoff is not None:
            parts.append(f"Last handoff [{_tag(handoff)}]: {handoff.text}")
            remaining = max(0, remaining - handoff.tokens)
        hits = store.recall(task, k=5, budget_tokens=remaining or None)
        hits = [h for h in hits if handoff is None or h.entry.id != handoff.id]
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
        entry = store.write(text, type="handoff", agent=agent)
        return f"Handoff saved ({entry.id}). The next agent gets it via memory_boot."

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
