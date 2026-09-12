"""Scripted handoff over two real MCP stdio processes; no LLM or API key.

    python examples/handoff_demo.py

The temporary store is removed on exit. Agent names label the two sessions;
this demonstrates the protocol, not a live Claude Code or Codex session.
"""

import asyncio
from contextlib import AsyncExitStack
import json
import os
from pathlib import Path
import sys
import tempfile

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def connect(stack, path, agent):
    source = Path(__file__).resolve().parents[1] / "src"
    env = {
        "AGENT_MEMORY_PATH": str(path),
        "AGENT_MEMORY_AGENT": agent,
        "AGENT_MEMORY_EMBEDDER": "hashing",
        "PYTHONPATH": str(source) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    if os.environ.get("PYTHONPYCACHEPREFIX"):
        env["PYTHONPYCACHEPREFIX"] = os.environ["PYTHONPYCACHEPREFIX"]
    streams = await stack.enter_async_context(
        stdio_client(
            StdioServerParameters(
                command=sys.executable, args=["-m", "agent_memory.mcp_server"], env=env
            )
        )
    )
    session = await stack.enter_async_context(ClientSession(*streams))
    await session.initialize()
    return session


async def call(session, name, *, error=False, **arguments):
    result = await session.call_tool(name, arguments)
    data = result.model_dump(by_alias=True)
    assert bool(data.get("isError")) == error, data
    return "\n".join(
        block["text"] for block in data["content"] if block["type"] == "text"
    )


async def demo():
    with tempfile.TemporaryDirectory(prefix="memory-handoff-") as directory:
        async with AsyncExitStack() as stack:
            first = await connect(stack, Path(directory) / "store.json", "agent-a")
            second = await connect(stack, Path(directory) / "store.json", "agent-b")
            saved = await call(
                first,
                "memory_write",
                text="Queue delivery retries at most three times.",
                type="decision",
                source={"path": "queue/policy.py", "event": "code_review"},
            )
            entry_id = saved.split()[1]
            await call(
                first,
                "memory_handoff",
                done="Reviewed queue delivery retries.",
                next_steps="Check the retry boundary.",
            )
            context = await call(
                second, "memory_boot", task="Check queue delivery retry boundary."
            )
            assert "three times" in context and "agent-a" in context
            print("1. Agent B receives Agent A's decision and handoff over MCP.")

            await call(
                second,
                "memory_update",
                id=entry_id,
                text="Queue delivery retries at most five times.",
                expected_revision=1,
            )
            conflict = await call(
                first, "memory_forget", id=entry_id, expected_revision=1, error=True
            )
            assert "revision" in conflict and "current 2" in conflict
            print(
                "2. Agent A's stale deletion is rejected: expected revision 1, current 2."
            )

            saved = await call(
                first,
                "memory_supersede",
                id=entry_id,
                text="Queue delivery retries at most twice; failed jobs go to dead letter.",
                expected_revision=2,
            )
            new_id = saved.split()[1]
            old = json.loads(await call(second, "memory_get", id=entry_id))
            assert old["status"] == "superseded" and old["superseded_by"] == new_id
            assert [r["revision"] for r in old["history"]] == [1, 2]
            recalled = await call(
                second,
                "memory_recall",
                query="Queue delivery retries",
                budget_tokens=300,
            )
            assert (
                "twice" in recalled
                and "five times" not in recalled
                and "three times" not in recalled
            )
            print(
                "3. A replacement decision is recalled; the old decision and its revisions remain inspectable."
            )
            print(
                "All checks passed. Two server processes; one temporary store; no model calls."
            )


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(demo(), timeout=30))
