"""Minimal end-to-end example: one agent writes memories and a handoff,
a different agent boots from the same store under a token budget.

    python examples/quickstart.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_memory import MemoryStore  # noqa: E402

store = MemoryStore()  # in-memory; pass path=... to persist

# --- session 1: Claude Code learns things and hands off -----------------
store.write("Bookings are stored in UTC and shown in Asia/Tbilisi.",
            type="decision", agent="claude-code")
store.write("The chatbot uses Google Gemini in server/gemini-chat.ts.",
            type="decision", agent="claude-code")
store.write("Admin routes are guarded by requireAdmin in server/auth.ts.",
            type="decision", agent="claude-code")
store.write("Done: fixed double confirmation emails. "
            "Next: add rate limiting to the public chat endpoint.",
            type="handoff", agent="claude-code")

# --- session 2: a different agent (Codex) boots from the same store -----
task = "add rate limiting to the chat endpoint"
print(f"Task: {task}\n")

handoff = store.latest("handoff")
print(f"Last handoff [{handoff.agent}]: {handoff.text}\n")

for hit in store.recall(task, k=3, budget_tokens=100):
    print(f"  {hit.score:.2f}  [{hit.entry.type} · {hit.entry.agent}] {hit.entry.text}")
