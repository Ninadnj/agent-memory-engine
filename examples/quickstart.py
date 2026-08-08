"""Minimal end-to-end example: one agent writes memories and a handoff,
a different agent boots from the same store under a token budget.

    python examples/quickstart.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_memory import MemoryStore, default_min_score  # noqa: E402

store = MemoryStore()  # in-memory; pass path=... to persist
floor = default_min_score(store.embedder)

# --- session 1: Claude Code learns things and hands off -----------------
store.write("Bookings are stored in UTC and shown in Asia/Tbilisi.",
            type="decision", agent="claude-code")
store.write("The chatbot uses Google Gemini in server/gemini-chat.ts.",
            type="decision", agent="claude-code")
store.write("Admin routes are guarded by requireAdmin in server/auth.ts.",
            type="decision", agent="claude-code")
store.write("The public chat endpoint /api/chat has no rate limiter; the Gemini "
            "API key is exposed to abuse until one is added.",
            type="issue", agent="claude-code")
store.write("Done: fixed double confirmation emails. "
            "Next: add rate limiting to the public chat endpoint.",
            type="handoff", agent="claude-code")

# Writing the same thing twice does not duplicate it, and says so.
_, stored = store.write_with_status(
    "Admin routes are guarded by requireAdmin in server/auth.ts.",
    type="decision", agent="claude-code")
print(f"Re-writing a known fact stored a new memory? {stored}\n")

# --- session 2: a different agent (Codex) boots from the same store -----
task = "add rate limiting to the chat endpoint"
print(f"Task: {task}\n")

handoff, hits = store.boot(task, k=3, budget_tokens=100, min_score=floor)
if handoff:
    print(f"Last handoff [{handoff.agent}]: {handoff.text}\n")
for hit in hits:
    print(f"  {hit.score:.2f}  [{hit.entry.type} · {hit.entry.agent}] {hit.entry.text}")

# --- a task the store knows nothing about --------------------------------
_, unrelated = store.boot("choose a color palette for the print brochure",
                          k=3, budget_tokens=100, min_score=floor)
print(f"\nMemories recalled for an unrelated task: {len(unrelated)} "
      f"(the relevance floor is {floor}; without it this would return 3)")

# --- correcting memory ----------------------------------------------------
stale = store.write("Deploys are triggered from the old Jenkins box.", type="state")
store.update(stale.id, text="Deploys are triggered from GitHub Actions.")
store.forget(stale.id)
print(f"Memories after update + forget: {store.stats()['count']}")
