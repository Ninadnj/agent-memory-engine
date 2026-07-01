import pytest

from agent_memory import HashingEmbedder, MemoryStore


@pytest.fixture
def store():
    s = MemoryStore(embedder=HashingEmbedder())
    s.write("Bookings are stored in UTC and shown in Asia/Tbilisi.", type="decision")
    s.write("The chatbot uses Google Gemini in server/gemini-chat.ts.", type="decision")
    s.write("Admin routes are guarded by requireAdmin in server/auth.ts.", type="decision")
    return s


def test_recall_ranks_relevant_first(store):
    hits = store.recall("how do we protect admin pages", k=1)
    assert hits[0].entry.text.startswith("Admin routes")


def test_recall_respects_k(store):
    assert len(store.recall("anything", k=2)) == 2


def test_type_filter(store):
    store.write("Next: add rate limiting to the chat endpoint.", type="handoff")
    hits = store.recall("what is the next step", k=5, type_filter="handoff")
    assert hits and all(h.entry.type == "handoff" for h in hits)


def test_dedup_skips_near_duplicates(store):
    before = store.stats()["count"]
    store.write("Bookings are stored in UTC and shown in Asia/Tbilisi.", type="decision")
    assert store.stats()["count"] == before


def test_rejects_unknown_type(store):
    with pytest.raises(ValueError):
        store.write("something", type="not_a_type")


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "store.json"
    s = MemoryStore(path=path, embedder=HashingEmbedder())
    s.write("PostgreSQL via Drizzle ORM; schema in shared/schema.ts.", type="project")
    s.write("Images go to object storage; ACL in server/objectAcl.ts.", type="project")

    reloaded = MemoryStore(path=path, embedder=HashingEmbedder())
    assert reloaded.stats()["count"] == 2
    hits = reloaded.recall("where is the database schema", k=1)
    assert "Drizzle" in hits[0].entry.text


def test_budget_recall_never_exceeds_cap(store):
    for i in range(5):
        store.write(f"Extra fact number {i} about deployment and CI pipelines.")
    budget = 25
    hits = store.recall("deployment", k=10, budget_tokens=budget)
    assert hits, "should still retrieve something under the budget"
    assert sum(h.entry.tokens for h in hits) <= budget


def test_budget_recall_skips_oversized_and_packs_smaller(store):
    big = "word " * 200  # far over any small budget
    store.write("Deploys go out via GitHub Actions." , type="decision")
    store.write(big + "deployment", type="decision")
    hits = store.recall("how do we deploy", k=10, budget_tokens=30)
    texts = [h.entry.text for h in hits]
    assert all(len(t) < 500 for t in texts), "oversized entry must be skipped"


def test_agent_attribution_roundtrip(tmp_path):
    from agent_memory import HashingEmbedder, MemoryStore

    path = tmp_path / "store.json"
    s = MemoryStore(path=path, embedder=HashingEmbedder())
    s.write("Refactored the booking router.", type="worklog", agent="claude-code")
    reloaded = MemoryStore(path=path, embedder=HashingEmbedder())
    assert reloaded.all()[0].agent == "claude-code"


def test_latest_returns_most_recent_handoff(store):
    store.write("Done: A. Next: B.", type="handoff", agent="claude-code")
    store.write("Done: B. Next: C.", type="handoff", agent="codex")
    latest = store.latest("handoff")
    assert latest is not None and latest.agent == "codex"
    assert store.latest("worklog") is None
