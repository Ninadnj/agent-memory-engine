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


def test_recall_can_exclude_an_entry(store):
    excluded = store.all()[0]
    hits = store.recall("bookings UTC", k=3, exclude_ids={excluded.id})
    assert all(hit.entry.id != excluded.id for hit in hits)


def test_boot_shares_one_budget_between_handoff_and_recall(store):
    handoff = store.write(
        "Done: fixed email retries. Next: protect admin routes.",
        type="handoff",
        agent="claude-code",
    )
    budget = handoff.tokens + 20

    included_handoff, hits = store.boot(
        "protect admin routes",
        k=5,
        budget_tokens=budget,
    )

    assert included_handoff == handoff
    assert handoff.tokens + sum(hit.entry.tokens for hit in hits) <= budget
    assert all(hit.entry.id != handoff.id for hit in hits)


def test_boot_skips_handoff_that_cannot_fit(store):
    oversized = store.write(
        ("long handoff " * 100) + "next deploy safely",
        type="handoff",
        agent="claude-code",
    )
    budget = 25

    included_handoff, hits = store.boot(
        "protect admin routes",
        k=5,
        budget_tokens=budget,
    )

    assert oversized.tokens > budget
    assert included_handoff is None
    assert sum(hit.entry.tokens for hit in hits) <= budget
    assert all(hit.entry.id != oversized.id for hit in hits)


def test_min_score_drops_unrelated_matches(store):
    """Without a floor, a query about nothing in the store still returns k
    memories — and the MCP layer hides the scores, so the agent can't tell."""
    assert store.recall("how do I bake sourdough bread at home", k=3) != []
    assert store.recall("how do I bake sourdough bread at home", k=3, min_score=0.15) == []


def test_min_score_keeps_genuine_matches(store):
    hits = store.recall("how do we protect admin pages", k=3, min_score=0.15)
    assert hits and hits[0].entry.text.startswith("Admin routes")


def test_write_with_status_reports_whether_it_stored(store):
    text = "Deploys go out through GitHub Actions on every push to main."
    entry, stored = store.write_with_status(text, type="decision")
    assert stored is True
    same, stored_again = store.write_with_status(text, type="decision")
    assert stored_again is False and same.id == entry.id


def test_forget_removes_the_entry_from_recall(store):
    entry = store.write("A stale note about the retired staging box.", type="state")
    assert store.forget(entry.id) is True
    assert all(h.entry.id != entry.id for h in store.recall("staging box", k=5))
    assert store.forget(entry.id) is False


def test_update_changes_text_and_ranking(store):
    entry = store.write("The API listens on port 5002.", type="project")
    updated = store.update(entry.id, text="The API listens on port 8080.")
    assert updated is not None and "8080" in updated.text
    assert "8080" in store.recall("which port does the API listen on", k=1)[0].entry.text


def test_update_returns_none_for_unknown_id(store):
    assert store.update("mem_9999", text="whatever") is None


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
