"""Time-sensitive memories fade; durable ones do not.

A month-old "currently implementing X" is usually false, and recalling it with
the same confidence as a fresh fact actively misleads the next session. A
decision, by contrast, stays true until someone supersedes it — fading it would
quietly lose the memories most worth keeping.
"""

from datetime import datetime, timedelta, timezone

import pytest

from agent_memory import HashingEmbedder, MemoryStore, age_in_days, decay_factor
from agent_memory.store import HALF_LIFE_DAYS, MemoryEntry


def aged(entry_type: str, text: str, days: float) -> MemoryEntry:
    written = datetime.now(timezone.utc) - timedelta(days=days)
    return MemoryEntry(
        id="mem_test",
        type=entry_type,
        text=text,
        created_at=written.isoformat(timespec="seconds"),
    )


def backdate(store: MemoryStore, entry_id: str, days: float) -> None:
    """Rewrite an entry's timestamp, as if it had been written `days` ago."""
    written = datetime.now(timezone.utc) - timedelta(days=days)
    for entry in store.all():
        if entry.id == entry_id:
            entry.created_at = written.isoformat(timespec="seconds")
            return
    raise AssertionError(f"no entry {entry_id}")


# ---- the decay curve -------------------------------------------------------
def test_a_fresh_memory_is_not_faded():
    # Timestamps are stored to the second, so "now" can already be a second old.
    # Anything written today is unfaded for practical purposes.
    assert decay_factor(aged("state", "x", 0)) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("entry_type", [t for t, hl in HALF_LIFE_DAYS.items() if hl])
def test_a_memory_at_its_half_life_is_worth_half(entry_type):
    half_life = HALF_LIFE_DAYS[entry_type]
    assert decay_factor(aged(entry_type, "x", half_life)) == pytest.approx(0.5, abs=0.01)
    assert decay_factor(aged(entry_type, "x", half_life * 2)) == pytest.approx(0.25, abs=0.01)


@pytest.mark.parametrize("entry_type", [t for t, hl in HALF_LIFE_DAYS.items() if not hl])
def test_durable_types_never_fade(entry_type):
    assert decay_factor(aged(entry_type, "x", 3650)) == 1.0


def test_decay_never_reaches_zero():
    """A very old memory should rank last, not become unreachable."""
    assert 0 < decay_factor(aged("state", "x", 3650)) < 1e-6


# ---- robustness ------------------------------------------------------------
def test_an_unreadable_timestamp_does_not_bury_a_memory():
    entry = MemoryEntry(id="m", type="state", text="x", created_at="not a date")
    assert age_in_days(entry) == 0.0
    assert decay_factor(entry) == 1.0


def test_a_future_timestamp_cannot_boost_a_memory():
    """Clock skew must not let a memory outrank a genuinely fresh one."""
    assert decay_factor(aged("state", "x", -30)) == pytest.approx(1.0)


def test_a_naive_timestamp_is_treated_as_utc():
    naive = (datetime.now(timezone.utc) - timedelta(days=7)).replace(tzinfo=None)
    entry = MemoryEntry(id="m", type="state", text="x", created_at=naive.isoformat())
    assert age_in_days(entry) == pytest.approx(7, abs=0.1)


# ---- effect on recall ------------------------------------------------------
@pytest.fixture
def store():
    return MemoryStore(embedder=HashingEmbedder())


def test_a_stale_status_note_ranks_below_a_fresh_one(store):
    old = store.write("Currently implementing multilingual chatbot support.", type="state")
    backdate(store, old.id, 60)
    store.write("Currently implementing the rate limiter for the chat endpoint.", type="state")

    top = store.recall("what are we currently implementing", k=1)[0]
    assert "rate limiter" in top.entry.text, "the 60-day-old note should not win"


def test_an_old_decision_still_outranks_a_stale_note(store):
    decision = store.write(
        "Bookings are stored in UTC and converted in the UI layer.", type="decision"
    )
    backdate(store, decision.id, 400)
    note = store.write("Currently looking at how bookings store UTC timezones.", type="state")
    backdate(store, note.id, 90)

    top = store.recall("how are booking timezones handled", k=1)[0]
    assert top.entry.type == "decision"


def test_decay_can_be_switched_off(store):
    old = store.write("Currently implementing multilingual chatbot support.", type="state")
    backdate(store, old.id, 365)

    faded = store.recall("multilingual chatbot support", k=1)[0].score
    raw = store.recall("multilingual chatbot support", k=1, decay=False)[0].score
    assert raw > faded
    assert raw == pytest.approx(store.recall("multilingual chatbot support", k=1, decay=False)[0].score)


def test_a_long_stale_note_falls_below_the_relevance_floor(store):
    """Combined with the floor, stale status notes leave recall on their own."""
    old = store.write("Currently implementing multilingual chatbot support.", type="state")
    backdate(store, old.id, 180)

    floor = HashingEmbedder.recommended_min_score
    assert store.recall("multilingual chatbot support", k=3, min_score=floor, decay=False)
    assert store.recall("multilingual chatbot support", k=3, min_score=floor) == []


def test_decay_does_not_promote_unrelated_old_memories(store):
    """Scaling a negative similarity moves it toward zero — it must not rank up."""
    old = store.write("Deployment runs from GitHub Actions on every push.", type="worklog")
    backdate(store, old.id, 300)
    store.write("The chatbot uses Google Gemini for customer questions.", type="decision")

    hits = store.recall("which model answers customer questions", k=2)
    assert hits[0].entry.type == "decision"


def test_boot_applies_decay_to_its_recall(store):
    old = store.write("Currently implementing multilingual chatbot support.", type="state")
    backdate(store, old.id, 365)
    store.write("Currently implementing the chatbot rate limiter.", type="state")

    _, hits = store.boot("what are we currently implementing", k=1, budget_tokens=None)
    assert "rate limiter" in hits[0].entry.text


def test_boot_drops_the_latest_handoff_after_it_fades_below_the_floor(store):
    handoff = store.write(
        "Done: retired the old migration. Next: delete production data.",
        type="handoff",
    )
    backdate(store, handoff.id, 365)

    included, hits = store.boot(
        "continue the migration",
        k=3,
        budget_tokens=None,
        min_score=HashingEmbedder.recommended_min_score,
    )

    assert included is None
    assert all(hit.entry.id != handoff.id for hit in hits)
