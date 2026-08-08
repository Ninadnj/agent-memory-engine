"""Persistence, concurrency and recovery.

The engine's pitch is one store shared by several agents, so the failure that
matters most is two agents writing at once and one of them silently winning.
These tests cover that, plus the recovery paths around switching embedders and
crash-safe writes.
"""

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from agent_memory import HashingEmbedder, MemoryStore

SRC = str((__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))


def store_at(path):
    return MemoryStore(path=path, embedder=HashingEmbedder())


def test_two_open_stores_both_keep_their_writes(tmp_path):
    path = tmp_path / "shared.json"
    seed = store_at(path)
    seed.write("Shared baseline fact about the project.", type="project")

    # Both agents opened the store before either wrote — the classic lost-update
    # setup: each holds a snapshot from before the other's write.
    claude = store_at(path)
    codex = store_at(path)
    claude.write("Claude Code: bookings are stored in UTC.", type="decision", agent="claude-code")
    codex.write("Codex: auth uses signed session cookies.", type="decision", agent="codex")

    final = store_at(path)
    assert final.stats()["count"] == 3
    agents = {e.agent for e in final.all()}
    assert {"claude-code", "codex"} <= agents


def test_parallel_processes_do_not_lose_writes(tmp_path):
    path = tmp_path / "parallel.json"
    worker = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {SRC!r})
        from agent_memory import MemoryStore, HashingEmbedder
        store = MemoryStore(path={str(path)!r}, embedder=HashingEmbedder())
        who = sys.argv[1]
        for i in range(5):
            store.write(
                f"Agent {{who}} learned distinct fact number {{i}} about subsystem {{who}}{{i}}.",
                type="fact", agent=who,
            )
        """
    )
    script = tmp_path / "worker.py"
    script.write_text(worker)

    procs = [subprocess.Popen([sys.executable, str(script), f"agent{i}"]) for i in range(6)]
    assert all(p.wait() == 0 for p in procs)

    final = store_at(path)
    ids = [e.id for e in final.all()]
    assert len(ids) == 30, f"lost {30 - len(ids)} writes"
    assert len(set(ids)) == 30, "ids must stay unique across processes"


def test_an_open_store_sees_another_agents_writes(tmp_path):
    """A long-running MCP server must not serve a snapshot from startup."""
    path = tmp_path / "live.json"
    reader = store_at(path)          # e.g. Claude Code's server, already running
    reader.write("Bookings are stored in UTC.", type="decision", agent="claude-code")

    writer = store_at(path)          # e.g. Codex, in another process
    writer.write(
        "Done: added the rate limiter. Next: re-run the integration suite.",
        type="handoff",
        agent="codex",
    )

    handoff = reader.latest("handoff")
    assert handoff is not None and handoff.agent == "codex"
    assert reader.stats()["count"] == 2
    assert any("rate limiter" in h.entry.text for h in reader.recall("rate limiting", k=3))


def test_stamp_is_taken_before_the_read_not_after(tmp_path):
    """Regression: a file replaced mid-read must not be recorded as 'current'.

    Reads are unlocked, so another agent can replace the store between our stat
    and our read. Stamping after the read pairs the *new* stamp with the *old*
    content, and every later freshness check then wrongly concludes we are up to
    date — which showed up as two agents choosing the same id and one write
    disappearing.
    """
    path = tmp_path / "toctou.json"
    writer = store_at(path)
    writer.write("First fact, present at read time.", type="fact")

    reader = MemoryStore(path=None, embedder=HashingEmbedder())
    reader.path = path

    original_read_text = type(path).read_text

    def read_then_mutate(self, *args, **kwargs):
        content = original_read_text(self, *args, **kwargs)
        if self == path:  # simulate another agent landing a write mid-read
            writer.write("Second fact, written during the read.", type="fact")
        return content

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(type(path), "read_text", read_then_mutate)
        reader.load()
    finally:
        monkeypatch.undo()

    assert reader.stats()["count"] == 2, "the store must notice it read stale content"


def test_ids_survive_explicit_ids_and_deletions(tmp_path):
    store = store_at(tmp_path / "ids.json")
    store.write("first entry", id="mem_0001")
    store.write("a second entry, wholly different", id="mem_0009")
    store.write("a third distinct entry about deployment")
    store.forget("mem_0009")
    store.write("a fourth separate entry about analytics")
    ids = [e.id for e in store.all()]
    assert len(set(ids)) == len(ids)
    assert "mem_0001" in ids


def test_switching_embedder_reembeds_instead_of_crashing(tmp_path):
    class OtherEmbedder:  # a different dimension, like sentence-transformers
        dim = 384
        recommended_min_score = 0.25

        def embed(self, texts):
            rng = np.random.default_rng(len(texts[0]))
            vecs = rng.normal(size=(len(texts), 384)).astype(np.float32)
            return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    path = tmp_path / "switch.json"
    first = MemoryStore(path=path, embedder=OtherEmbedder())
    first.write("Admin routes are guarded by requireAdmin in server/auth.ts.", type="decision")
    first.write("Bookings are stored in UTC.", type="decision")

    reopened = store_at(path)  # 512-d hashing embedder over a 384-d store
    assert reopened.stats()["count"] == 2
    hits = reopened.recall("how do we protect admin pages", k=1)
    assert hits and "Admin routes" in hits[0].entry.text


def test_save_is_atomic_and_leaves_no_partial_file(tmp_path):
    path = tmp_path / "atomic.json"
    store = store_at(path)
    for i in range(10):
        store.write(f"Fact number {i} about a distinct subsystem {i}.", type="fact")
    assert json.loads(path.read_text())["entries"]
    assert not list(tmp_path.glob("*.tmp")), "temp files must be renamed away"
    assert not list(tmp_path.glob("*.lock")), "locks must be released"


def test_legacy_float_list_stores_still_load(tmp_path):
    """v1 wrote embeddings as JSON float lists; those stores must keep working."""
    path = tmp_path / "legacy.json"
    store = store_at(path)
    store.write("Bookings are stored in UTC.", type="decision")

    payload = json.loads(path.read_text())
    vec = HashingEmbedder().embed(["Bookings are stored in UTC."])[0]
    payload.pop("format", None)
    payload["entries"][0]["embedding"] = vec.tolist()
    path.write_text(json.dumps(payload))

    reopened = store_at(path)
    assert reopened.stats()["count"] == 1
    assert reopened.recall("timezone handling for bookings", k=1)


def test_unknown_fields_in_a_newer_store_do_not_break_load(tmp_path):
    path = tmp_path / "future.json"
    store = store_at(path)
    store.write("Bookings are stored in UTC.", type="decision")

    payload = json.loads(path.read_text())
    payload["entries"][0]["confidence"] = 0.9  # written by a future version
    path.write_text(json.dumps(payload))

    assert store_at(path).stats()["count"] == 1


def test_forget_and_update_persist(tmp_path):
    path = tmp_path / "edit.json"
    store = store_at(path)
    keep = store.write("The API listens on port 5002.", type="project")
    drop = store.write("A stale note about the old deployment box.", type="state")

    assert store.update(keep.id, text="The API listens on port 8080.") is not None
    assert store.forget(drop.id) is True
    assert store.forget(drop.id) is False

    reopened = store_at(path)
    assert [e.text for e in reopened.all()] == ["The API listens on port 8080."]
    hits = reopened.recall("which port does the API listen on", k=1)
    assert "8080" in hits[0].entry.text


def test_update_reembeds_so_recall_follows_the_new_text(tmp_path):
    store = store_at(tmp_path / "reembed.json")
    entry = store.write("The chatbot uses Google Gemini.", type="decision")
    store.update(entry.id, text="The chatbot uses Anthropic Claude via the Messages API.")
    hits = store.recall("which model does the chatbot use", k=1)
    assert "Claude" in hits[0].entry.text


def test_update_rejects_unknown_type(tmp_path):
    store = store_at(tmp_path / "badtype.json")
    entry = store.write("something", type="fact")
    with pytest.raises(ValueError):
        store.update(entry.id, type="not_a_type")


def test_paths_with_a_tilde_are_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = MemoryStore(path="~/nested/store.json", embedder=HashingEmbedder())
    store.write("Bookings are stored in UTC.", type="decision")
    assert (tmp_path / "nested" / "store.json").exists()
