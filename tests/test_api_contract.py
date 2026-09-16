"""Public API contracts: snapshots cannot bypass tracked writes."""

import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from agent_memory import HashingEmbedder, MemoryConflictError, MemoryStore, embeddings
from agent_memory.rendering import boot_context


@pytest.mark.parametrize(
    "operation",
    [
        "write",
        "write_with_status",
        "duplicate",
        "get",
        "all",
        "latest",
        "recall",
        "boot_handoff",
        "boot_hit",
        "update",
        "noop",
        "supersede",
    ],
)
def test_returned_memories_are_detached_snapshots(tmp_path, operation):
    path = tmp_path / "memory.json"
    store = MemoryStore(path, HashingEmbedder())
    entry = store.write(
        "Bookings use UTC.",
        type="handoff" if operation == "boot_handoff" else "fact",
        metadata={"labels": ["calendar"]},
        source={"path": "calendar.py", "evidence": ["review"]},
    )
    if operation == "write_with_status":
        entry, stored = store.write_with_status("Bookings use Paris time.")
        assert stored
    elif operation == "duplicate":
        entry, stored = store.write_with_status(
            entry.text, source=entry.source, metadata=entry.metadata
        )
        assert not stored
    elif operation == "get":
        entry = store.get(entry.id)
    elif operation == "all":
        entry = store.all()[0]
    elif operation == "latest":
        entry = store.latest("fact")
    elif operation == "recall":
        entry = store.recall("Bookings UTC")[0].entry
    elif operation == "boot_handoff":
        entry = store.boot("Bookings UTC")[0]
    elif operation == "boot_hit":
        entry = store.boot("Bookings UTC")[1][0].entry
    elif operation in ("update", "noop"):
        entry = store.update(
            entry.id,
            "Bookings use Paris time." if operation == "update" else entry.text,
            expected_revision=1,
        )
    elif operation == "supersede":
        entry = store.supersede(
            entry.id, "Bookings use Paris time.", expected_revision=1
        )

    original = deepcopy(asdict(entry))
    entry.text = "Untracked replacement."
    entry.revision = 999
    entry.metadata.setdefault("labels", []).append("tampered")
    entry.source.setdefault("evidence", []).append("tampered")
    if entry.history:
        entry.history[0]["text"] = "Tampered history."
    entry.history.append({"injected": True})

    assert asdict(store.get(original["id"])) == original
    # A later, legitimate write must not accidentally commit the local edits.
    store.write("Invoices are archived monthly.")
    reopened = MemoryStore(path, HashingEmbedder())
    assert asdict(reopened.get(original["id"])) == original
    hit = reopened.recall(original["text"], k=1)[0]
    assert hit.entry.id == original["id"] and hit.score > 0.99


def test_saved_snapshot_keeps_the_revision_that_was_actually_read():
    store = MemoryStore(embedder=HashingEmbedder())
    earlier = store.write("Retry three times.")
    store.update(earlier.id, "Retry five times.", expected_revision=earlier.revision)
    assert earlier.revision == 1 and earlier.text == "Retry three times."
    with pytest.raises(MemoryConflictError, match="current 2"):
        store.forget(earlier.id, expected_revision=earlier.revision)


def test_duplicate_behavior_has_an_explicit_boolean_option():
    store = MemoryStore(embedder=HashingEmbedder())
    first = store.write("Bookings use UTC.")
    duplicate, stored = store.write_with_status(first.text)
    assert duplicate.id == first.id and not stored
    repeated, stored = store.write_with_status(first.text, deduplicate=False)
    assert stored and repeated.id != first.id
    assert store.write(first.text, deduplicate=False).id != first.id
    with pytest.raises(ValueError, match="deduplicate"):
        store.write(first.text, deduplicate=0.97)


@pytest.mark.parametrize("choice", [None, "auto", "hashing", " HASHING "])
def test_default_backend_does_not_attempt_a_model_load(monkeypatch, choice):
    if choice is None:
        monkeypatch.delenv("AGENT_MEMORY_EMBEDDER", raising=False)
    else:
        monkeypatch.setenv("AGENT_MEMORY_EMBEDDER", choice)

    def unexpected_model_load(*args, **kwargs):
        pytest.fail("offline defaults attempted to load a model")

    monkeypatch.setattr(
        embeddings, "SentenceTransformerEmbedder", unexpected_model_load
    )
    assert isinstance(embeddings.default_embedder(), HashingEmbedder)


def test_unknown_backend_is_reported_instead_of_silently_falling_back(monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_EMBEDDER", "typo")
    with pytest.raises(ValueError, match="AGENT_MEMORY_EMBEDDER"):
        embeddings.default_embedder()


def test_explicit_semantic_backend_reports_load_failure(monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_EMBEDDER", "sentence-transformers")

    def unavailable(*args, **kwargs):
        raise ImportError("model dependency unavailable")

    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", unavailable)
    with pytest.raises(RuntimeError, match="model dependency unavailable"):
        embeddings.default_embedder()


def test_reopening_keeps_existing_embedding_configuration(tmp_path, monkeypatch):
    path = tmp_path / "memory.json"
    MemoryStore(path, HashingEmbedder(dim=128)).write("Bookings use UTC.")
    monkeypatch.setenv("AGENT_MEMORY_EMBEDDER", " AUTO ")
    reopened = MemoryStore(path)
    assert reopened.embedder.dim == 128
    assert json.loads(path.read_text())["embedding_config"]["dim"] == 128


@pytest.mark.parametrize("read_method", ["get", "recall"])
def test_concurrent_reader_cannot_observe_a_write_that_rolls_back(
    tmp_path, monkeypatch, read_method
):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Event

    store = MemoryStore(tmp_path / "memory.json", HashingEmbedder())
    entry = store.write("Bookings use UTC.")
    saving, release = Event(), Event()

    def fail_save(*args):
        saving.set()
        assert release.wait(timeout=5), "test did not release the pending write"
        raise OSError("disk unavailable")

    def read():
        if read_method == "get":
            return store.get(entry.id).text
        return store.recall("Bookings", k=1)[0].entry.text

    monkeypatch.setattr(store, "_save_unlocked", fail_save)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(store.update, entry.id, "Uncommitted booking policy.")
        try:
            assert saving.wait(timeout=5)
            reader = pool.submit(read)
            with pytest.raises(TimeoutError):
                reader.result(timeout=0.05)
        finally:
            release.set()
        with pytest.raises(OSError, match="disk unavailable"):
            writer.result(timeout=5)
        assert reader.result(timeout=5) == "Bookings use UTC."


@pytest.mark.parametrize("rendered", [False, True], ids=["library", "rendered"])
def test_boot_uses_one_snapshot_when_another_store_replaces_the_handoff(
    tmp_path, monkeypatch, rendered
):
    path = tmp_path / "memory.json"
    writer = MemoryStore(path, HashingEmbedder())
    old = writer.write("Deployment is approved. Next: deploy now.", type="handoff")
    fact = writer.write("Deployment requires a verified backup.")
    reader = MemoryStore(path, HashingEmbedder())
    replacement_text = "Deployment is NOT approved. Next: wait for review."
    latest = reader.latest

    def latest_then_replace(*args, **kwargs):
        handoff = latest(*args, **kwargs)
        # Deterministically interleave a separate store's committed write between
        # handoff selection and recall; no scheduling sleeps are necessary.
        writer.supersede(old.id, replacement_text, expected_revision=1)
        return handoff

    def context():
        if rendered:
            return boot_context(reader, "Deployment", budget=1000, min_score=-1)
        handoff, hits = reader.boot("Deployment", budget_tokens=1000, min_score=-1)
        return "\n".join([handoff.text, *(hit.entry.text for hit in hits)])

    with monkeypatch.context() as race:
        race.setattr(reader, "latest", latest_then_replace)
        before = context()

    assert writer.get(old.id).status == "superseded"
    assert old.text in before and fact.text in before
    assert replacement_text not in before
    # Consistency must not become permanent staleness: the next call refreshes.
    after = context()
    assert replacement_text in after and fact.text in after
    assert old.text not in after
