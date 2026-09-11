"""Traceable corrections and failure atomicity through public store operations."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from agent_memory import (
    HashingEmbedder,
    MemoryStore,
    MemoryConflictError,
    StoreFormatError,
)
from agent_memory.rendering import recall_context
from agent_memory.tokens import count_tokens


def open_store(path=None):
    return MemoryStore(path, HashingEmbedder())


@pytest.mark.parametrize("legacy_text", ["", "A" * 20001], ids=["empty", "oversized"])
def test_legacy_records_outside_new_input_limits_remain_editable(tmp_path, legacy_text):
    path = tmp_path / "store.json"
    legacy = {"id": "mem_0001", "text": legacy_text, "type": "fact",
              "metadata": {"legacy": "x" * 17000}, "created_at": "2020-01-01T00:00:00+00:00"}
    path.write_text(json.dumps({"format": 1, "entries": [legacy]}))
    store = open_store(path)
    assert store.get("mem_0001").text == legacy_text
    store.write("An unrelated new fact.")
    reopened = open_store(path)
    corrected = reopened.update("mem_0001", text="A corrected legacy fact.", expected_revision=1)
    assert corrected.metadata == legacy["metadata"]
    assert corrected.history[0]["text"] == legacy_text
    assert open_store(path).forget("mem_0001", expected_revision=2)


def test_long_unique_explicit_identity_is_not_truncated(tmp_path):
    store = open_store(tmp_path / "store.json")
    identity = "caller-" + "x" * 600
    store.write("A caller-owned identity.", id=identity)
    assert open_store(store.path).get(identity).id == identity
    with pytest.raises(ValueError, match="duplicate"):
        store.write("Unrelated text.", id=identity)


def test_supersession_requires_an_actual_revision():
    store = open_store()
    entry = store.write("The previous decision.")
    with pytest.raises(ValueError, match="expected_revision"):
        store.supersede(entry.id, "Replacement.", expected_revision=None)
    assert len(store.all()) == 1


def test_revision_history_and_source_survive_reopening(tmp_path):
    path = tmp_path / "memory.json"
    store = open_store(path)
    first = store.write(
        "Use UTC timestamps.",
        agent="claude",
        source={"path": "settings.py", "commit": "abc1234"},
    )
    original = asdict(first)
    revised = store.update(
        first.id,
        "Display dates in Europe/Paris.",
        expected_revision=1,
        agent="codex",
        source={"path": "ui.py", "commit": "def1234"},
    )
    assert revised.id == first.id and revised.revision == 2
    assert revised.created_at == original["created_at"] and revised.agent == "claude"
    assert revised.updated_by == "codex" and revised.updated_at
    original.pop("history")
    assert revised.history == [original]
    assert asdict(open_store(path).get(first.id)) == asdict(revised)


@pytest.mark.parametrize("operation", ["update", "forget", "supersede"])
def test_stale_revision_cannot_change_another_agents_correction(tmp_path, operation):
    path = tmp_path / "shared.json"
    first = open_store(path)
    entry = first.write("Use UTC timestamps.")
    stale_revision = entry.revision
    other = open_store(path)
    other.update(
        entry.id, "Display dates in Europe/Paris.", expected_revision=1, agent="codex"
    )
    before = path.read_bytes()
    with pytest.raises(MemoryConflictError, match="current 2"):
        if operation == "forget":
            first.forget(entry.id, expected_revision=stale_revision)
        else:
            getattr(first, operation)(
                entry.id, "Use local timestamps.", expected_revision=stale_revision
            )
    assert path.read_bytes() == before


def test_noop_update_does_not_refresh_or_append_history():
    store = open_store()
    entry = store.write("Use UTC timestamps.")
    original = asdict(entry)
    assert (
        asdict(store.update(entry.id, entry.text, expected_revision=1, agent="codex"))
        == original
    )


def test_supersession_keeps_old_decision_out_of_recall_and_boot(tmp_path):
    path = tmp_path / "memory.json"
    store = open_store(path)
    first = store.write("Next: deploy the staging server.", type="handoff")
    second = store.supersede(
        first.id, "Next: retire the staging server.", expected_revision=1, agent="codex"
    )
    reopened = open_store(path)
    old = reopened.get(first.id)
    assert old.status == "superseded" and old.superseded_by == second.id
    assert old.history[0]["text"] == "Next: deploy the staging server."
    assert [h.entry.id for h in reopened.recall("staging server", k=10)] == [second.id]
    handoff, hits = reopened.boot("staging server")
    assert handoff.id == second.id
    assert all(hit.entry.id != first.id for hit in hits)
    with pytest.raises(MemoryConflictError, match="superseded"):
        reopened.update(first.id, "Bring staging back.", expected_revision=2)


@pytest.mark.parametrize("operation", ["write", "update", "forget", "supersede"])
def test_failed_disk_replace_rolls_back_data_and_cleans_tempfile(
    tmp_path, monkeypatch, operation
):
    path = tmp_path / "memory.json"
    store = open_store(path)
    entry = store.write("Use UTC timestamps.")
    before, entries, vectors = (
        path.read_bytes(),
        [asdict(e) for e in store.all()],
        store._matrix.copy(),
    )

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr("agent_memory.store.os.replace", fail)
    with pytest.raises(OSError):
        if operation == "write":
            store.write("Images are private.")
        elif operation == "forget":
            store.forget(entry.id, expected_revision=1)
        else:
            getattr(store, operation)(
                entry.id, "Use local timestamps.", expected_revision=1
            )
    assert path.read_bytes() == before
    assert [asdict(e) for e in store.all()] == entries
    np.testing.assert_array_equal(store._matrix, vectors)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "damage", ["truncated", "duplicate", "vector", "future", "status"]
)
def test_invalid_store_is_preserved_and_reported(tmp_path, damage):
    path = tmp_path / "memory.json"
    store = open_store(path)
    store.write("Use UTC timestamps.")
    original = [asdict(e) for e in store.all()]
    payload = json.loads(path.read_text())
    if damage == "duplicate":
        payload["entries"].append(payload["entries"][0])
    elif damage == "vector":
        payload["entries"][0]["embedding"] = "invalid-base64!"
    elif damage == "future":
        payload["format"] = 999
    elif damage == "status":
        payload["entries"][0]["status"] = "made-up"
    path.write_text("{" if damage == "truncated" else json.dumps(payload))
    damaged = path.read_bytes()
    with pytest.raises(StoreFormatError):
        store.write("Images are private.")
    assert path.read_bytes() == damaged
    assert [asdict(e) for e in store._entries] == original


def test_deleted_store_is_not_resurrected_by_an_open_instance(tmp_path):
    path = tmp_path / "memory.json"
    store = open_store(path)
    store.write("Use UTC timestamps.")
    path.unlink()
    fresh = store.write("Images are private.")
    assert open_store(path).all() == [fresh]


def test_snapshot_export_is_explicit_and_does_not_overwrite_by_default(tmp_path):
    path, target = tmp_path / "memory.json", tmp_path / "backup.json"
    store = open_store(path)
    entry = store.write("Use UTC timestamps.")
    store.export(target)
    assert open_store(target).all() == [entry]
    with pytest.raises(FileExistsError):
        store.export(target)
    with pytest.raises(ValueError):
        store.export(path)


@pytest.mark.parametrize("budget", [0, 7, 30, 80, 150, 300])
def test_rendered_context_obeys_the_whole_text_budget(budget):
    store = open_store()
    for text in [
        "Bookings use UTC timestamps.",
        "Bookings have a forty minute duration.",
        "Invoices are archived monthly.",
    ]:
        store.write(text, source={"path": "src/settings.py", "commit": "a" * 40})
    result = recall_context(store, "Bookings", k=5, budget=budget)
    assert count_tokens(result) <= budget
    if budget >= 150:
        assert "Bookings" in result and "src/settings.py" in result
