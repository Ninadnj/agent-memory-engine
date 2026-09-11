"""Regression contracts from the September engineering review."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from agent_memory import HashingEmbedder, MemoryStore
from agent_memory.store import _file_lock


def open_store(path=None):
    return MemoryStore(path=path, embedder=HashingEmbedder())


def test_documented_prompt_event_reaches_the_hook_process(tmp_path):
    path = tmp_path / "store.json"
    open_store(path).write("Admin routes are guarded by requireAdmin in server/auth.ts.")
    env = {**os.environ, "AGENT_MEMORY_PATH": str(path), "AGENT_MEMORY_EMBEDDER": "hashing"}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src") + os.pathsep + env.get("PYTHONPATH", "")
    event = {"session_id": "test", "cwd": str(tmp_path), "hook_event_name": "UserPromptSubmit",
             "prompt": "how do we protect the admin pages?"}
    result = subprocess.run([sys.executable, "-m", "agent_memory.cli", "hook", "user-prompt"],
                            input=json.dumps(event), text=True, capture_output=True, env=env, timeout=15)
    assert result.returncode == 0
    assert "requireAdmin" in json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]


def test_startup_does_not_reintroduce_an_expired_handoff():
    store = open_store()
    entry = store.write("Next: deploy the retired staging server.", type="handoff")
    entry.created_at = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    handoff, hits = store.boot("fix invoice rounding", min_score=0.15)
    assert handoff is None
    assert all(hit.entry.id != entry.id for hit in hits)


def test_correction_refreshes_an_old_state_without_erasing_creation_time():
    store = open_store()
    entry = store.write("Currently updating the staging server.", type="state")
    entry.created_at = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    created = entry.created_at
    store.update(entry.id, text="Currently fixing invoice rounding.")
    hits = store.recall("Currently fixing invoice rounding.", min_score=0.15)
    assert hits and hits[0].entry.id == entry.id
    assert entry.created_at == created


def test_negation_is_not_a_duplicate():
    store = open_store()
    shared = "Release procedure: run database migrations, verify backups, check monitoring, notify support, and validate the rollback plan. "
    first = store.write(shared + "The deployment is approved.")
    second, stored = store.write_with_status(shared + "The deployment is not approved.")
    assert stored is True and second.id != first.id


def test_identical_text_with_a_different_type_is_preserved():
    store = open_store()
    store.write("Use UTC timestamps.", type="state")
    entry, stored = store.write_with_status("Use UTC timestamps.", type="decision")
    assert stored is True and entry.type == "decision"


def test_stale_snapshot_save_cannot_destroy_another_writers_data(tmp_path):
    path = tmp_path / "store.json"
    old = open_store(path)
    old.write("Use UTC timestamps.")
    current = open_store(path)
    current.write("Images use private object storage.")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="changed|stale"):
        old.save()
    assert path.read_bytes() == before


def test_an_active_lock_cannot_be_stolen_because_of_its_age(tmp_path):
    path = tmp_path / "store.json"
    with _file_lock(path):
        with pytest.raises(TimeoutError):
            with _file_lock(path, timeout=0.03, stale_after=0):
                pass


def test_same_dimension_different_model_reembeds(tmp_path):
    class Model:
        dim = 2

        def __init__(self, name):
            self.model_name = name
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            vector = [1., 0.] if self.model_name == "a" else [0., 1.]
            return np.array([vector] * len(texts), dtype=np.float32)

    path = tmp_path / "store.json"
    MemoryStore(path, Model("a")).write("A durable fact.")
    model = Model("b")
    store = MemoryStore(path, model)
    assert model.calls == 1
    assert store.recall("A durable fact.")[0].score == pytest.approx(1)


@pytest.mark.parametrize("k", [0, -1])
def test_nonpositive_k_never_returns_a_memory(k):
    store = open_store()
    store.write("Use UTC timestamps.")
    if k < 0:
        with pytest.raises(ValueError):
            store.recall("UTC", k=k)
    else:
        assert store.recall("UTC", k=k) == []


def test_failed_embedding_update_does_not_change_the_entry():
    store = open_store()
    entry = store.write("Use UTC timestamps.")

    def fail(texts):
        raise RuntimeError("embedding unavailable")

    store.embedder.embed = fail
    with pytest.raises(RuntimeError):
        store.update(entry.id, text="Use local timestamps.")
    assert store.all()[0].text == "Use UTC timestamps."


@pytest.mark.parametrize("text", ["", "  ", "x" * 20001], ids=["empty", "whitespace", "oversized"])
def test_invalid_memory_text_is_rejected_before_writing(tmp_path, text):
    path = tmp_path / "store.json"
    store = open_store(path)
    with pytest.raises(ValueError):
        store.write(text)
    assert not path.exists()
