"""Fixtures that create historical memories through the actual write path."""

import subprocess
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_EMBEDDER", "hashing")
    monkeypatch.setenv("AGENT_MEMORY_AGENT", "claude-code")


@pytest.fixture
def repo(tmp_path):
    """A real Git repository with one commit, shared by the hook tests."""
    root = tmp_path / "project"
    root.mkdir()

    def run(*args):
        return subprocess.run(args, cwd=root, capture_output=True, check=True)

    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "Test")
    (root / "app.py").write_text("v1\n")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "initial commit")
    return root


@pytest.fixture
def write_aged(monkeypatch):
    def write(store, text, *, days, **kwargs):
        timestamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with monkeypatch.context() as clock:
            clock.setattr("agent_memory.store._now_iso", lambda: timestamp)
            return store.write(text, **kwargs)

    return write
