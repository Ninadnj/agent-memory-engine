"""Windows contracts also exercised with portable, deterministic fixtures."""

import json

import pytest

from agent_memory import HashingEmbedder, MemoryStore
from agent_memory import hooks


@pytest.mark.parametrize(
    "executable",
    [
        r"C:\Python\Scripts\agent-memory.EXE",
        r"C:\Program Files\Python\Scripts\agent-memory.exe",
        "/a path/venv/bin/agent-memory",
    ],
)
def test_hook_management_recognizes_quoted_and_windows_executables(
    tmp_path, monkeypatch, executable
):
    monkeypatch.setattr(hooks, "_executable", lambda: executable)
    path = tmp_path / "settings.json"
    unrelated = {"type": "command", "command": "echo 'agent-memory hook session-start'"}
    path.write_text(json.dumps({"hooks": {"SessionStart": [{"hooks": [unrelated]}]}}))
    hooks.install(path, ["SessionStart", "SessionEnd"])
    hooks.install(path, ["SessionStart", "SessionEnd"])
    data = json.loads(path.read_text())
    commands = [
        h
        for groups in data["hooks"].values()
        for group in groups
        for h in group["hooks"]
    ]
    assert len(commands) == 3 and unrelated in commands
    assert sorted(hooks.installed_events(path)) == ["SessionEnd", "SessionStart"]
    hooks.uninstall(path)
    assert json.loads(path.read_text()) == {
        "hooks": {"SessionStart": [{"hooks": [unrelated]}]}
    }


@pytest.mark.parametrize("code", [5, 32])
def test_transient_windows_reader_does_not_lose_a_write(tmp_path, monkeypatch, code):
    import os

    path = tmp_path / "store.json"
    store = MemoryStore(path, embedder=HashingEmbedder())
    old = store.write("An existing fact.")
    replace = os.replace
    blocked = True

    def reader_is_closing(source, target):
        nonlocal blocked
        if blocked:
            blocked = False
            error = PermissionError("temporary Windows reader sharing conflict")
            error.winerror = code
            raise error
        return replace(source, target)

    monkeypatch.setattr(os, "replace", reader_is_closing)
    new = store.write("A second distinct fact.")
    reopened = MemoryStore(path, embedder=HashingEmbedder())
    assert {entry.id for entry in reopened.all()} == {old.id, new.id}


def test_permanent_windows_replace_failure_is_bounded_and_keeps_original(
    tmp_path, monkeypatch
):
    import os
    import time
    from agent_memory._locking import _replace_file

    source, target = tmp_path / "new.json", tmp_path / "store.json"
    source.write_text("new")
    target.write_text("original")

    def denied(*args):
        error = PermissionError("persistent sharing failure")
        error.winerror = 32
        raise error

    monkeypatch.setattr(os, "replace", denied)
    start = time.monotonic()
    with pytest.raises(PermissionError):
        _replace_file(source, target, timeout=0.05)
    assert time.monotonic() - start < 1
    assert target.read_text() == "original" and source.read_text() == "new"
