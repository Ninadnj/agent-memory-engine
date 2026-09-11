"""Real Git state and client payload contracts, separate from handler internals."""

import json
from datetime import datetime, timedelta, timezone

from agent_memory import hooks
from test_hooks import repo, payload, store_for, offline  # noqa: F401
from agent_memory.tokens import count_tokens


def test_documented_field_takes_precedence_over_legacy_alias(repo):
    store_for(repo).write("Admin routes use requireAdmin.")
    result = hooks.user_prompt(payload(repo, "UserPromptSubmit", prompt="how do admin routes use requireAdmin?",
                                       user_input="how to bake sourdough bread?"))
    assert "requireAdmin" in result["additionalContext"]


def test_startup_drops_old_handoff_and_worklog(repo):
    store = store_for(repo)
    for kind in ("handoff", "worklog"):
        entry = store.write("Retired staging server deployment.", type=kind)
        entry.created_at = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    store.save()
    assert hooks.session_start(payload(repo, "SessionStart")) == {}


def test_preexisting_dirty_file_does_not_become_this_sessions_work(repo):
    (repo / "app.py").write_text("preexisting changes\n")
    hooks.session_start(payload(repo, "SessionStart"))
    assert hooks.session_end(payload(repo, "SessionEnd")) == {}
    assert not store_for(repo).all()


def test_unstaged_filename_and_spaces_survive_git_parsing(repo):
    hooks.session_start(payload(repo, "SessionStart"))
    (repo / "app.py").write_text("new changes\n")
    (repo / "notes with spaces.md").write_text("new note\n")
    hooks.session_end(payload(repo, "SessionEnd"))
    note = store_for(repo).latest("worklog")
    assert "app.py" in note.text and "notes with spaces.md" in note.text
    assert note.source["event"] == "SessionEnd"


def test_session_id_cannot_escape_the_marker_directory(repo):
    hooks.session_start(payload(repo, "SessionStart", session_id="../../outside"))
    assert not (repo / "outside.json").exists()
    assert len(list((repo / ".agent_memory" / "sessions").glob("*.json"))) == 1


def test_startup_never_splits_an_atomic_multiline_memory(repo, monkeypatch):
    monkeypatch.setattr(hooks, "SESSION_START_BUDGET", 30)
    store_for(repo).write("Warning: only deploy when all checks pass.\n" + "extra conditions " * 80, type="handoff")
    store_for(repo).write("Bookings use UTC.", type="decision")
    result = hooks.session_start(payload(repo, "SessionStart"))["additionalContext"]
    assert "Warning:" not in result and "Bookings use UTC." in result
    assert count_tokens(result) <= 30
