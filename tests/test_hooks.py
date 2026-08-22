"""Session hooks — the half of memory that does not depend on the model.

Two properties matter most here and are tested hardest:

* a hook must never break the user's session, whatever it is handed;
* installing must never damage hooks belonging to other tools.
"""

import io
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_memory import hooks


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_EMBEDDER", "hashing")
    monkeypatch.setenv("AGENT_MEMORY_AGENT", "claude-code")


@pytest.fixture
def repo(tmp_path):
    """A real git repository with one commit."""
    root = tmp_path / "project"
    root.mkdir()
    run = lambda *a: subprocess.run(a, cwd=root, capture_output=True, check=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "Test")
    (root / "app.py").write_text("v1\n")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "initial commit")
    return root


def payload(root, event, **extra):
    return {"session_id": "sess-1", "cwd": str(root), "hook_event_name": event, **extra}


def store_for(root):
    from agent_memory import MemoryStore

    return MemoryStore(path=root / ".agent_memory" / "store.json")


# ---- SessionStart ----------------------------------------------------------
def test_session_start_injects_handoff_and_orientation(repo):
    store = store_for(repo)
    store.write("Bookings are stored in UTC.", type="decision", agent="claude-code")
    store.write("Done: fixed emails. Next: add rate limiting.", type="handoff", agent="codex")

    out = hooks.session_start(payload(repo, "SessionStart"))
    context = out["additionalContext"]
    assert "Last handoff [codex]" in context
    assert "rate limiting" in context
    assert "Bookings are stored in UTC." in context
    assert out["hookSpecificOutput"]["hookEventName"] == "SessionStart"


def test_session_start_surfaces_the_previous_session_note(repo):
    """The SessionEnd autosave is pointless if the next session never reads it."""
    hooks.session_start(payload(repo, "SessionStart"))
    (repo / "app.py").write_text("v2\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "Add rate limiting"], cwd=repo, check=True, capture_output=True)
    hooks.session_end(payload(repo, "SessionEnd"))

    context = hooks.session_start(payload(repo, "SessionStart", session_id="next"))[
        "additionalContext"
    ]
    assert "Last session:" in context
    assert "Add rate limiting" in context


def test_session_start_does_not_inject_expired_handoffs_or_worklogs(repo):
    store = store_for(repo)
    handoff = store.write(
        "Done: old migration. Next: delete production data.", type="handoff"
    )
    worklog = store.write("An obsolete session note.", type="worklog")
    old = datetime.now(timezone.utc) - timedelta(days=365)
    handoff.created_at = old.isoformat(timespec="seconds")
    worklog.created_at = old.isoformat(timespec="seconds")
    store.save()
    store.write("Bookings are stored in UTC.", type="project")

    context = hooks.session_start(payload(repo, "SessionStart"))["additionalContext"]

    assert "Bookings are stored in UTC." in context
    assert "delete production data" not in context
    assert "obsolete session note" not in context


def test_session_start_is_silent_on_an_empty_store(repo):
    assert hooks.session_start(payload(repo, "SessionStart")) == {}


def test_session_start_respects_a_token_budget(repo, monkeypatch):
    monkeypatch.setattr(hooks, "SESSION_START_BUDGET", 40)
    store = store_for(repo)
    for i in range(12):
        store.write(f"Durable project fact number {i} about subsystem {i}.", type="project")

    from agent_memory.tokens import count_tokens

    context = hooks.session_start(payload(repo, "SessionStart"))["additionalContext"]
    memories = [ln for ln in context.splitlines() if ln.startswith("- [")]
    assert memories, "should still inject something"
    assert sum(count_tokens(ln) for ln in memories) <= 40 + 10  # + list markers


def test_session_start_records_a_marker(repo):
    hooks.session_start(payload(repo, "SessionStart"))
    marker = repo / ".agent_memory" / "sessions" / "sess-1.json"
    assert marker.exists()
    assert json.loads(marker.read_text())["branch"]


# ---- SessionEnd ------------------------------------------------------------
def test_session_end_saves_what_actually_changed(repo):
    hooks.session_start(payload(repo, "SessionStart"))

    (repo / "app.py").write_text("v2\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "Add rate limiting"], cwd=repo, check=True, capture_output=True)
    (repo / "notes.md").write_text("wip\n")

    out = hooks.session_end(payload(repo, "SessionEnd", end_reason="clear"))
    assert "systemMessage" in out

    saved = [e for e in store_for(repo).all() if e.type == "worklog"]
    assert len(saved) == 1
    text = saved[0].text
    assert "Add rate limiting" in text          # the commit made this session
    assert "notes.md" in text                   # still-uncommitted work
    assert saved[0].agent == "claude-code"


def test_session_end_writes_nothing_when_nothing_happened(repo):
    """The store lives inside the repo, so its own files must not read as work."""
    hooks.session_start(payload(repo, "SessionStart"))
    store_for(repo).write("A fact written during the session.", type="decision")

    assert hooks.session_end(payload(repo, "SessionEnd")) == {}
    assert [e.type for e in store_for(repo).all()] == ["decision"]  # no worklog note


def test_session_end_ignores_dirty_files_that_predate_the_session(repo):
    (repo / "notes.md").write_text("already dirty\n")
    hooks.session_start(payload(repo, "SessionStart"))

    assert hooks.session_end(payload(repo, "SessionEnd")) == {}
    assert [e for e in store_for(repo).all() if e.type == "worklog"] == []


def test_session_end_detects_further_edits_to_a_preexisting_dirty_file(repo):
    notes = repo / "notes.md"
    notes.write_text("already dirty\n")
    hooks.session_start(payload(repo, "SessionStart"))
    notes.write_text("already dirty, then changed again during this session\n")

    assert "systemMessage" in hooks.session_end(payload(repo, "SessionEnd"))
    worklogs = [e for e in store_for(repo).all() if e.type == "worklog"]
    assert len(worklogs) == 1
    assert "notes.md" in worklogs[0].text


def test_session_end_removes_the_marker(repo):
    hooks.session_start(payload(repo, "SessionStart"))
    marker = repo / ".agent_memory" / "sessions" / "sess-1.json"
    assert marker.exists()
    hooks.session_end(payload(repo, "SessionEnd"))
    assert not marker.exists()


@pytest.mark.parametrize(
    "path, is_ours",
    [
        (".agent_memory/", True),                 # git reports untracked dirs this way
        (".agent_memory/store.json", True),
        ("./.agent_memory/sessions/a.json", True),
        ('".agent_memory/odd name.json"', True),  # quoted when it contains a space
        ("agent_memory/real_code.py", False),     # must not over-match
        ("src/app.py", False),
        (".agentic/notes.md", False),
    ],
)
def test_only_our_own_bookkeeping_is_filtered_out(path, is_ours):
    assert hooks._is_store_path(path) is is_ours


def test_session_end_outside_a_repository_is_harmless(tmp_path):
    loose = tmp_path / "not-a-repo"
    loose.mkdir()
    assert hooks.session_end(payload(loose, "SessionEnd")) == {}


# ---- UserPromptSubmit ------------------------------------------------------
def test_user_prompt_injects_relevant_memories(repo):
    store_for(repo).write(
        "Admin routes are guarded by requireAdmin in server/auth.ts.", type="decision"
    )
    out = hooks.user_prompt(
        payload(repo, "UserPromptSubmit", user_input="how do we protect the admin pages?")
    )
    assert "requireAdmin" in out["additionalContext"]


def test_user_prompt_ignores_short_prompts(repo):
    store_for(repo).write("Bookings are stored in UTC.", type="decision")
    assert hooks.user_prompt(payload(repo, "UserPromptSubmit", user_input="yes")) == {}


def test_user_prompt_is_silent_when_nothing_is_relevant(repo):
    store_for(repo).write("Bookings are stored in UTC.", type="decision")
    out = hooks.user_prompt(
        payload(repo, "UserPromptSubmit", user_input="what is the best sourdough bread recipe")
    )
    assert out == {}


# ---- robustness: a hook must never break a session -------------------------
@pytest.mark.parametrize(
    "raw", ["", "   ", "not json at all", "[1,2,3]", '{"cwd": null}', '{"cwd": "/nonexistent/xyz"}']
)
@pytest.mark.parametrize("event", ["SessionStart", "SessionEnd", "UserPromptSubmit"])
def test_run_survives_any_input(raw, event):
    out = io.StringIO()
    assert hooks.run(event, stdin=io.StringIO(raw), stdout=out) == 0
    printed = out.getvalue().strip()
    if printed:
        json.loads(printed)  # whatever we emit must be valid JSON


def test_run_survives_a_handler_that_raises(monkeypatch, capsys):
    monkeypatch.setitem(
        hooks.EVENT_HANDLERS, "SessionStart", lambda p: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    out = io.StringIO()
    assert hooks.run("SessionStart", stdin=io.StringIO("{}"), stdout=out) == 0
    assert out.getvalue() == ""                       # nothing on the protocol channel
    assert "boom" in capsys.readouterr().err          # reported on stderr instead


def test_run_emits_only_json_on_stdout(repo):
    store_for(repo).write("Bookings are stored in UTC.", type="decision")
    out = io.StringIO()
    hooks.run("SessionStart", stdin=io.StringIO(json.dumps(payload(repo, "SessionStart"))), stdout=out)
    json.loads(out.getvalue())  # raises if anything else leaked in


# ---- installing into settings.json ----------------------------------------
@pytest.fixture
def settings(tmp_path):
    return tmp_path / ".claude" / "settings.json"


def test_install_creates_settings_and_is_idempotent(settings):
    hooks.install(settings, ["SessionStart", "SessionEnd"])
    hooks.install(settings, ["SessionStart", "SessionEnd"])

    data = json.loads(settings.read_text())
    entries = [
        h
        for groups in data["hooks"].values()
        for g in groups
        for h in g["hooks"]
    ]
    assert len(entries) == 2, f"re-running must not stack copies: {entries}"
    assert sorted(entry["args"][-1] for entry in entries) == [
        "session-end",
        "session-start",
    ]


def test_install_preserves_other_tools_and_settings(settings):
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({
        "permissions": {"allow": ["Bash(npm test)"]},
        "hooks": {
            "SessionStart": [{"hooks": [{"type": "command", "command": "other-tool init"}]}],
            "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "my-linter"}]}],
        },
    }))

    hooks.install(settings, ["SessionStart", "SessionEnd"])
    data = json.loads(settings.read_text())

    start = [h for g in data["hooks"]["SessionStart"] for h in g["hooks"]]
    assert any(h["command"] == "other-tool init" for h in start)
    assert any(h.get("args") == ["hook", "session-start"] for h in start)
    assert data["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "my-linter"
    assert data["permissions"] == {"allow": ["Bash(npm test)"]}


def test_uninstall_removes_only_our_hooks(settings):
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({
        "hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "other-tool init"}]}]}
    }))
    hooks.install(settings, ["SessionStart", "SessionEnd"])
    hooks.uninstall(settings)

    data = json.loads(settings.read_text())
    remaining = [h["command"] for g in data["hooks"]["SessionStart"] for h in g["hooks"]]
    assert remaining == ["other-tool init"]
    assert "SessionEnd" not in data["hooks"]


def test_uninstall_on_a_clean_file_leaves_no_empty_scaffolding(settings):
    hooks.uninstall(settings)
    assert json.loads(settings.read_text()) == {}


def test_prompt_recall_hook_carries_a_timeout_under_the_event_limit(settings):
    hooks.install(settings, ["UserPromptSubmit"])
    data = json.loads(settings.read_text())
    entry = data["hooks"]["UserPromptSubmit"][0]["hooks"][0]
    assert entry["timeout"] < 30, "UserPromptSubmit is capped at 30s by the client"


def test_installed_command_is_an_absolute_path(settings):
    """The client spawns hooks without our virtualenv necessarily on PATH."""
    hooks.install(settings, ["SessionStart"])
    entry = json.loads(settings.read_text())["hooks"]["SessionStart"][0]["hooks"][0]
    assert entry["args"] == ["hook", "session-start"]
    assert entry["command"] == "agent-memory" or Path(entry["command"]).is_absolute()


def test_installed_command_preserves_an_executable_path_with_spaces(settings, monkeypatch):
    executable = "/tmp/Agent Memory/bin/agent-memory"
    monkeypatch.setattr(hooks, "_executable", lambda: executable)

    hooks.install(settings, ["SessionStart"])

    entry = json.loads(settings.read_text())["hooks"]["SessionStart"][0]["hooks"][0]
    assert entry["command"] == executable
    assert entry["args"] == ["hook", "session-start"]


def test_hooks_written_bare_are_still_recognised(settings):
    """An entry installed by an older version must still be found and replaced."""
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"hooks": {"SessionStart": [
        {"hooks": [{"type": "command", "command": "agent-memory hook session-start"}]}
    ]}}))
    hooks.install(settings, ["SessionStart"])

    commands = [
        h["command"]
        for g in json.loads(settings.read_text())["hooks"]["SessionStart"]
        for h in g["hooks"]
    ]
    assert len(commands) == 1, f"the old entry should be replaced, not kept: {commands}"


def test_installed_events_reports_what_is_wired(settings):
    assert hooks.installed_events(settings) == []
    hooks.install(settings, ["SessionStart", "SessionEnd"])
    assert sorted(hooks.installed_events(settings)) == ["SessionEnd", "SessionStart"]


def test_malformed_settings_file_is_reported_not_overwritten(settings):
    settings.parent.mkdir(parents=True)
    settings.write_text("{ this is not json")
    with pytest.raises(SystemExit):
        hooks.install(settings, ["SessionStart"])
    assert settings.read_text() == "{ this is not json"  # left alone
