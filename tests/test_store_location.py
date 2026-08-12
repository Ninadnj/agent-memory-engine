"""Which store a command uses.

Memories used to land in one global file shared by every project, so recall
could surface another codebase's answers. The store now follows the project.
"""

import json

import pytest

from agent_memory import GLOBAL_STORE, default_store_path, find_project_root
from agent_memory.cli import build_parser, main
from agent_memory.store import relocation_notice


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A directory that looks like a git checkout, with a nested subdirectory."""
    monkeypatch.delenv("AGENT_MEMORY_PATH", raising=False)
    root = tmp_path / "my-project"
    (root / ".git").mkdir(parents=True)
    (root / "src" / "deep").mkdir(parents=True)
    return root


def test_store_follows_the_project(repo, monkeypatch):
    monkeypatch.chdir(repo)
    assert default_store_path() == repo / ".agent_memory" / "store.json"


def test_store_is_found_from_a_nested_directory(repo, monkeypatch):
    monkeypatch.chdir(repo / "src" / "deep")
    assert default_store_path() == repo / ".agent_memory" / "store.json"


def test_two_projects_get_two_stores(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_PATH", raising=False)
    first, second = tmp_path / "alpha", tmp_path / "beta"
    for project in (first, second):
        (project / ".git").mkdir(parents=True)
    monkeypatch.chdir(first)
    a = default_store_path()
    monkeypatch.chdir(second)
    b = default_store_path()
    assert a != b


def test_git_worktrees_and_submodules_are_recognised(tmp_path, monkeypatch):
    """`.git` is a file, not a directory, in a worktree or submodule."""
    monkeypatch.delenv("AGENT_MEMORY_PATH", raising=False)
    root = tmp_path / "worktree"
    root.mkdir()
    (root / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n")
    monkeypatch.chdir(root)
    assert find_project_root() == root
    assert default_store_path() == root / ".agent_memory" / "store.json"


def test_outside_a_repository_falls_back_to_the_global_store(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_PATH", raising=False)
    loose = tmp_path / "not-a-repo"
    loose.mkdir()
    monkeypatch.chdir(loose)
    assert default_store_path() == GLOBAL_STORE


def test_explicit_env_var_still_wins(repo, monkeypatch, tmp_path):
    monkeypatch.chdir(repo)
    monkeypatch.setenv("AGENT_MEMORY_PATH", str(tmp_path / "chosen.json"))
    assert default_store_path() == tmp_path / "chosen.json"


def test_memories_written_in_a_project_stay_in_that_project(repo, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_MEMORY_EMBEDDER", "hashing")
    monkeypatch.chdir(repo)
    main(["write", "Bookings are stored in UTC.", "--type", "decision"])

    store_file = repo / ".agent_memory" / "store.json"
    assert store_file.exists()
    assert "UTC" in store_file.read_text()

    other = repo.parent / "other-project"
    (other / ".git").mkdir(parents=True)
    monkeypatch.chdir(other)
    capsys.readouterr()
    main(["recall", "how are timezones handled"])
    assert "UTC" not in capsys.readouterr().out, "another project's memory leaked in"


def test_global_flag_opts_back_into_the_shared_store(repo, monkeypatch):
    monkeypatch.chdir(repo)
    args = build_parser().parse_args(["--global", "stats"])
    assert args.use_global is True


def test_relocation_notice_points_at_existing_global_memories(tmp_path, monkeypatch):
    """Upgrading must not look like every memory was deleted."""
    fake_global = tmp_path / "global.json"
    fake_global.write_text(json.dumps({"entries": [{"id": "mem_0001", "text": "x"}]}))
    monkeypatch.setattr("agent_memory.store.GLOBAL_STORE", fake_global)

    fresh = tmp_path / "proj" / ".agent_memory" / "store.json"
    notice = relocation_notice(fresh)
    assert notice and "1 memories" in notice and str(fake_global) in notice

    # Silent once the project store exists, and silent for the global store itself.
    fresh.parent.mkdir(parents=True)
    fresh.write_text(json.dumps({"entries": []}))
    assert relocation_notice(fresh) is None
    assert relocation_notice(fake_global) is None


def test_notice_is_silent_when_there_is_nothing_to_migrate(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_memory.store.GLOBAL_STORE", tmp_path / "absent.json")
    assert relocation_notice(tmp_path / "proj" / "store.json") is None
