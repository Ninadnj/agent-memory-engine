import json
import subprocess
import sys

import pytest

from agent_memory import HashingEmbedder, MemoryStore
from agent_memory.cli import main
from agent_memory.diagnostics import explain_recall, inspect_memory
from agent_memory.store import _file_lock


def test_source_check_detects_changed_file(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    source = tmp_path / "policy.py"
    source.write_text("RETRIES = 3\n")
    git("add", "policy.py")
    git("commit", "-qm", "policy")
    store = MemoryStore(tmp_path / ".agent_memory" / "store.json", embedder=HashingEmbedder())
    entry = store.write("Retry at most three times.", source={"path": "policy.py", "commit": git("rev-parse", "HEAD")})
    assert "unchanged" in inspect_memory(store, entry.id)["source_check"]
    source.write_text("RETRIES = 5\n")
    assert inspect_memory(store, entry.id)["source_check"] == "source changed; review this memory"


def test_cli_revision_conflict_and_doctor(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_EMBEDDER", "hashing")
    path = tmp_path / "store.json"
    store = MemoryStore(path)
    entry = store.write("Retry three times.")
    base = ["--path", str(path)]
    with pytest.raises(SystemExit) as missing:
        main(base + ["update", entry.id, "Retry five times."])
    assert missing.value.code == 2
    assert store.get(entry.id).revision == 1
    main(base + ["update", entry.id, "Retry five times.", "--expected-revision", "1"])
    before = path.read_bytes()
    with pytest.raises(SystemExit) as conflict:
        main(base + ["forget", entry.id, "--expected-revision", "1"])
    assert conflict.value.code == 2
    assert path.read_bytes() == before
    capsys.readouterr()
    main(base + ["doctor", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["active"] == 1
    path.write_text("{broken")
    with pytest.raises(SystemExit) as invalid:
        main(base + ["doctor", "--json"])
    assert invalid.value.code == 1
    assert not json.loads(capsys.readouterr().out)["ok"]
    assert path.read_text() == "{broken"


def test_explain_reports_budget_and_supersession():
    store = MemoryStore(embedder=HashingEmbedder())
    old = store.write("Retry three times.")
    store.supersede(old.id, "Retry five times.", expected_revision=1)
    report = explain_recall(store, "retry", budget=0)
    reasons = {item["reason"] for item in report["candidates"]}
    assert reasons == {"superseded", "rendered text exceeds remaining budget"}
    assert report["rendered_tokens"] == 0


def test_process_exit_releases_lock(tmp_path):
    # The child exits without context-manager cleanup. No age heuristic or
    # manual lock-file removal should be needed before the next writer.
    code = """
import os, sys
from pathlib import Path
from agent_memory.store import _file_lock
with _file_lock(Path(sys.argv[1])):
    os._exit(0)
"""
    import os
    from pathlib import Path
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src") + os.pathsep + env.get("PYTHONPATH", "")
    path = tmp_path / "store.json"
    subprocess.run([sys.executable, "-c", code, str(path)], env=env, check=True, timeout=20)
    with _file_lock(path, timeout=0.1):
        pass


@pytest.mark.parametrize("field", ["source", "metadata"])
@pytest.mark.parametrize("value", [[], "", False, {"bad": float("nan")}])
def test_invalid_mapping_rejected_without_write(tmp_path, field, value):
    path = tmp_path / "store.json"
    store = MemoryStore(path, embedder=HashingEmbedder())
    with pytest.raises(ValueError):
        store.write("Some fact.", **{field: value})
    assert not path.exists()


def test_existing_embedding_configuration_is_sticky(tmp_path, monkeypatch):
    path = tmp_path / "store.json"
    store = MemoryStore(path, embedder=HashingEmbedder(dim=128))
    entry = store.write("Booking dates are UTC.")
    monkeypatch.delenv("AGENT_MEMORY_EMBEDDER", raising=False)
    reopened = MemoryStore(path)
    assert reopened.embedder.dim == 128
    assert reopened.recall("Booking dates")[0].entry.id == entry.id


def test_long_markdown_paragraphs_are_bounded_without_losing_content():
    import importlib.util
    from pathlib import Path
    from agent_memory import count_tokens
    spec = importlib.util.spec_from_file_location("ingest", Path(__file__).resolve().parents[1] / "scripts" / "ingest_markdown.py")
    ingest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingest)
    text = "Booking café საქართველო " * 100
    chunks = ingest.sections(text, max_tokens=40)
    assert len(chunks) > 1
    assert all(0 < count_tokens(chunk) <= 40 for chunk in chunks)
    assert "".join("".join(chunks).split()) == "".join(text.split())
