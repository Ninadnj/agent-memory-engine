"""Offline adapter contracts, not live-agent performance measurements."""

import io
import json
import subprocess
import sys
from pathlib import Path

import codex_adapter
import pytest
import run_tasks
from task_cases import TASKS


def events(*rows):
    return io.BytesIO(b"\n".join(json.dumps(row).encode() for row in rows))


def test_usage_comes_from_completed_turn_not_messages_or_tool_outputs():
    usage = {
        "input_tokens": 1200,
        "output_tokens": 180,
        "cached_input_tokens": 900,
        "reasoning_output_tokens": 50,
    }
    result = codex_adapter.parse_events(
        events(
            {"type": "thread.started", "thread_id": "fresh-thread"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"type": "command_execution"}},
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "0 tokens"},
            },
            {"type": "turn.completed", "usage": usage},
        )
    )
    assert result == usage
    # Cached/reasoning components must not be added again to the totals.
    assert result["input_tokens"] + result["output_tokens"] == 1380


@pytest.mark.parametrize("usage", [None, "absent"])
def test_missing_usage_remains_unknown(usage):
    completed = {"type": "turn.completed"}
    if usage is None:
        completed["usage"] = None
    assert codex_adapter.parse_events(events(completed)) is None


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"type": "turn.started"}],
        [{"type": "turn.failed", "error": {"message": "out of quota"}}],
        [{"type": "turn.completed"}, {"type": "error", "message": "failed"}],
        [{"type": "turn.completed"}, {"type": "turn.completed"}],
        [[], {"type": "turn.completed"}],
        [
            {
                "type": "turn.completed",
                "usage": {"input_tokens": True, "output_tokens": 2},
            }
        ],
        [{"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": -1}}],
        [{"type": "turn.completed", "usage": {"input_tokens": 1}}],
        [{"type": "turn.completed", "usage": "unknown"}],
        [
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "cached_input_tokens": -1,
                },
            }
        ],
    ],
)
def test_failed_incomplete_or_ambiguous_sessions_are_rejected(rows):
    with pytest.raises(ValueError):
        codex_adapter.parse_events(events(*rows))


@pytest.mark.parametrize("raw", [b"not json\n", b"\xff\n"])
def test_invalid_event_encoding_is_rejected(raw):
    with pytest.raises(ValueError, match="invalid JSON"):
        codex_adapter.parse_events(io.BytesIO(raw))


def test_event_size_and_total_output_limits(monkeypatch):
    monkeypatch.setattr(codex_adapter, "MAX_EVENT_BYTES", 100)
    monkeypatch.setattr(codex_adapter, "MAX_OUTPUT_BYTES", 120)
    with pytest.raises(ValueError, match="limit"):
        codex_adapter.parse_events(io.BytesIO(b"x" * 101))
    with pytest.raises(ValueError, match="limit"):
        codex_adapter.parse_events(events(*[{"type": "turn.started"}] * 10))


@pytest.mark.parametrize("failure", ["missing", "old_cli", "not_logged_in"])
def test_preflight_stops_before_any_task_or_model_call(monkeypatch, failure):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command[1:])
        if failure == "missing":
            raise FileNotFoundError()
        output = "codex-cli test-double"
        status = 0
        if command[1:] == ["exec", "--help"]:
            output = (
                "" if failure == "old_cli" else " ".join(codex_adapter.REQUIRED_FLAGS)
            )
        if command[1:] == ["login", "status"]:
            status = 1 if failure == "not_logged_in" else 0
        return subprocess.CompletedProcess(command, status, output.encode(), b"")

    monkeypatch.setattr(codex_adapter.subprocess, "run", fake_run)
    with pytest.raises(ValueError):
        codex_adapter.check_cli()
    assert all(
        call in [["--version"], ["exec", "--help"], ["login", "status"]]
        for call in calls
    )


def test_adapter_edits_only_fixture_and_passes_separate_memory_data(
    tmp_path, monkeypatch
):
    task = TASKS[0]
    workspace = tmp_path / "workspace with spaces"
    run_tasks.prepare(workspace, task)
    observed = {}
    real_run = subprocess.run
    fake_cli = tmp_path / "fake_cli.py"
    fake_cli.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "prompt = sys.stdin.buffer.read().decode('utf-8')\n"
        "Path('received.txt').write_text(prompt, encoding='utf-8')\n"
        "p = Path('policy.py')\n"
        "p.write_text(p.read_text().replace('hours_before > 6', 'hours_before >= 6'))\n"
        "print(json.dumps({'type': 'turn.started'}))\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 123, 'output_tokens': 45}}))\n"
    )

    def invoke_fake(command, **kwargs):
        if command[0] != "codex":
            return real_run(command, **kwargs)
        observed["command"] = command
        return real_run([sys.executable, str(fake_cli), *command[1:]], **kwargs)

    monkeypatch.setattr(codex_adapter.subprocess, "run", invoke_fake)
    memory = 'Use ≥6 hours. Literal "quotes" and\nnewlines are data.'
    result = codex_adapter.run_request(
        {
            "protocol_version": 1,
            "workspace": str(workspace),
            "prompt": task.prompt,
            "memory_context": memory,
        },
        model="explicit-test-model",
        effort="low",
    )
    assert run_tasks.grade(workspace, task)
    assert result["model"] == "explicit-test-model"
    assert result["usage"] == {"input_tokens": 123, "output_tokens": 45}
    assert "requested_cli_argument" in result["agent_metadata"]["model_identity_source"]
    received = (workspace / "received.txt").read_text(encoding="utf-8")
    assert (
        task.prompt in received and json.dumps(memory, ensure_ascii=False) in received
    )
    command = observed["command"]
    # Public CLI contract: new session, explicit model and bounded permissions.
    assert "--ephemeral" in command and "resume" not in command
    assert "--ignore-user-config" in command
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert "features.memories=false" in command
    assert "memories.generate_memories=false" in command
    assert "features.multi_agent=false" in command
    assert "--ignore-rules" not in command


@pytest.mark.parametrize(
    "override",
    [
        {"protocol_version": True},
        {"protocol_version": 2},
        {"workspace": "relative"},
        {"prompt": " "},
        {"memory_context": {}},
    ],
)
def test_invalid_requests_never_launch_codex(tmp_path, monkeypatch, override):
    def must_not_run(*args, **kwargs):
        pytest.fail("invalid request launched a process")

    monkeypatch.setattr(codex_adapter.subprocess, "run", must_not_run)
    request = {
        "protocol_version": 1,
        "workspace": str(tmp_path),
        "prompt": "fix",
        "memory_context": "",
    }
    request.update(override)
    with pytest.raises(ValueError):
        codex_adapter.run_request(request, model="test-model")


def test_runner_codex_preflight_failure_does_not_create_results(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tasks.py",
            "--codex-model",
            "explicit-model",
            "--output",
            str(output),
            "--codex-executable",
            str(tmp_path / "missing-codex"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        run_tasks.main()
    assert error.value.code == 2
    assert "Codex CLI was not found" in capsys.readouterr().err
    assert not output.exists()


def test_check_agent_reports_configuration_without_evaluating(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        codex_adapter, "check_cli", lambda executable: {"cli_version": "test-double"}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tasks.py",
            "--codex-model",
            "explicit-model",
            "--check-agent",
            "--output",
            str(tmp_path / "unused"),
        ],
    )
    run_tasks.main()
    report = json.loads(capsys.readouterr().out)
    assert report["requested_model"] == "explicit-model"
    assert report["reasoning_effort"] == "medium"
    assert not (tmp_path / "unused").exists()


def test_adapter_subprocess_failure_preserves_actionable_diagnostic(tmp_path):
    adapter = Path(codex_adapter.__file__).resolve()
    with pytest.raises(RuntimeError, match="Codex adapter"):
        run_tasks.run_agent(
            [
                sys.executable,
                str(adapter),
                "--model",
                "test-model",
                "--codex-executable",
                str(tmp_path / "missing-codex"),
            ],
            {
                "protocol_version": 1,
                "workspace": str(tmp_path),
                "prompt": "Fix the function",
                "memory_context": "",
            },
            timeout=10,
        )
