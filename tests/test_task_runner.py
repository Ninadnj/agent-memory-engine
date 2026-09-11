import json
from pathlib import Path
import subprocess
import sys

import pytest

from agent_memory import count_tokens
from run_tasks import context_for, evaluate_tasks, grade, prepare, run_agent, summarize
from task_cases import TASKS


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_task_grader_rejects_bug_and_accepts_reference(tmp_path, task):
    prepare(tmp_path, task)
    assert not grade(tmp_path, task)
    prepare(tmp_path, task, reference=True)
    assert grade(tmp_path, task)


def test_current_policy_context_and_budget():
    for task in TASKS:
        context = context_for(task, "engine")
        assert count_tokens(context) <= 400
        if task.previous_policy:
            assert task.previous_policy not in context
        assert task.policy in context_for(task, "curated_markdown")
        assert context_for(task, "no_memory") == ""


def test_runner_runs_matched_arms_and_retains_missing_telemetry(tmp_path, monkeypatch):
    import run_tasks

    task = TASKS[0]
    monkeypatch.setattr(run_tasks, "TASKS", [task])
    wrapper = tmp_path / "agent.py"
    # A tiny deterministic test double, not a reported agent benchmark.
    wrapper.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "request = json.load(sys.stdin)\n"
        "path = Path(request['workspace']) / 'policy.py'\n"
        "path.write_text(path.read_text().replace('hours_before > 6', 'hours_before >= 6'))\n"
        "print(json.dumps({'model': 'test-double', 'usage': None}))\n"
    )
    report = evaluate_tasks(
        [sys.executable, str(wrapper)],
        output=tmp_path / "results",
        label="test-double",
        split="calibration",
        repetitions=1,
        timeout=10,
    )
    assert all(row["passed"] == row["runs"] == 1 for row in report["summary"].values())
    assert all(
        row["mean_full_session_tokens"] is None for row in report["summary"].values()
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "runs.jsonl").read_text().splitlines()
    ]
    assert {row["arm"] for row in rows} == {"engine", "curated_markdown", "no_memory"}
    assert len({row["task"] for row in rows}) == 1
    with pytest.raises(FileExistsError):
        evaluate_tasks(
            [sys.executable, str(wrapper)],
            output=tmp_path / "results",
            label="test-double",
        )


def test_agent_timeout_is_bounded(tmp_path):
    with pytest.raises(TimeoutError):
        run_agent(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            {"workspace": str(tmp_path)},
            timeout=0.1,
        )


def test_errors_count_as_failed_runs_and_missing_usage_stays_unknown():
    rows = [
        {
            "arm": "engine",
            "passed": False,
            "error": "timeout",
            "usage": None,
            "seconds": 10,
            "memory_tokens": 100,
            "scenario": "fresh",
        },
        {
            "arm": "engine",
            "passed": True,
            "error": None,
            "usage": {"input_tokens": 200, "output_tokens": 100},
            "seconds": 5,
            "memory_tokens": 100,
            "scenario": "fresh",
        },
    ]
    report = summarize(rows)["engine"]
    assert report["runs"] == 2 and report["pass_rate"] == 0.5 and report["errors"] == 1
    assert report["mean_full_session_tokens"] is None


def test_mcp_demo_uses_two_stdio_processes():
    pytest.importorskip("mcp")
    path = Path(__file__).resolve().parents[1] / "examples" / "handoff_demo.py"
    result = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, timeout=40
    )
    assert result.returncode == 0, result.stderr
    assert "All checks passed. Two server processes" in result.stdout
