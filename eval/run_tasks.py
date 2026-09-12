"""Run executable coding tasks against a user-supplied, provider-neutral agent.

No model API calls or credentials are built in. --verify-fixtures checks the
grader against known broken and reference implementations, not an agent.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_memory import HashingEmbedder, MemoryStore, count_tokens, default_min_score
from agent_memory.embeddings import embedding_config
from agent_memory.rendering import recall_context
from agent_memory.tokens import using_exact_tokenizer
from task_cases import TASKS, Task, project_source

ARMS = ("no_memory", "curated_markdown", "engine")


def prepare(workspace: Path, task: Task, *, reference=False):
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "policy.py").write_text(
        project_source(task.project, task if reference else None), encoding="utf-8"
    )
    (workspace / "README.md").write_text(
        f"# {task.project} policy fixture\n\nSmall synthetic project for a coding evaluation.\n"
        "Fix only the requested policy function. The evaluator checks boundary behavior.\n",
        encoding="utf-8",
    )


def grade(workspace: Path, task: Task) -> bool:
    # Checks live outside the agent's workspace and are not sent in the request.
    # This is not an adversarial sandbox; run untrusted agents in a container.
    code = (
        "import hashlib, importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('candidate', {str(workspace / 'policy.py')!r})\n"
        "p = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(p)\n"
        + task.checks
        + "\n"
    )
    with tempfile.TemporaryDirectory(prefix="memory-grader-") as directory:
        check = Path(directory) / "check.py"
        check.write_text(code, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-B", str(check)],
                cwd=directory,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False


def verify_fixtures() -> dict:
    rows = []
    with tempfile.TemporaryDirectory(prefix="memory-fixtures-") as directory:
        workspace = Path(directory)
        for task in TASKS:
            prepare(workspace, task)
            broken_rejected = not grade(workspace, task)
            prepare(workspace, task, reference=True)
            reference_passed = grade(workspace, task)
            rows.append(
                {
                    "task": task.id,
                    "broken_rejected": broken_rejected,
                    "reference_passed": reference_passed,
                }
            )
    return {
        "kind": "fixture_validation_not_agent_performance",
        "tasks": len(rows),
        "projects": 3,
        "calibration": 6,
        "test": 24,
        "passed": sum(r["broken_rejected"] and r["reference_passed"] for r in rows),
        "results": rows,
    }


def context_for(task: Task, arm: str, budget=400) -> str:
    policies = [t for t in TASKS if t.project == task.project]
    if arm == "no_memory":
        return ""
    if arm == "curated_markdown":
        # A maintained Markdown baseline gets the same CURRENT facts and no
        # artificial contradictions. It receives the entire document.
        return "# Current project policies\n\n" + "\n".join(
            f"- {t.policy}" for t in policies
        )
    if arm != "engine":
        raise ValueError(f"unknown arm {arm}")
    store = MemoryStore(embedder=HashingEmbedder())
    for policy in policies:
        source = {"path": "policy.py", "event": "project_policy"}
        if policy.previous_policy:
            previous = store.write(
                policy.previous_policy, type="decision", source=source
            )
            store.supersede(
                previous.id, policy.policy, expected_revision=1, source=source
            )
        else:
            store.write(policy.policy, type="decision", source=source)
    return recall_context(
        store,
        task.prompt,
        k=5,
        budget=budget,
        min_score=default_min_score(store.embedder),
    )


def run_agent(command: list[str], request: dict, timeout: float) -> dict:
    """One JSON request on stdin, one JSON result on stdout; logs on stderr."""
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        def failure_detail():
            stderr.seek(0, os.SEEK_END)
            stderr.seek(max(0, stderr.tell() - 2000))
            detail = stderr.read().decode("utf-8", errors="replace").strip()
            return f": {detail}" if detail else ""

        process = subprocess.Popen(
            command,
            cwd=request["workspace"],
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            start_new_session=os.name != "nt",
        )
        try:
            process.communicate(json.dumps(request).encode("utf-8"), timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                # Killing only the wrapper leaves Codex and its tools running.
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=10, check=True,
                    )
                except (OSError, subprocess.SubprocessError):
                    process.kill()
                    process.wait()
                    raise RuntimeError("agent timed out; process-tree cleanup failed") from None
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise TimeoutError("agent timed out" + failure_detail()) from None
        if process.returncode:
            raise RuntimeError(
                f"agent exited with status {process.returncode}"
                + failure_detail()
            )
        stdout.seek(0)
        raw = stdout.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ValueError("agent response exceeds 1 MiB")
        result = json.loads(raw)
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("model"), str)
            or not result["model"]
        ):
            raise ValueError("agent response requires a nonempty model label")
        usage = result.get("usage")
        if result.get("agent_metadata") is not None and not isinstance(
            result["agent_metadata"], dict
        ):
            raise ValueError("agent_metadata must be an object or null")
        if usage is not None:
            if not isinstance(usage, dict):
                raise ValueError("usage must be an object or null")
            for key in ("input_tokens", "output_tokens"):
                value = usage.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        "usage must report nonnegative full-session input/output tokens"
                    )
        return result


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        if not selected:
            continue
        usage = [row["usage"] for row in selected if row["usage"] is not None]
        summary[arm] = {
            "runs": len(selected),
            "passed": sum(row["passed"] for row in selected),
            "errors": sum(row["error"] is not None for row in selected),
            "pass_rate": sum(row["passed"] for row in selected) / len(selected),
            "mean_seconds": sum(row["seconds"] for row in selected) / len(selected),
            "mean_memory_tokens": sum(row["memory_tokens"] for row in selected)
            / len(selected),
            "usage_reported_runs": len(usage),
            # Never call missing telemetry zero or silently average a subset.
            "mean_full_session_tokens": (
                sum(u["input_tokens"] + u["output_tokens"] for u in usage) / len(usage)
                if len(usage) == len(selected)
                else None
            ),
            "by_scenario": {
                scenario: {
                    "runs": sum(row["scenario"] == scenario for row in selected),
                    "passed": sum(
                        row["scenario"] == scenario and row["passed"]
                        for row in selected
                    ),
                }
                for scenario in sorted({row["scenario"] for row in selected})
            },
        }
    return summary


def evaluate_tasks(
    command,
    *,
    output: Path,
    label: str,
    split="test",
    repetitions=3,
    seed=0,
    timeout=300,
    agent_config=None,
):
    cases = [task for task in TASKS if task.split == split]
    jobs = [
        (task, arm, repeat)
        for task in cases
        for arm in ARMS
        for repeat in range(repetitions)
    ]
    random.Random(seed).shuffle(jobs)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "runs.jsonl"
    # Refuse accidental replacement of an expensive run.
    with raw_path.open("x", encoding="utf-8") as raw:
        metadata = {
            "kind": "coding_agent_evaluation",
            "fixture_kind": "synthetic_policy_projects",
            "agent_label": label,
            "agent_config": agent_config,
            "split": split,
            "repetitions": repetitions,
            "seed": seed,
            "timeout_seconds": timeout,
            "engine_budget": 400,
            "embedding_config": embedding_config(HashingEmbedder()),
            "tokenizer": "cl100k_base" if using_exact_tokenizer() else "approximate",
            "dataset_sha256": hashlib.sha256(
                json.dumps([asdict(t) for t in TASKS], sort_keys=True).encode()
            ).hexdigest(),
        }
        (output / "config.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        rows = []
        for task, arm, repeat in jobs:
            with tempfile.TemporaryDirectory(prefix="memory-task-") as directory:
                workspace = Path(directory)
                prepare(workspace, task)
                context = context_for(task, arm)
                request = {
                    "protocol_version": 1,
                    "workspace": str(workspace),
                    "task_id": task.id,
                    "prompt": task.prompt,
                    "memory_context": context,
                }
                row = {
                    "task": task.id,
                    "project": task.project,
                    "scenario": task.scenario,
                    "arm": arm,
                    "repeat": repeat,
                    "memory_tokens": count_tokens(context),
                    "passed": False,
                    "error": None,
                    "usage": None,
                    "model": None,
                    "agent_metadata": None,
                }
                start = time.monotonic()
                try:
                    result = run_agent(command, request, timeout)
                    row["model"], row["usage"] = result["model"], result.get("usage")
                    row["agent_metadata"] = result.get("agent_metadata")
                    row["passed"] = grade(workspace, task)
                except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                row["seconds"] = round(time.monotonic() - start, 3)
                rows.append(row)
                raw.write(json.dumps(row) + "\n")
                raw.flush()
    report = {
        **metadata,
        "models_reported": sorted({r["model"] for r in rows if r["model"]}),
        "summary": summarize(rows),
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-fixtures", action="store_true")
    agent = parser.add_mutually_exclusive_group()
    agent.add_argument(
        "--agent-command",
        help="JSON argv array, e.g. '[\"/absolute/path/to/wrapper\"]'",
    )
    agent.add_argument("--codex-model", help="Use the bundled Codex CLI adapter with this model")
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--codex-reasoning-effort", default="medium")
    parser.add_argument("--check-agent", action="store_true", help="Check Codex setup without model calls")
    parser.add_argument(
        "--agent-label", help="model/settings/version identifier for reproducibility"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "task-results")
    parser.add_argument("--split", choices=["calibration", "test"], default="test")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.verify_fixtures:
        result = verify_fixtures()
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["passed"] == result["tasks"] else 1)
    if args.repetitions < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("repetitions and timeout must be positive and finite")
    agent_config = None
    if args.codex_model:
        from codex_adapter import REASONING_EFFORTS, check_cli

        if not args.codex_model.strip():
            parser.error("--codex-model must not be empty")
        if args.codex_reasoning_effort not in REASONING_EFFORTS:
            parser.error("unsupported --codex-reasoning-effort")
        try:
            agent_config = check_cli(args.codex_executable)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        agent_config.update(
            requested_model=args.codex_model,
            reasoning_effort=args.codex_reasoning_effort,
            sandbox="workspace-write", user_config_loaded=False,
        )
        if args.check_agent:
            print(json.dumps(agent_config, indent=2))
            return
        command = [
            sys.executable, str(ROOT / "eval" / "codex_adapter.py"),
            "--model", args.codex_model, "--codex-executable", agent_config["executable"],
            "--reasoning-effort", args.codex_reasoning_effort,
        ]
        args.agent_label = args.agent_label or f"{agent_config['cli_version']}/{args.codex_model}"
    else:
        if args.check_agent:
            parser.error("--check-agent requires --codex-model")
        if not args.agent_command or not args.agent_label:
            parser.error("provide --codex-model OR (--agent-command and --agent-label), or --verify-fixtures")
        try:
            command = json.loads(args.agent_command)
        except ValueError:
            parser.error("--agent-command must be a JSON argv array")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(x, str) for x in command)
    ):
        parser.error("--agent-command must be a nonempty JSON array of strings")
    print(
        json.dumps(
            evaluate_tasks(
                command,
                output=args.output,
                label=args.agent_label,
                split=args.split,
                repetitions=args.repetitions,
                seed=args.seed,
                timeout=args.timeout,
                agent_config=agent_config,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
