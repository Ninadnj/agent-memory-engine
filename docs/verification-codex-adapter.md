# Codex evaluation adapter verification

September 12, 2026. Follow-up to the [v0.4 candidate verification](verification-v0.4.md), on the same isolated review branch. The original checkout is preserved. Production memory code and the task dataset are unchanged in this follow-up.

## Changes

- `eval/codex_adapter.py`: fresh Codex sessions, explicit model/settings, setup checks, bounded event parsing and honest usage/provenance fields.
- `eval/run_tasks.py`: direct `--codex-model` command, preflight before evaluation, retained adapter metadata and failure diagnostics, Windows process-tree timeout cleanup.
- `tests/test_codex_adapter.py` and `tests/test_task_runner.py`: 32 added offline tests, including actual subprocess fixtures, a tool-child timeout check and retained timeout diagnostics.
- CI: macOS Python 3.12 added alongside Linux and Windows.
- README and evaluation guide: commands for setup checks, 18-session calibration and the frozen 216-session test run.

The existing quoted-executable hook review was checked against the already committed regression tests and marked resolved. No memory behavior or existing assertion was weakened.

## Local checks

Linux, Python 3.12.14, NumPy 2.5.3, pytest 9.1.1, MCP 2.2.0, tiktoken 0.14.0 and ruff 0.16.7. Dependencies were installed with `uv pip install -e '.[dev]'` into a fresh virtual environment. An initial attempt using the previous dependency directory aborted while importing a native cryptography extension with SIGBUS; it is not counted as a completed test run.

With that environment activated:

```bash
AGENT_MEMORY_EMBEDDER=hashing python -m pytest -q -ra tests/test_codex_adapter.py tests/test_task_runner.py tests/test_windows_contract.py
# 73 passed, 0 failed, 0 skipped

AGENT_MEMORY_EMBEDDER=hashing python -m pytest -q -ra
# 297 passed, 0 failed, 1 skipped

ruff check --select E9,F63,F7,F82,F401 src tests eval scripts examples
git diff --check
# Both clean

python eval/run_tasks.py --codex-model YOUR_MODEL --check-agent
# Exit 2 before evaluation: Codex CLI was not found on PATH
```

The optional semantic-model module accounts for the single skip. The initial check-agent invocation did not create benchmark results or make a model call. The subprocess doubles verify adapter behavior, not model quality.

## Real CLI and connection pilot

Codex CLI **0.154.0** was subsequently installed into an isolated directory outside the repository. The same check with `--codex-executable` pointing to that executable passed: version, required flags and saved login were recognized. No model request is needed for this preflight.

Two bounded connection pilots then attempted the calibration task `booking/can_cancel` with the engine context, requested model `gpt-5.5` and medium reasoning effort. Each timed out after 45 seconds. The diagnostic retry recorded an **Unauthorized** response. No completed task or usable token telemetry was returned. These attempts are connection checks, not a comparative benchmark; the 18-session calibration and 216-session test runs were not started.

Saved-login presence does not prove the service will accept the credentials. The next live run requires a working authenticated Codex connection. Raw local pilot diagnostics are excluded from Git and are not published as performance data.

On September 12, 2026, the maintainer chose to defer the live evaluation and continue candidate release preparation without it. Neither batch was started. The adapter and offline regression coverage remain available; no model authentication is required for the candidate's release checks.

## Completed platform CI

[CI run 34687492192](https://github.com/Ninadnj/agent-memory-engine/actions/runs/34687492192), on implementation commit `f18e156c7b0c473c301a816f7f4d0544417c4868`, completed with **7 jobs passed and 1 optional semantic job skipped**.

| Platform | Full hashing suite |
| --- | --- |
| Linux, Python 3.10 | 297 passed, 0 failed, 1 skipped |
| Linux, Python 3.12 | 297 passed, 0 failed, 1 skipped |
| Windows, Python 3.12 | 297 passed, 0 failed, 1 skipped |
| macOS, Python 3.12 | 297 passed, 0 failed, 1 skipped |

The other three passing jobs verify both MCP SDK majors and package build/fresh installation. Each full-suite skip is the optional semantic-model module; these results do not establish semantic retrieval quality or live agent performance.

## Remaining evidence

Current platform CI results are recorded on [PR #6](https://github.com/Ninadnj/agent-memory-engine/pull/6). A workflow definition alone is not evidence of a platform pass.

Completed authenticated Codex tasks, actual client use of the MCP tools, and the optional semantic model remain unverified. The adapter label records the requested model; it does not assert that an alias resolves to an immutable server model. Managed configuration and global instructions can still influence a run, so the [evaluation guide](evaluation.md#run-with-codex-cli) requires documenting a clean evaluation environment.

Release publication remains pending; these changes do not merge the PR, tag a release or publish a package.
