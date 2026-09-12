# Evaluation: from retrieval to completed work

There are two separate checks. Neither establishes production performance by itself.

## Retrieval diagnostic

```bash
AGENT_MEMORY_EMBEDDER=hashing python eval/run_eval.py
```

The existing seven-query benchmark reports relevant-memory recall, token counts, a random control, paraphrase sensitivity and distractor scaling. Its body-token budget uses the Python API, not MCP's rendered-text budget. The same small dataset informed the relevance floor, so it is a development diagnostic rather than an independent test set. Results remain in [eval/results.md](../eval/results.md).

## Executable coding tasks

```bash
python eval/run_tasks.py --verify-fixtures
```

The new suite contains 30 tasks across three deliberately small, synthetic Python projects: booking policies, asset handling and job queues. Tasks exercise boundaries, filtering, byte ranges, hashing, time conversion and changed decisions. Seven tasks include a superseded policy. Six tasks (the first two in each project) are reserved for calibration; the remaining 24 are the default test split.

Fixture validation checks two implementations for every task: the known bug must fail and the reference fix must pass. The committed [validation result](../eval/fixture-validation.json) reports 30/30. This is **grader verification**, not an agent success rate.

Run an actual coding agent through a local executable adapter:

```bash
python eval/run_tasks.py \
  --agent-command '["/absolute/path/to/agent-wrapper"]' \
  --agent-label 'model-version/settings/wrapper-commit' \
  --split test --repetitions 3 --seed 0 \
  --output eval/task-results/run-001
```

No provider API or credentials are built into this runner. The adapter uses your existing agent and authentication. The default test run invokes it 216 times (24 tasks × 3 arms × 3 repetitions); use calibration with one repetition to check the adapter first. Any provider charges come from the agent you choose.

### Run with Codex CLI

The bundled adapter removes the need to write a wrapper. Install and sign in to Codex CLI in your evaluation environment first. Use an explicit model available to that account in place of `YOUR_MODEL` below; the adapter does not choose a model or start a login flow.

```bash
# Check installation, required flags and saved login; no model calls.
python eval/run_tasks.py --codex-model YOUR_MODEL --check-agent

# Adapter calibration: 6 tasks × 3 arms × 1 repetition = 18 sessions.
AGENT_MEMORY_EMBEDDER=hashing python eval/run_tasks.py \
  --codex-model YOUR_MODEL --codex-reasoning-effort medium \
  --split calibration --repetitions 1 --seed 0 \
  --output eval/task-results/codex-calibration-001

# After checking calibration errors and freezing the settings: 216 sessions.
AGENT_MEMORY_EMBEDDER=hashing python eval/run_tasks.py \
  --codex-model YOUR_MODEL --codex-reasoning-effort medium \
  --split test --repetitions 3 --seed 0 \
  --output eval/task-results/codex-test-001
```

Use `--codex-executable /absolute/path/to/codex` if it is outside PATH. On Windows, use a native executable, or run the commands in WSL; shell `.cmd` wrappers are not a supported adapter executable. In PowerShell set `$env:AGENT_MEMORY_EMBEDDER = "hashing"` first and enter each Python command on one line.

Each invocation starts a new `codex exec --json --ephemeral` session with workspace-write permissions. The adapter skips user configuration, disables built-in memory use/generation, web search and subagents, and sends the task and memory in separate text sections. It keeps saved CLI authentication and does not edit configuration or bypass execution rules. Managed settings, global instructions and skills can still affect a run: use a clean, dedicated evaluation environment and document those settings. This adapter measures the effect of **supplied recall text**; it does not test whether Codex independently discovers or calls the MCP tools.

Preflight failures stop before a results directory or paid task session is created. Preflight does not establish model access, available quota or sandbox compatibility; calibration checks those. Keep settings unchanged for the test split and do not tune on test results.

Reports record CLI version, adapter SHA-256, requested model, reasoning effort and per-run model-identity provenance. The CLI event contract does not guarantee a resolved server model version, so the model label is explicitly the **requested CLI argument**, not a verified immutable model ID. Keep aliases and their evaluation date visible when reporting results.

Usage comes from the single `turn.completed` event after tool rounds. Cached-input and reasoning-output components are retained without adding them again to the input/output totals. Missing usage remains null; failed/incomplete sessions, malformed telemetry or multiple completion events are errors. Token totals are CLI telemetry, not a provider billing estimate. The runner retains up to 2,000 bytes of failure stderr in the error field; review diagnostic text before sharing results.

The interface follows OpenAI's [non-interactive mode](https://developers.openai.com/codex/noninteractive), [CLI reference](https://developers.openai.com/codex/cli/reference) and [configuration reference](https://developers.openai.com/codex/config-reference), checked September 12, 2026. Tests use deterministic CLI doubles and real local subprocesses. Preflight also passed against Codex CLI 0.154.0; two live connection pilots timed out with an Unauthorized response in the diagnostic retry. **A completed live Codex task remains unverified.** See the [verification record](verification-codex-adapter.md).

### Adapter contract

The runner launches the command without a shell, in a fresh temporary workspace for each run, with one JSON request on stdin:

```json
{
  "protocol_version": 1,
  "workspace": "/temporary/workspace",
  "task_id": "booking/can_cancel",
  "prompt": "Fix can_cancel in policy.py according to this project's current policy. ...",
  "memory_context": "..."
}
```

The adapter must start a **fresh agent session**, supply the prompt and memory as distinct task/data sections, let it edit `policy.py`, and wait until the session finishes. Disable unrelated persistent memory and use identical model, tools, limits and settings in all arms. Write logs to stderr and one JSON response to stdout:

```json
{
  "model": "actual-model-version",
  "usage": {"input_tokens": 12000, "output_tokens": 1500}
}
```

Numbers above illustrate the schema only. Usage must cover the whole agent session, including tool rounds. If the client cannot report that, return `"usage": null`; never substitute prompt-only counts. Record caching/reasoning accounting conventions and the wrapper's version alongside published results.

### Fair comparison

| Arm | Supplied context |
| --- | --- |
| `no_memory` | No memory context |
| `curated_markdown` | All ten current project policies in a maintained Markdown document |
| `engine` | The same policies, retrieved through production rendering, hashing relevance floor and a 400-token cap; retired policies remain stored but excluded |

All arms receive identical source and task prompts. The maintained Markdown arm gets current facts rather than being deliberately polluted with old ones. Task/arm/repetition combinations are shuffled with a recorded seed. Checks and reference answers are not included in the agent request or workspace; they are public in this repository, so this is not a benchmark hardened against cheating. Fixture functions intentionally omit some business requirements that memory supplies: that measures policy recovery, not general coding skill.

The agent and grader execute code locally. Use a disposable container or VM for untrusted agents; the runner is **not a sandbox**. On POSIX, timeout cleanup kills the agent process group; adapters must keep tool children in that group. Windows cleanup uses `taskkill /T /F` on the adapter's process tree and reports an error if cleanup fails. Detached or deliberately escaped processes are outside this contract. Grading has a separate ten-second timeout. Do not give benchmark agents access to production files or credentials they do not need.

### Reporting

`config.json` records settings, dataset hash, tokenizer and embedding configuration. `runs.jsonl` is flushed after each run so interrupted work remains inspectable. Each row retains failures, errors, latency, supplied memory tokens and reported session usage. `summary.json` reports results by arm and fresh/superseded scenario. Existing output directories containing a run are never overwritten.

Missing usage stays unknown; the mean full-session token figure is withheld if any run in that arm lacks telemetry. Errors and timeouts count as failed runs. Report paired per-task differences and uncertainty before generalizing; repeated runs of the same task are not independent new tasks. Elapsed time includes adapter execution and grading, but excludes context construction.

**Current evidence:** fixture validation and runner tests pass. A live coding-agent comparison has not been run in this candidate. The next evidence step is a frozen test-split run, followed by tasks from independently maintained repositories. Do not claim coding improvements or whole-session token savings from the retrieval table or reference fixes.
