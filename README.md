# Agent Memory Engine

[![CI](https://github.com/Ninadnj/agent-memory-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Ninadnj/agent-memory-engine/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Local project memory that coding agents can share, inspect and correct.**

Keep architectural decisions, bug findings and session handoffs in one project store. Retrieve relevant notes within a text token budget. When a decision changes, retain its history and keep stale agents from overwriting the correction.

For developers switching between coding agents or returning to a project after a break. It provides a Python library, CLI and local MCP server. No model API key is required; the lightweight installation uses offline lexical retrieval.

**v0.4 release candidate:** new correction APIs and store format. Existing IDs remain intact. Read the [migration notes](docs/migration-v0.4.md) before upgrading a shared store.

## Try it in five minutes

Python 3.10+ and Git are required. From this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[mcp]"
python examples/handoff_demo.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

The demo starts **two real MCP server processes** against a temporary store. Its output:

```text
1. Agent B receives Agent A's decision and handoff over MCP.
2. Agent A's stale deletion is rejected: expected revision 1, current 2.
3. A replacement decision is recalled; the old decision and its revisions remain inspectable.
All checks passed. Two server processes; one temporary store; no model calls.
```

This is a scripted protocol demonstration. It does not measure an LLM's ability to use memory.

To use your project's store:

```bash
agent-memory --agent developer write "Bookings are stored in UTC; the UI converts to local time." --type decision
agent-memory recall "booking timezone UTC" --budget 200
agent-memory doctor
agent-memory recall "booking timezone UTC" --budget 200 --explain
```

Inside a Git project, the default is `.agent_memory/store.json`. Outside one, it is `~/.agent_memory/store.json`. `--path` or `AGENT_MEMORY_PATH` selects an explicit store; use the same absolute path for clients that should share it. Keep memory files out of Git.

## Correct a memory without losing the explanation

```python
from agent_memory import HashingEmbedder, MemoryStore

store = MemoryStore(".agent_memory/store.json", embedder=HashingEmbedder())
old = store.write("Queue retries are limited to three attempts.", type="decision",
                  source={"path": "queue/policy.py"}, agent="reviewer")
new = store.supersede(old.id, "Queue retries are limited to five attempts.",
                      expected_revision=old.revision, agent="implementer")
assert store.get(old.id).superseded_by == new.id
```

Use `update` for a correction to the same memory, `supersede` for a replacement decision, and `forget` for permanent deletion. MCP and CLI edits require the revision you read. Python accepts `expected_revision` for compatibility; **pass it when multiple callers can edit**. Updates retain the last 20 prior revisions. Forgetting removes the entry and its history from the current store.

CLI equivalents, replacing `MEMORY_ID` and revision with values from `list`:

```bash
agent-memory list
agent-memory inspect MEMORY_ID
agent-memory update MEMORY_ID "Corrected fact." --expected-revision 1
agent-memory supersede MEMORY_ID "Replacement decision." --expected-revision 2
agent-memory export backup.json
```

Sources can include a project-relative `path`, Git `commit`, `event`, or caller-supplied `verified_at`. Inspection flags a file that changed since its recorded commit. These references help investigation; they do not certify that a memory is true.

## Connect a coding agent

See [client setup](docs/clients.md) for Claude Code, Codex CLI and Cursor. All use the same stdio command, `agent-memory-mcp`, and can share a store on the same machine.

| Task | MCP tool |
| --- | --- |
| Start a session | `memory_boot(task, budget_tokens=300)` |
| Find relevant notes | `memory_recall(query, k=5, budget_tokens=300)` |
| Save a fact with a source | `memory_write(text, type="fact", source=...)` |
| Leave the next session a starting point | `memory_handoff(done, next_steps, warnings="")` |
| Inspect identity, source and history | `memory_get(id)`, `memory_list()` |
| Correct or replace a decision | `memory_update(...)`, `memory_supersede(...)` with `expected_revision` |
| Permanently remove a memory | `memory_forget(id, expected_revision)` |
| Inspect store totals | `memory_stats()` |

Optional Claude Code hooks inject fresh startup notes and record observed Git changes. Install with `agent-memory install-hooks`; enable per-prompt recall with `--with-prompt-recall`. Hooks are fallible observations and do not establish who authored a change.

## What is guaranteed, and what is measured?

- New IDs use `mem_` plus UUID4. Deletion cannot reset an ID counter. Unique explicit IDs and legacy `mem_0001` IDs remain usable; duplicate explicit IDs are rejected.
- Mutations reload under an OS lock and replace the JSON file atomically. Failed writes roll back the local snapshot. This contract covers cooperating v0.4 processes on a local filesystem.
- Correction revisions reject stale edits. Superseded notes leave normal recall; old handoffs and worklogs leave startup context. Exact duplicates are skipped only within the same type and source; similar wording never silently merges a contradiction.
- MCP/CLI context budgets include the returned text's labels, IDs and source references. `cl100k_base` is used when tiktoken is installed, otherwise counting is approximate. Client wrappers, tool schemas and full session usage are outside this budget. The Python store API budgets memory bodies only.

The [retrieval benchmark](eval/results.md) is a small diagnostic: 14 memories and seven queries. Its paraphrase results expose the limits of lexical matching. It is not evidence of improved coding outcomes.

The [coding evaluation](docs/evaluation.md) adds **30 executable tasks across three synthetic projects**, with six calibration tasks and 24 test tasks. It compares no memory, curated Markdown and engine recall with the same external agent. A [bundled Codex CLI adapter](docs/evaluation.md#run-with-codex-cli) provides setup checks and a direct evaluation command. [Fixture validation](eval/fixture-validation.json) verifies all 30 graders. **The live-agent comparison is deferred for v0.4.0rc1; no live-agent performance result is published.** You can install, use and test the engine without running that comparison or supplying model credentials.

## Install options

| Installation | Contents |
| --- | --- |
| `pip install -e .` | Python library and CLI; NumPy only |
| `pip install -e ".[mcp]"` | Adds the MCP server |
| `pip install -e ".[real,mcp]"` | Adds optional sentence-transformers embeddings and tiktoken; first model use can download weights |
| `pip install -e ".[dev]"` | Tests, MCP, exact tokenizer and build tools |

Set `AGENT_MEMORY_EMBEDDER=hashing` for offline behavior. A new store otherwise prefers the semantic backend when installed. Existing v0.4 stores retain their embedding configuration unless explicitly overridden. Configuration changes trigger re-embedding; immutable model revisions are recommended for reproducibility.

## Scope

This is a local JSON + NumPy store for small project memories. It has no network server, authorization layer, cloud sync or automatic truth checker. Memory content may reach your coding agent's provider. Treat it as fallible data and verify operational claims against code.

Keep maintained Markdown if a few short files already solve your problem. Use this engine when selective retrieval, cross-session handoffs and inspectable corrections justify the extra component. Vector database migrations and automatic LLM compaction are deferred until measurements justify them.

The original Markdown convention remains in [scaffold/](scaffold/). Import existing notes with `python scripts/ingest_markdown.py path/to/notes --path .agent_memory/store.json`; oversized sections are split into bounded chunks.

[Architecture](docs/architecture.md) · [Migration](docs/migration-v0.4.md) · [Evaluation](docs/evaluation.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Changelog](CHANGELOG.md) · [Release checks](docs/release.md)
