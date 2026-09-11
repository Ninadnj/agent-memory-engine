# v0.4 candidate verification

Verified locally on Linux, Python 3.12.14, NumPy 2.3.5, pytest 9.1.1, tiktoken 0.14.0 and MCP 2.2.0. A separate dependency environment tested MCP 1.30.0. The wheel installation check resolved NumPy 2.5.3 and MCP 2.2.0 in a fresh environment.

Work began from the existing UUID fix branch (`1230ca8`), preserving that work and isolating this candidate on `feat/reliable-memory-v0.4`. The original checkout remained untouched. This branch includes the earlier ID fix when compared with main.

## Baseline and failing regressions

Before implementation, the existing suite passed **159 tests**, with **1 optional backend module skipped**. Fourteen new tests then failed against the existing code. The test-only commit precedes the implementation commit, making that failure state reviewable.

```bash
AGENT_MEMORY_EMBEDDER=hashing python -m pytest -q -ra --ignore=tests/test_reliability.py
# 159 passed, 0 failed, 1 skipped
AGENT_MEMORY_EMBEDDER=hashing python -m pytest -q --tb=short tests/test_reliability.py
# Before implementation: 0 passed, 14 failed, 0 skipped
```

The failures cover the real hook prompt payload, stale startup handoffs, correction freshness, negation/type deduplication, stale snapshot overwrites, age-based lock stealing, model configuration mismatch, zero/negative result limits, failed embeddings and invalid new text.

## Final checks

After installing `.[dev]`, these commands reproduce the test selections. The local runner supplied prepared dependencies through `PYTHONPATH` and redirected bytecode through `PYTHONPYCACHEPREFIX`; neither changes the selections below.

```bash
AGENT_MEMORY_EMBEDDER=hashing python -m pytest -q -ra tests/test_store.py tests/test_persistence.py tests/test_mcp_server.py tests/test_reliability.py tests/test_revisions.py tests/test_hook_contract.py tests/test_diagnostics.py
# 136 passed, 0 failed, 0 skipped

AGENT_MEMORY_EMBEDDER=hashing python -m pytest -q -ra
# 259 passed, 0 failed, 1 skipped

# In the separate MCP 1.30.0 environment:
AGENT_MEMORY_EMBEDDER=hashing python -m pytest -q -ra tests/test_mcp_server.py tests/test_task_runner.py
# 53 passed, 0 failed, 0 skipped

# The same focused selection also passed against MCP 2.2.0:
AGENT_MEMORY_EMBEDDER=hashing python -m pytest -q --tb=short tests/test_mcp_server.py tests/test_task_runner.py
# 53 passed, 0 failed, 0 skipped

AGENT_MEMORY_EMBEDDER=hashing python eval/run_eval.py
git diff --exit-code -- eval/results.md eval/results.json
# Existing retrieval results unchanged

AGENT_MEMORY_EMBEDDER=hashing python eval/run_tasks.py --verify-fixtures
# 30 validated: each broken implementation rejected and reference accepted

AGENT_MEMORY_EMBEDDER=hashing python examples/handoff_demo.py
AGENT_MEMORY_EMBEDDER=hashing python examples/quickstart.py
# Both pass; the handoff demo also runs in the test suite

python -m build --no-isolation
# Wheel and sdist built with preinstalled hatchling/build
python scripts/check_wheel.py
# Fresh environment: wheel import, legacy-ID correction/reopen, MCP construction and CLI doctor

ruff check --select E9,F63,F7,F82,F401 src tests eval scripts examples
git diff --check
# Both clean
```

The skipped module is `tests/test_sentence_transformers.py`: optional sentence-transformers/model weights were not installed in the local test environment. No existing behavioral assertion was loosened. Legacy schema expectations were extended to include revision fields; rendered-budget assertions were strengthened to include all returned text.

## Scope of evidence

The tests cover old IDs, stale IDs, duplicates, multiple store instances, failed writes, corrupted data, restart behavior, source changes, correction conflicts, retained history, startup freshness, rendered budgets, executable task grading and real MCP subprocess transport.

Not run locally: Windows/macOS, Python 3.10, optional semantic model tests, actual Claude Code/Codex/Cursor application sessions, or a live coding-agent comparison. CI definitions add Windows/Python/SDK coverage and a manually requested semantic job, but definitions alone are not evidence those jobs passed. Published agent-performance improvements remain unclaimed.

This candidate retains the JSON/NumPy backend. Breaking changes and limits are explicit in the [migration notes](migration-v0.4.md). A prepared package is not yet a published GitHub/PyPI release; publishing status must be checked separately.
