# Contributing

Start with a reproducible problem and the smallest change that solves it. This project favors a local store and explicit behavior over infrastructure added in anticipation of scale.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
AGENT_MEMORY_EMBEDDER=hashing python -m pytest -q -ra
python eval/run_eval.py
python eval/run_tasks.py --verify-fixtures
python examples/handoff_demo.py
ruff check src tests eval scripts examples
ruff format --check src
python -m build
```

On PowerShell, activate `.venv\Scripts\Activate.ps1` and set `$env:AGENT_MEMORY_EMBEDDER = "hashing"` before running tests.

For a bug, first demonstrate a failing regression and record the existing baseline. Preserve current work in a dedicated branch/worktree. Do not loosen behavioral assertions to accommodate a change. Include focused and full test results, migration effects and any unrun checks in the pull request.

Keep generated retrieval results current if their underlying behavior changes. Keep calibration and test tasks separate; do not tune retrieval on the coding test split. Never publish fixture/reference results as model performance. Include actual model/settings identifiers with agent runs.

Core dependencies should stay small. MCP and model integrations remain optional. For persistence changes, cover multiple instances, malformed data, failed writes, legacy IDs and restart behavior. For tool changes, test the real MCP schema and stdio boundary.

Keep client adapters thin, record rules in `models.py`, and file encoding in
`persistence.py`. Store operations own revisions, retrieval and transactions.
Return detached snapshots from public APIs; never expose live internal entries.
Include same-object threaded readers as well as separate writer processes when
changing transaction behavior. See [architecture](docs/architecture.md).

Feature requests should include a concrete workflow and what currently fails. A benchmark showing where the current store stops working is more useful than a new backend in search of a workload.
