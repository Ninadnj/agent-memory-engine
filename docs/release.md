# Releasing v0.4

The current package version is `0.4.0rc2`. It remains a release candidate.
The [migration notes](migration-v0.4.md) cover format-3 stores and the rc2
Python API changes. A local build is not a published release.

## Candidate scope

rc2 protects returned snapshots, serializes operations within one store object,
separates records and JSON persistence, and defaults new stores to offline
hashing. The local JSON/NumPy backend and existing file format remain unchanged.

The live-agent comparison remains deferred. The tests, fixture checks and MCP
demo below require no model credentials. Verified client sessions and migration
feedback are still required before a stable release.

## Reproduce the checks

```bash
python -m pip install -e ".[dev]"
AGENT_MEMORY_EMBEDDER=hashing python -m pytest -q -ra
AGENT_MEMORY_EMBEDDER=hashing python eval/run_eval.py
git diff --exit-code -- eval/results.md eval/results.json
python eval/run_tasks.py --verify-fixtures
python examples/handoff_demo.py
python examples/quickstart.py
ruff check src tests eval scripts examples
ruff format --check src
python -m build
python scripts/check_wheel.py
```

CI checks Linux, macOS, Windows, Python 3.10/3.12 and MCP 1.x/2.x.
The optional semantic job downloads a model and is run manually.
See the [rc2 verification record](verification-rc2.md) for actual results;
a job's presence in CI configuration alone is not evidence of a pass.

## Review before publishing

- Check snapshot ownership, revision conflicts, rollback and both locking scopes.
- Read the Python compatibility changes; format 3 itself needs no conversion.
- Verify the documented connection in an actual coding client and record versions.
- Keep retrieval diagnostics distinct from real agent performance evidence.

## Publish

Build wheel and sdist from the reviewed commit after required CI passes.
Tag that commit `v0.4.0rc2` and attach the artifacts and changelog to a GitHub
prerelease. Publish to PyPI only through the project's configured publisher.
Do not silently relabel candidate artifacts as stable builds.

The earlier candidate's evidence remains in the
[v0.4 verification](verification-v0.4.md) and
[adapter verification](verification-codex-adapter.md) records.
