# Releasing v0.4

The package version is `0.4.0rc1`. It is a reviewable release candidate, with an explicit format/API migration. A local build does not imply a GitHub or PyPI release exists.

## Reproduce the checks

```bash
python -m pip install -e ".[dev]"
AGENT_MEMORY_EMBEDDER=hashing python -m pytest -q -ra
AGENT_MEMORY_EMBEDDER=hashing python eval/run_eval.py
git diff --exit-code -- eval/results.md eval/results.json
python eval/run_tasks.py --verify-fixtures
python examples/handoff_demo.py
python examples/quickstart.py
python -m build
```

Install the generated wheel in a clean environment and run `agent-memory doctor` against a temporary path before publishing. CI also checks Python/OS combinations and both MCP SDK majors. The optional semantic job downloads a model and is run manually; its absence must not be presented as a pass.

## Candidate review

- Inspect the legacy migration, stale-revision errors, file-lock protocol and rollback behavior.
- Test the documented connection in at least one actual coding client, then record application/SDK/OS versions.
- Run the external-agent evaluation with a frozen test split before making coding-performance claims. A release without those results must keep the limitation visible.
- Check that no project memory files, secrets, private logs or local backup files entered the release.

## Publish

After review and successful required CI, use the reviewed commit for the tag and release. Build wheel and sdist from that tag, attach them and the changelog to a GitHub prerelease, and label it `v0.4.0rc1`. Publish to PyPI only with the project's configured publisher credentials/trusted publisher; no credentials are embedded here. Keep the original-store backup and rollback instructions in the release notes.

A stable `0.4.0` release should follow migration feedback and verified client sessions. Do not silently relabel an existing candidate artifact; build again with the stable version from its reviewed commit.

The implementation record and exact local results are in [verification](verification-v0.4.md).
