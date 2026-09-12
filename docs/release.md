# Releasing v0.4

The package version is `0.4.0rc1`. It is a reviewable release candidate, with an explicit format/API migration. A local build does not imply a GitHub or PyPI release exists.

## Candidate scope

The live-agent comparison is deferred for this candidate. Release preparation uses the offline tests, fixture checks, MCP demo and package checks below; none requires model credentials or a live model session. The evaluation adapter remains available for future measurements. Do not describe the deferred comparison as passed or use the retrieval diagnostic to claim better coding outcomes or lower whole-session costs.

The candidate is available for review in [PR #6](https://github.com/Ninadnj/agent-memory-engine/pull/6). Stable `0.4.0` remains subject to migration feedback and verified coding-client sessions.

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

## Prerelease notes

Use the following scope when preparing the GitHub prerelease, with artifacts built from the reviewed tag:

**Agent Memory Engine v0.4.0rc1** adds durable local memory identity and traceable corrections for coding agents. UUID4 IDs prevent deletion from resetting the generated ID sequence; legacy and unique caller-supplied IDs stay usable. Revision checks reject stale edits, while bounded history and explicit supersession make changed decisions inspectable. Shared writes use OS locks, atomic replacement and rollback. The storage backend remains local JSON and NumPy.

The candidate also includes a five-minute MCP demo, CLI diagnostics, migration guidance and an optional coding-evaluation harness. The demo uses two real MCP server processes and makes no model calls.

**Validation:** the full hashing suite passed with **297 passed, 0 failed and 1 skipped** on Linux Python 3.10/3.12, Windows Python 3.12 and macOS Python 3.12. Both MCP SDK majors, package build and fresh installation passed. All 30 task graders reject the known broken implementation and accept the reference fix. The [verification record](verification-codex-adapter.md) links the completed CI and distinguishes these checks from agent performance.

**Upgrade:** stop all shared writers and save a byte-for-byte backup of the original store before upgrading every client. v0.4 reads formats 1/2 and writes format 3; mixed old/new writers are unsupported. MCP/CLI edits now require the inspected revision. For rollback, stop writers and restore the original backup with its matching old software; a v0.4 export is not a downgrade converter. Read the full [migration notes](migration-v0.4.md).

**Limitations:** the live-agent comparison is deferred, actual coding-client MCP use remains unverified, and the optional semantic-model check was skipped. This prerelease makes no claim of improved agent coding performance or whole-session token savings. Its concurrency contract covers cooperating v0.4 processes on a local filesystem.

## Publish

After review and successful required CI, use the reviewed commit for the tag and release. Build wheel and sdist from that tag, attach them and the changelog to a GitHub prerelease, and label it `v0.4.0rc1`. Publish to PyPI only with the project's configured publisher credentials/trusted publisher; no credentials are embedded here. Keep the original-store backup and rollback instructions in the release notes.

A stable `0.4.0` release should follow migration feedback and verified client sessions. Do not silently relabel an existing candidate artifact; build again with the stable version from its reviewed commit.

The implementation record and exact local results are in [verification](verification-v0.4.md).
