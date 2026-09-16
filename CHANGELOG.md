# Changelog

## 0.4.0rc2 — independent snapshots and a simpler core

- Return detached entries from every public operation, including nested source,
  metadata and history. Caller mutations cannot bypass revisions or re-embedding.
- Serialize operations on each store object with a reentrant thread lock. Readers
  cannot observe a partial write or a change that later rolls back.
- Build startup context from one snapshot so another writer's correction cannot
  mix a retired handoff with its replacement in the same response.
- Count literal tokenizer markers as ordinary text, preserving recall, statistics
  and rendered budgets for memories that quote those markers.
- Separate record rules (`models.py`) and JSON persistence (`persistence.py`)
  from memory operations. Preserve format 3, old IDs and migration support.
- Replace `dedup_threshold` with explicit `deduplicate=False`; make the writing
  agent keyword-only and document the Python compatibility changes.
- Default new stores to offline hashing. Preserve existing backend configuration,
  require semantic opt-in, and reject unknown configuration values.
- Share display tags, avoid copying full histories just to count candidates,
  and make evaluation output directories work on first use.
- Add API ownership and concurrent-read regressions, an architecture walkthrough,
  and project-local lint/format checks in CI.

See [migration](docs/migration-v0.4.md) and [verification](docs/verification-rc2.md).
The optional semantic-model and live-agent comparisons remain outside this
candidate's verified scope.

## 0.4.0rc1 — release candidate

- Generate UUID4 memory IDs and reject duplicate explicit IDs; retain legacy and caller-supplied identities.
- Add revisions, bounded history, source metadata and explicit supersession. Require inspected revisions for MCP/CLI edits; report actionable conflicts across MCP SDK versions.
- Replace age-based lock stealing with OS locks; reject stale snapshot saves; validate loaded data; roll back failed writes and embeddings.
- Persist complete embedding configuration and retain an existing store's backend by default.
- Use conservative exact deduplication so negation and changed numbers cannot disappear as similar text.
- Apply startup freshness rules and budget complete rendered context. Fix Claude prompt payload handling and Git worklog filename/baseline handling.
- Add inspect, doctor, recall explanations, explicit snapshot export and bounded Markdown import chunks.
- Add a two-process MCP handoff demo, 30 executable coding fixtures, and a provider-neutral evaluation adapter contract with honest missing-telemetry reporting.
- Rewrite onboarding, document compatibility/trust boundaries, and add package/OS/SDK checks.
- Handle quoted/Windows hook executables without duplicating hooks or deleting unrelated commands; retry transient Windows file-sharing conflicts during atomic replacement.
- Add an optional Codex CLI evaluation adapter with setup checks, completion-event usage reporting and process cleanup; extend CI to macOS.

**Compatibility:** format 3 writes, required MCP/CLI `expected_revision`, changed deduplication and rendered-budget behavior. Upgrade all shared writers together. See [migration](docs/migration-v0.4.md).

**Verified:** the full hashing suite passes on Linux, Windows and macOS; both MCP SDK majors and fresh wheel installation pass in [CI](https://github.com/Ninadnj/agent-memory-engine/actions/runs/34687492192). See the [verification record](docs/verification-codex-adapter.md) for counts and scope.

**Deferred:** the live-agent coding comparison is outside this candidate's release scope. Actual coding-client MCP sessions and the optional semantic model remain unverified. This candidate makes no coding-performance or whole-session token-savings claim. See the [release notes](docs/release.md).
