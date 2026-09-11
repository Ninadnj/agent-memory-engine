# Changelog

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

**Compatibility:** format 3 writes, required MCP/CLI `expected_revision`, changed deduplication and rendered-budget behavior. Upgrade all shared writers together. See [migration](docs/migration-v0.4.md).

**Evidence still needed:** live client sessions, a live-agent coding comparison and target-platform results beyond the local checks recorded in [release notes](docs/release.md). This candidate makes no new coding-performance claim.
