# v0.4.0rc2 verification

Review started from `6ffb103` (`v0.4.0rc1`). The implementation keeps format 3
and adds no runtime dependencies.

## Regressions demonstrated before implementation

The existing local suite passed: **278 passed, 3 skipped**, without the optional
MCP package or semantic backend.

The first API regression selection then produced **19 failures and 2 passes**.
It covered mutation through every entry-returning API, nested metadata/history,
stale snapshots acquiring newer revisions, explicit deduplication, deterministic
offline defaults and invalid backend configuration.

Two further tests demonstrated concurrent `get` and `recall` observing a change
while its save was pending, even though that save subsequently failed. Both
failed before the per-store thread lock was added. They now verify that readers
wait and receive the original, successfully stored state after rollback.

Historical test fixtures now create entries with a controlled write clock.
They exercise the real write path instead of relying on mutable returned objects.
The original freshness, conflict and persistence assertions are retained.

The pre-merge review added ten regression cases: **9 failed and 1 passed** before
the fixes. They cover another store superseding a handoff between startup
selection and recall, literal tokenizer markers, persisted-memory budgets, and
an MCP roundtrip. All ten now pass. The startup cases also verify that the next
call observes the correction rather than retaining an indefinitely stale snapshot.

## Local results (including the September 16 pre-merge fixes)

macOS, Python 3.12.1. The development environment uses NumPy 2.5.3,
pytest 9.1.1, MCP 2.2.0 and tiktoken 0.14.0.

| Check | Result |
| --- | --- |
| Full development suite | 330 passed; only the optional semantic-model module skipped |
| Minimal environment, NumPy 1.26.3, approximate token counting | 304 passed, 9 optional checks skipped (MCP, semantic model and exact-tokenizer cases) |
| Retrieval diagnostic with exact token counting | JSON and Markdown results match the committed baseline byte-for-byte |
| Executable task fixtures | All 30 broken implementations rejected and all 30 reference fixes accepted |
| MCP 2.2.0 demo | Two real stdio server processes share a store, reject a stale deletion and recall a replacement decision |
| MCP 1.30.0 compatibility | 89 focused API, MCP, task-runner and token tests passed, including the two-process demo |
| Python quickstart | Passed |
| Lint and source formatting | Passed using the repository's Ruff configuration |
| Wheel and source distribution | Built successfully as 0.4.0rc2 |
| Fresh wheel installation outside the checkout | Import, revision/reopen, MCP construction and CLI doctor passed |

The [release guide](release.md) contains the commands. The current platform CI
results are available in the pull request's checks; these local results describe
macOS only.

## Scope

The hashing retriever's measured relevance did not change. This work improves
state integrity, concurrency, maintainability and configuration predictability;
it does not establish improved LLM coding outcomes. The semantic model and actual
coding-client sessions remain unverified in this run. The 30 fixture checks
validate graders, not agent success rates.

The file-size limit remains a guard rather than a scale benchmark. File locks
cover cooperating processes on a local filesystem; thread locks cover concurrent
public operations on a single store instance. No network-share or distributed
storage guarantee is added.
