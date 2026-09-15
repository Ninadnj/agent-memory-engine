# Upgrading to v0.4

## From rc1 to rc2

The on-disk format remains **3**. Existing IDs, vectors, sources and histories
are preserved; this cleanup does not require data conversion. Restart clients
after upgrading so they use the same version.

- Python results are now **detached snapshots**. Assigning `entry.text`, changing
  nested dictionaries, or calling `save()` afterward no longer edits stored
  memory. Use `update` or `supersede` with the revision you read.
- `dedup_threshold` is removed. Exact deduplication remains the default; replace
  `dedup_threshold=2` with `deduplicate=False`. Semantic deduplication is not
  supported. `agent` is now keyword-only in `write` and `write_with_status`, so
  an old positional threshold fails instead of being mistaken for an agent name.
- New stores default to hashing, even when sentence-transformers is installed.
  Select `AGENT_MEMORY_EMBEDDER=sentence-transformers` for semantic retrieval.
  An unset value or `auto` still restores an existing store's saved backend.
  Unknown configuration values now fail instead of silently choosing a backend.
- Public imports from `agent_memory` remain available. Internal record and
  persistence helpers moved into `models.py` and `persistence.py`.

## From v0.3 or earlier

v0.4 reads existing format 1 and 2 stores and writes format 3 on the next mutation. Reading alone does not rewrite the file. All existing IDs, texts, creation dates, authors and metadata are retained. New UUID IDs are opaque: never infer sequence or ordering from them.

1. Stop **all** processes writing the shared store. v0.3 uses a different lock protocol and does not understand revisions. Mixed-version writers are unsupported.
2. Make a byte-for-byte copy of the original JSON file before opening it with new software. Keep this original if a downgrade may be needed.
3. Install v0.4 for every client. Run `agent-memory --path /your/store.json doctor` and inspect a few known IDs.
4. Restart clients so they discover the revised MCP schemas. Perform a write, correction and recall using a temporary test entry first.

New fields default to revision 1, active status, no update timestamp, empty source and empty history. Legacy stores without embedding configuration are re-embedded in memory with the selected backend. A subsequent write persists that configuration; no ID migration is needed.

## API changes

| Operation | v0.4 behavior |
| --- | --- |
| MCP update, forget, supersede | `expected_revision` required; conflicts identify the current revision |
| CLI update, forget, supersede | `--expected-revision N` required |
| Python update and forget | Existing calls remain valid; pass `expected_revision` to reject stale edits |
| Python save on an attached store | Rejects a stale snapshot instead of overwriting another writer |
| Python export | Explicit snapshot to a different path; existing target rejected unless `overwrite=True` |
| Deduplication | Whitespace-normalized, case-sensitive exact text, same type and source; explicit IDs bypass deduplication |
| Deduplication option | rc2 uses `deduplicate=False` to retain repeats. The old `dedup_threshold` argument is removed |
| Recall and boot in MCP/CLI | Budgets count complete rendered text; fewer memories may fit than before |
| Startup | Expired handoffs/worklogs are omitted; corrections refresh the effective age |

Use `memory_get` or `agent-memory inspect` before editing. A no-op update does not increment a revision or refresh its age. For a substantive correction, the original creation date remains and `updated_at` records freshness. The most recent 20 prior revisions are retained, not an unlimited audit ledger.

Supersession creates a new memory and links the retired one atomically. The retired record remains available to inspection and listing, but normal recall excludes it. `forget` permanently deletes a record and its revision history from the current store; copies and backups are unaffected.

## Persistence and limits

The sibling `.guard` file is a persistent OS lock target. Its existence does not mean the store is locked; do not delete it while clients are active. Kernel locks are released when a process exits. This covers cooperating processes on one local filesystem, not network shares or older writers.

Writes flush and fsync the temporary file before atomic replacement. This prevents partial JSON from normal interrupted writes; it is not a universal hardware power-loss guarantee. Invalid files are rejected and left untouched. New inputs are limited to 20,000 characters per memory/query and 16 KiB each for metadata/source. Existing stored text and metadata outside these input limits remain readable and editable; IDs are never truncated. History retains 20 snapshots. The 128 MiB store-file limit requires larger stores to be reduced on a backup before migration; files are never silently truncated.

Format 3 is not safe to edit with v0.3. For rollback, stop new writers and restore the original backup with the matching old software. A v0.4 export is format 3, not a downgrade converter.
