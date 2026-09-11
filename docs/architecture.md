# Architecture

The engine keeps a small in-memory list of entries and a NumPy embedding matrix, persisted as JSON with base64 vectors. CLI, MCP and hooks share the same store implementation. There is no separate database service.

## Identity and writes

Every generated identity is `mem_` plus `uuid.uuid4().hex`. The store checks current IDs as a collision guard; explicit caller IDs are preserved and duplicates rejected before mutation. IDs are never recomputed from the remaining records.

A mutation acquires an OS advisory lock, reloads a changed file, validates the caller's revision, computes embeddings, applies the change, then atomically replaces the JSON. Failed embedding or persistence operations restore the local entry and vector snapshots. The file stamp includes mtime, size and inode. Reads observe replacement on their next call. Deleting the backing file clears the next snapshot rather than resurrecting it.

The JSON contains an embedding configuration: backend, dimensions, model/revision when applicable, and normalization or feature-version information. Mismatches cause re-embedding. A floating remote model name cannot establish that downloaded weights stayed identical; use an immutable revision when that matters.

## Corrections and evidence

An entry has original creation provenance, current revision, last update provenance, optional source references, and up to 20 prior snapshots. A replacement marks the old record superseded and records the replacement ID. Source checks only compare a project-relative file against a Git commit; they do not evaluate text or execute stored instructions.

Expected revisions provide optimistic concurrency control for one record. They are required at the agent-facing edit boundary and optional for legacy Python calls. This is not a multi-record transaction API or tamper-proof audit system. Python entry objects remain mutable for backward compatibility; use the mutation methods when revision tracking matters.

## Retrieval and rendering

Recall ranks active records by embedding dot product, with age decay for state, handoff and worklog types. Durable facts and decisions do not decay automatically. Startup also imposes hard age limits on handoffs and worklogs so a low relevance floor cannot revive months-old startup instructions.

The store-level budget is the sum of memory body tokens. The presentation layer packs complete rendered blocks and counts labels, IDs, dates, source references and separators. It skips an oversized block and tries the next candidate. Protocol envelopes, tool descriptions and client-added formatting remain outside that text budget.

Hashing is lexical: synonyms and paraphrases can be missed. Optional sentence-transformers provides a semantic backend, but still needs evaluation for the target workload. Similarity is never used to merge memories; a changed number or negation must not silently disappear.

## Tradeoffs

Writes serialize through one lock and rewrite the whole file. Recall is a linear matrix scan. These choices keep deployment and recovery understandable for small local stores. Benchmark actual workload size and contention before introducing a vector database, distributed locks, background workers or automatic summarization.
