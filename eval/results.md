# Evaluation results

- Benchmark: **14 memories**, **7 tasks**, top-k = **3**, budget = **120 tokens**
- Embedder: `HashingEmbedder` · exact tokenizer: `True`

| Arm | Avg context tokens | Recall | Precision | Tokens saved vs baseline |
| --- | ---: | ---: | ---: | ---: |
| No memory (control) | 0 | 0.00 | 0.00 | 100% |
| Full context (baseline — load every memory) | 424 | 1.00 | 0.10 | 0% |
| Random k (control — the saving without the retrieval) | 93 | 0.07 | 0.05 | 78% |
| Targeted retrieval (this engine) | 93 | 0.93 | 0.43 | 78% |
| Budget recall (≤ 120 tokens) | 93 | 0.93 | 0.43 | 78% |

**Read the random arm first.** It loads the same number of memories as the engine, so it reports the same ~78% token saving — at **0.07** recall against the engine's **0.93**. The saving is arithmetic (k memories out of n); only the recall gap is evidence that retrieval does anything.

Precision is reported for completeness, but the baseline's **0.10** is just `|relevant| / |store|` — an artefact of loading everything, not a meaningful comparison.

## Does it survive a rephrase?

| Query phrasing | Word overlap with gold memories | Recall |
| --- | ---: | ---: |
| Developer phrasing (as labelled) | 40% | 0.93 |
| Outsider paraphrase (vocabulary avoided) | 3% | 0.43 |

The default embedder is feature hashing — lexical, not semantic. When the query stops sharing words with the memory, recall drops by **0.50**. That gap is the honest limit of the offline default, and the reason the optional `sentence-transformers` backend exists.

## Does the cost stay flat as the store grows?

| Memories in store | Full context tokens | Budget recall tokens | Budget recall |
| ---: | ---: | ---: | ---: |
| 14 | 424 | 93 | 0.93 |
| 34 | 805 | 87 | 0.93 |
| 54 | 1159 | 82 | 0.93 |

Distractor memories are added from the same project. The baseline grows linearly; the budgeted arm does not. Recall is measured against the same labels throughout, so any drop is real interference from the added memories.

## Relevance floor (`min_score`)

| min_score | Labelled recall | Off-topic memories returned |
| ---: | ---: | ---: |
| 0.00 | 0.93 | 12/12 |
| 0.05 | 0.93 | 12/12 |
| 0.10 | 0.93 | 10/12 |
| 0.15 | 0.93 | 6/12 |
| 0.20 | 0.93 | 2/12 |
| 0.25 | 0.93 | 2/12 |

Four deliberately off-topic queries stand in for a task the store knows nothing about. Without a floor the engine returns k memories anyway, and the MCP layer strips scores, so the agent cannot tell. `HashingEmbedder.recommended_min_score` is set from this sweep.

## Honest limits

- 7 tasks and 10 gold labels, all hand-written by the author. This is an engineering check, not a production-scale claim; one retrieval either way moves recall by ~0.07.
- The labels, the queries and the retriever all come from the same person, which is exactly the setup that flatters a retriever. The paraphrase arm exists to push back on that.
- Numbers use the offline hashing embedder so they are stable in CI. The `sentence-transformers` backend is not measured here.
