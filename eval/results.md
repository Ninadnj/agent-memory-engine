# Evaluation results

- Benchmark: **14 memories**, **7 tasks**, top-k = **3**, budget = **120 tokens**
- Embedder: `HashingEmbedder` · exact tokenizer: `False`

| Arm | Avg context tokens | Recall | Precision | Tokens saved vs baseline |
| --- | ---: | ---: | ---: | ---: |
| No memory (control) | 0 | 0.00 | 0.00 | 100% |
| Full context (baseline) | 550 | 1.00 | 0.10 | 0% |
| Semantic recall (this engine) | 120 | 0.93 | 0.43 | 78% |
| Budget recall (≤ 120 tokens) | 113 | 0.93 | 0.43 | 80% |

**Headline:** semantic recall retrieves **93%** of the relevant memories while loading **78% fewer context tokens** than dumping every memory file, and at **43%** precision vs **10%** for the baseline. Under a hard **120-token cap** recall is still **93%** — the cost of memory stays fixed as the store grows.

_Numbers above use the offline hashing embedder so they are byte-stable in CI. Installing `sentence-transformers` (the `real` extra) raises recall further on paraphrased queries._
