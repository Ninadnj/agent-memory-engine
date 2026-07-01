"""Evaluate semantic recall against full-context and no-memory baselines.

The claim under test: *retrieving only the top-k relevant memories loads far
fewer context tokens than dumping every memory file, without dropping the facts
the agent needs.* This harness measures exactly that on a labeled benchmark and
prints a results table.

Four arms per task:
  * no_memory      — control. The agent gets nothing (0 tokens, 0 recall).
  * full_context   — the Markdown-scaffold baseline: load every memory.
  * semantic_recall — this engine: load only the top-k by vector similarity.
  * budget_recall  — this engine under a hard token cap (recall never loads
                     more than --budget tokens, whatever the store size).

Run:  python eval/run_eval.py            (writes eval/results.md + results.json)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src/` importable when run directly, without an install step.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_memory import MemoryStore, count_tokens  # noqa: E402
from agent_memory.tokens import using_exact_tokenizer  # noqa: E402


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text())


def _prf(retrieved_ids: list[str], gold_ids: set[str]) -> tuple[float, float]:
    if not retrieved_ids:
        return 0.0, 0.0
    hit = len(set(retrieved_ids) & gold_ids)
    precision = hit / len(retrieved_ids)
    recall = hit / len(gold_ids) if gold_ids else 0.0
    return precision, recall


def evaluate(dataset: dict, k: int = 3, budget: int = 120) -> dict:
    store = MemoryStore()
    for m in dataset["memories"]:
        store.write(m["text"], type=m["type"], id=m["id"])

    all_ids = [m["id"] for m in dataset["memories"]]
    full_tokens = sum(count_tokens(m["text"]) for m in dataset["memories"])

    arms = {
        "no_memory": {"tokens": [], "precision": [], "recall": []},
        "full_context": {"tokens": [], "precision": [], "recall": []},
        "semantic_recall": {"tokens": [], "precision": [], "recall": []},
        "budget_recall": {"tokens": [], "precision": [], "recall": []},
    }
    per_task = []

    for task in dataset["tasks"]:
        gold = set(task["relevant_ids"])

        # no_memory
        arms["no_memory"]["tokens"].append(0)
        arms["no_memory"]["precision"].append(0.0)
        arms["no_memory"]["recall"].append(0.0)

        # full_context
        p, r = _prf(all_ids, gold)
        arms["full_context"]["tokens"].append(full_tokens)
        arms["full_context"]["precision"].append(p)
        arms["full_context"]["recall"].append(r)

        # semantic_recall
        hits = store.recall(task["query"], k=k)
        retrieved_ids = [h.entry.id for h in hits]
        tokens = sum(h.entry.tokens for h in hits)
        p, r = _prf(retrieved_ids, gold)
        arms["semantic_recall"]["tokens"].append(tokens)
        arms["semantic_recall"]["precision"].append(p)
        arms["semantic_recall"]["recall"].append(r)

        # budget_recall — same engine, hard token cap
        bhits = store.recall(task["query"], k=k, budget_tokens=budget)
        bids = [h.entry.id for h in bhits]
        btokens = sum(h.entry.tokens for h in bhits)
        assert btokens <= budget, "budget packing must never overflow"
        bp, br = _prf(bids, gold)
        arms["budget_recall"]["tokens"].append(btokens)
        arms["budget_recall"]["precision"].append(bp)
        arms["budget_recall"]["recall"].append(br)

        per_task.append(
            {
                "task": task["id"],
                "query": task["query"],
                "gold": sorted(gold),
                "retrieved": retrieved_ids,
                "recall": round(r, 3),
                "tokens": tokens,
            }
        )

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    summary = {}
    baseline_tokens = mean(arms["full_context"]["tokens"])
    for name, m in arms.items():
        avg_tokens = mean(m["tokens"])
        summary[name] = {
            "avg_context_tokens": round(avg_tokens, 1),
            "precision": round(mean(m["precision"]), 3),
            "recall": round(mean(m["recall"]), 3),
            "token_reduction_vs_baseline": (
                round(100 * (1 - avg_tokens / baseline_tokens), 1)
                if baseline_tokens
                else 0.0
            ),
        }

    return {
        "k": k,
        "budget": budget,
        "exact_tokenizer": using_exact_tokenizer(),
        "embedder": store.stats()["embedder"],
        "n_memories": len(dataset["memories"]),
        "n_tasks": len(dataset["tasks"]),
        "summary": summary,
        "per_task": per_task,
    }


def render_markdown(results: dict) -> str:
    s = results["summary"]
    rows = [
        ("No memory (control)", s["no_memory"]),
        ("Full context (baseline)", s["full_context"]),
        ("Semantic recall (this engine)", s["semantic_recall"]),
        (f"Budget recall (≤ {results['budget']} tokens)", s["budget_recall"]),
    ]
    lines = [
        "# Evaluation results",
        "",
        f"- Benchmark: **{results['n_memories']} memories**, "
        f"**{results['n_tasks']} tasks**, top-k = **{results['k']}**, "
        f"budget = **{results['budget']} tokens**",
        f"- Embedder: `{results['embedder']}` · "
        f"exact tokenizer: `{results['exact_tokenizer']}`",
        "",
        "| Arm | Avg context tokens | Recall | Precision | Tokens saved vs baseline |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, m in rows:
        lines.append(
            f"| {label} | {m['avg_context_tokens']:.0f} | {m['recall']:.2f} "
            f"| {m['precision']:.2f} | {m['token_reduction_vs_baseline']:.0f}% |"
        )
    sem = s["semantic_recall"]
    bud = s["budget_recall"]
    lines += [
        "",
        f"**Headline:** semantic recall retrieves **{sem['recall']:.0%}** of the "
        f"relevant memories while loading **{sem['token_reduction_vs_baseline']:.0f}% "
        f"fewer context tokens** than dumping every memory file, and at "
        f"**{sem['precision']:.0%}** precision vs "
        f"**{s['full_context']['precision']:.0%}** for the baseline. Under a hard "
        f"**{results['budget']}-token cap** recall is still **{bud['recall']:.0%}** — "
        f"the cost of memory stays fixed as the store grows.",
        "",
        "_Numbers above use the offline hashing embedder so they are byte-stable "
        "in CI. Installing `sentence-transformers` (the `real` extra) raises recall "
        "further on paraphrased queries._",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the memory recall evaluation.")
    parser.add_argument("--k", type=int, default=3, help="top-k memories to recall")
    parser.add_argument(
        "--budget", type=int, default=120, help="token cap for the budget arm"
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path(__file__).parent / "dataset.json"
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    results = evaluate(dataset, k=args.k, budget=args.budget)

    md = render_markdown(results)
    (args.out_dir / "results.md").write_text(md + "\n")
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(md)


if __name__ == "__main__":
    main()
