"""Evaluate targeted retrieval against full-context, random and no-memory arms.

The claim under test: *retrieving only the top-k relevant memories loads far
fewer context tokens than dumping every memory file, without dropping the facts
the agent needs.*

Two things make that claim easy to overstate, so both are measured here:

  * **The token saving is arithmetic, not skill.** Loading 3 of 14 memories
    costs ~21% of the tokens no matter which 3 you pick. The `random_k` arm
    picks k memories at random and therefore reports the *same* saving with
    useless recall. Only the gap in recall between `random_k` and the retriever
    is evidence that retrieval works.
  * **The default embedder is lexical, not semantic.** It matches shared
    wording, so a benchmark whose queries reuse the vocabulary of the answers
    flatters it. Every task carries a `paraphrase` that avoids that vocabulary,
    and both are reported side by side.

Arms per task:
  * no_memory      — control. The agent gets nothing (0 tokens, 0 recall).
  * full_context   — the Markdown-scaffold baseline: load every memory.
  * random_k       — control. k memories chosen at random: the token saving
                     without the retrieval.
  * retrieval      — this engine: load only the top-k by vector similarity.
  * budget_recall  — this engine under a hard token cap (recall never loads
                     more than --budget tokens, whatever the store size).

Also reported:
  * the same retrieval arm re-run on the paraphrased queries;
  * a scaling experiment that grows the store with distractor memories, to test
    whether recall cost really stays flat as memory accumulates;
  * a sweep of the `min_score` relevance floor used by the CLI and MCP server.

Run:  python eval/run_eval.py            (writes eval/results.md + results.json)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Make `src/` importable when run directly, without an install step.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_memory import HashingEmbedder, MemoryStore, count_tokens  # noqa: E402
from agent_memory.tokens import using_exact_tokenizer  # noqa: E402

ARM_LABELS = {
    "no_memory": "No memory (control)",
    "full_context": "Full context (baseline — load every memory)",
    "random_k": "Random k (control — the saving without the retrieval)",
    "retrieval": "Targeted retrieval (this engine)",
    "budget_recall": "Budget recall (hard token cap)",
}


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text())


def _prf(retrieved_ids: list[str], gold_ids: set[str]) -> tuple[float, float]:
    if not retrieved_ids:
        return 0.0, 0.0
    hit = len(set(retrieved_ids) & gold_ids)
    precision = hit / len(retrieved_ids)
    recall = hit / len(gold_ids) if gold_ids else 0.0
    return precision, recall


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def build_store(memories: list[dict]) -> MemoryStore:
    # Explicit embedder: the published numbers are the offline one's, and we do
    # not want them to change silently on a machine with sentence-transformers.
    store = MemoryStore(embedder=HashingEmbedder())
    for m in memories:
        store.write(m["text"], type=m["type"], id=m["id"])
    return store


def evaluate(dataset: dict, k: int = 3, budget: int = 120, seed: int = 0) -> dict:
    memories = dataset["memories"]
    store = build_store(memories)
    rng = random.Random(seed)

    all_ids = [m["id"] for m in memories]
    full_tokens = sum(count_tokens(m["text"]) for m in memories)
    by_id = {m["id"]: m for m in memories}

    arms = {name: {"tokens": [], "precision": [], "recall": []} for name in ARM_LABELS}
    per_task = []

    for task in dataset["tasks"]:
        gold = set(task["relevant_ids"])

        def record(name: str, ids: list[str], tokens: int) -> tuple[float, float]:
            p, r = _prf(ids, gold)
            arms[name]["tokens"].append(tokens)
            arms[name]["precision"].append(p)
            arms[name]["recall"].append(r)
            return p, r

        record("no_memory", [], 0)
        record("full_context", all_ids, full_tokens)

        # Control: the token saving with none of the retrieval.
        picked = rng.sample(all_ids, min(k, len(all_ids)))
        record("random_k", picked, sum(count_tokens(by_id[i]["text"]) for i in picked))

        hits = store.recall(task["query"], k=k)
        retrieved_ids = [h.entry.id for h in hits]
        tokens = sum(h.entry.tokens for h in hits)
        _, r = record("retrieval", retrieved_ids, tokens)

        bhits = store.recall(task["query"], k=k, budget_tokens=budget)
        btokens = sum(h.entry.tokens for h in bhits)
        assert btokens <= budget, "budget packing must never overflow"
        record("budget_recall", [h.entry.id for h in bhits], btokens)

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
        "seed": seed,
        "exact_tokenizer": using_exact_tokenizer(),
        "embedder": store.stats()["embedder"],
        "n_memories": len(memories),
        "n_tasks": len(dataset["tasks"]),
        "summary": summary,
        "per_task": per_task,
        "phrasing": phrasing_gap(dataset, k=k),
        "scaling": scaling(dataset, k=k, budget=budget),
        "min_score_sweep": min_score_sweep(dataset, k=k),
    }


def phrasing_gap(dataset: dict, k: int = 3) -> dict:
    """Recall on the developer phrasing vs an outsider's paraphrase.

    The default embedder matches shared wording. This is the number that says
    how much of the headline recall comes from the benchmark's phrasing.
    """
    store = build_store(dataset["memories"])
    out = {}
    for field in ("query", "paraphrase"):
        recalls, overlaps = [], []
        for task in dataset["tasks"]:
            if field not in task:
                continue
            gold = set(task["relevant_ids"])
            hits = store.recall(task[field], k=k)
            _, r = _prf([h.entry.id for h in hits], gold)
            recalls.append(r)
            overlaps.append(_word_overlap(task[field], dataset["memories"], gold))
        out[field] = {
            "recall": round(mean(recalls), 3),
            "avg_query_word_overlap_with_gold": round(mean(overlaps), 3),
        }
    return out


def _word_overlap(query: str, memories: list[dict], gold: set[str]) -> float:
    """Share of the query's content words that also appear in its gold memories."""
    import re

    stop = set(
        "a an the to is are of in on for and it that this with be as at by from "
        "how do we make sure so they their there these it's its can should".split()
    )
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in stop}
    if not words:
        return 0.0
    gold_text = " ".join(m["text"] for m in memories if m["id"] in gold).lower()
    gold_words = set(re.findall(r"[a-z0-9]+", gold_text))
    return len(words & gold_words) / len(words)


def scaling(dataset: dict, k: int = 3, budget: int = 120) -> list[dict]:
    """Grow the store with distractors and re-measure.

    "The cost of memory stays fixed as the store grows" is a claim about a store
    that grows, so the store has to actually grow for it to mean anything.
    """
    distractors = dataset.get("distractors", [])
    rows = []
    for extra in (0, len(distractors) // 2, len(distractors)):
        memories = list(dataset["memories"])
        memories += [
            {"id": f"dis_{i:04d}", "type": d["type"], "text": d["text"]}
            for i, d in enumerate(distractors[:extra])
        ]
        store = build_store(memories)
        full_tokens = sum(count_tokens(m["text"]) for m in memories)
        recalls, tokens = [], []
        for task in dataset["tasks"]:
            gold = set(task["relevant_ids"])
            hits = store.recall(task["query"], k=k, budget_tokens=budget)
            _, r = _prf([h.entry.id for h in hits], gold)
            recalls.append(r)
            tokens.append(sum(h.entry.tokens for h in hits))
        rows.append(
            {
                "n_memories": len(memories),
                "full_context_tokens": full_tokens,
                "budget_recall_tokens": round(mean(tokens), 1),
                "budget_recall": round(mean(recalls), 3),
            }
        )
    return rows


def min_score_sweep(dataset: dict, k: int = 3) -> list[dict]:
    """Calibrate the relevance floor used by the CLI and MCP server.

    Labelled recall on real queries must not drop, while queries about things
    the store knows nothing about should return nothing at all.
    """
    store = build_store(dataset["memories"])
    off_topic = [
        "how do I bake sourdough bread at home",
        "what is the capital of Peru",
        "my cat will not eat her food",
        "best hiking boots for winter walking",
    ]
    rows = []
    for threshold in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        recalls = []
        for task in dataset["tasks"]:
            gold = set(task["relevant_ids"])
            hits = store.recall(task["query"], k=k, min_score=threshold)
            _, r = _prf([h.entry.id for h in hits], gold)
            recalls.append(r)
        junk = sum(len(store.recall(q, k=k, min_score=threshold)) for q in off_topic)
        rows.append(
            {
                "min_score": threshold,
                "labelled_recall": round(mean(recalls), 3),
                "off_topic_memories_returned": junk,
                "off_topic_max": k * len(off_topic),
            }
        )
    return rows


def render_markdown(results: dict) -> str:
    s = results["summary"]
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
    for name, label in ARM_LABELS.items():
        m = s[name]
        if name == "budget_recall":
            label = f"Budget recall (≤ {results['budget']} tokens)"
        lines.append(
            f"| {label} | {m['avg_context_tokens']:.0f} | {m['recall']:.2f} "
            f"| {m['precision']:.2f} | {m['token_reduction_vs_baseline']:.0f}% |"
        )

    ret, rnd = s["retrieval"], s["random_k"]
    lines += [
        "",
        "**Read the random arm first.** It loads the same number of memories as "
        f"the engine, so it reports the same ~{rnd['token_reduction_vs_baseline']:.0f}% "
        "token saving — at "
        f"**{rnd['recall']:.2f}** recall against the engine's **{ret['recall']:.2f}**. "
        "The saving is arithmetic (k memories out of n); only the recall gap is "
        "evidence that retrieval does anything.",
        "",
        "Precision is reported for completeness, but the baseline's "
        f"**{s['full_context']['precision']:.2f}** is just "
        "`|relevant| / |store|` — an artefact of loading everything, not a "
        "meaningful comparison.",
        "",
        "## Does it survive a rephrase?",
        "",
        "| Query phrasing | Word overlap with gold memories | Recall |",
        "| --- | ---: | ---: |",
    ]
    ph = results["phrasing"]
    for field, label in (
        ("query", "Developer phrasing (as labelled)"),
        ("paraphrase", "Outsider paraphrase (vocabulary avoided)"),
    ):
        row = ph.get(field)
        if row:
            lines.append(
                f"| {label} | {row['avg_query_word_overlap_with_gold']:.0%} | "
                f"{row['recall']:.2f} |"
            )
    lines += [
        "",
        "The default embedder is feature hashing — lexical, not semantic. When the "
        "query stops sharing words with the memory, recall drops by "
        f"**{(ph['query']['recall'] - ph['paraphrase']['recall']):.2f}**. That gap is "
        "the honest limit of the offline default, and the reason the optional "
        "`sentence-transformers` backend exists.",
        "",
        "## Does the cost stay flat as the store grows?",
        "",
        "| Memories in store | Full context tokens | Budget recall tokens | Budget recall |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in results["scaling"]:
        lines.append(
            f"| {row['n_memories']} | {row['full_context_tokens']} | "
            f"{row['budget_recall_tokens']:.0f} | {row['budget_recall']:.2f} |"
        )
    lines += [
        "",
        "Distractor memories are added from the same project. The baseline grows "
        "linearly; the budgeted arm does not. Recall is measured against the same "
        "labels throughout, so any drop is real interference from the added memories.",
        "",
        "## Relevance floor (`min_score`)",
        "",
        "| min_score | Labelled recall | Off-topic memories returned |",
        "| ---: | ---: | ---: |",
    ]
    for row in results["min_score_sweep"]:
        lines.append(
            f"| {row['min_score']:.2f} | {row['labelled_recall']:.2f} | "
            f"{row['off_topic_memories_returned']}/{row['off_topic_max']} |"
        )
    lines += [
        "",
        "Four deliberately off-topic queries stand in for a task the store knows "
        "nothing about. Without a floor the engine returns k memories anyway, and "
        "the MCP layer strips scores, so the agent cannot tell. `HashingEmbedder."
        "recommended_min_score` is set from this sweep.",
        "",
        "## Honest limits",
        "",
        f"- {results['n_tasks']} tasks and "
        f"{sum(len(t['gold']) for t in results['per_task'])} gold labels, all "
        "hand-written by the author. This is an engineering check, not a "
        "production-scale claim; one retrieval either way moves recall by ~0.07.",
        "- The labels, the queries and the retriever all come from the same person, "
        "which is exactly the setup that flatters a retriever. The paraphrase arm "
        "exists to push back on that.",
        "- Numbers use the offline hashing embedder so they are stable in CI. The "
        "`sentence-transformers` backend is not measured here.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the memory recall evaluation.")
    parser.add_argument("--k", type=int, default=3, help="top-k memories to recall")
    parser.add_argument(
        "--budget", type=int, default=120, help="token cap for the budget arm"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for the random-k control arm"
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path(__file__).parent / "dataset.json"
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    results = evaluate(dataset, k=args.k, budget=args.budget, seed=args.seed)

    md = render_markdown(results)
    (args.out_dir / "results.md").write_text(md + "\n")
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(md)
    if not results["exact_tokenizer"]:
        print(
            "\n[warning] tiktoken is not installed, so token counts are approximate "
            'and will not match the published table. Install it with: pip install -e ".[dev]"',
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
