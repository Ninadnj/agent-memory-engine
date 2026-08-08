import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import evaluate  # noqa: E402


def _dataset():
    return json.loads((ROOT / "eval" / "dataset.json").read_text())


def _results(**kw):
    return evaluate(_dataset(), **kw)


def test_retrieval_uses_fewer_tokens_than_the_full_context_baseline():
    r = _results(k=3)
    assert (
        r["summary"]["retrieval"]["avg_context_tokens"]
        < r["summary"]["full_context"]["avg_context_tokens"]
    )


def test_the_token_saving_is_not_evidence_on_its_own():
    """The random control must show the same saving as the engine.

    If this ever fails, the token-reduction number has started measuring
    something other than "we loaded k of n memories" and the README's framing
    needs revisiting.
    """
    r = _results(k=3)
    engine = r["summary"]["retrieval"]["token_reduction_vs_baseline"]
    random_arm = r["summary"]["random_k"]["token_reduction_vs_baseline"]
    assert abs(engine - random_arm) < 15


def test_retrieval_beats_the_random_control_by_a_wide_margin():
    """This — not the token saving — is the claim worth making."""
    r = _results(k=3)
    assert r["summary"]["retrieval"]["recall"] - r["summary"]["random_k"]["recall"] > 0.5


def test_recall_is_high_enough_to_be_useful():
    assert _results(k=3)["summary"]["retrieval"]["recall"] >= 0.8


def test_budget_arm_respects_cap_and_stays_useful():
    r = _results(k=3, budget=120)
    bud = r["summary"]["budget_recall"]
    assert bud["avg_context_tokens"] <= 120
    assert bud["recall"] >= 0.7


def test_paraphrase_arm_is_reported_and_is_the_weaker_number():
    """The lexical default loses recall when wording is not shared.

    The point of the assertion is that the gap stays *measured and published*,
    not that it stays small.
    """
    ph = _results(k=3)["phrasing"]
    assert ph["paraphrase"]["recall"] <= ph["query"]["recall"]
    assert ph["paraphrase"]["avg_query_word_overlap_with_gold"] < (
        ph["query"]["avg_query_word_overlap_with_gold"]
    )


def test_budget_cost_stays_flat_while_the_store_grows():
    rows = _results(k=3, budget=120)["scaling"]
    assert len(rows) >= 3
    assert rows[-1]["n_memories"] > rows[0]["n_memories"] * 2
    # The baseline grows with the store; the budgeted arm does not.
    assert rows[-1]["full_context_tokens"] > 2 * rows[0]["full_context_tokens"]
    assert rows[-1]["budget_recall_tokens"] <= 120
    assert rows[-1]["budget_recall"] >= 0.7


def test_min_score_sweep_supports_the_configured_floor():
    from agent_memory import HashingEmbedder

    rows = {row["min_score"]: row for row in _results(k=3)["min_score_sweep"]}
    floor = HashingEmbedder.recommended_min_score
    assert floor in rows, "the shipped floor must appear in the published sweep"
    assert rows[floor]["labelled_recall"] == rows[0.0]["labelled_recall"]
    assert rows[floor]["off_topic_memories_returned"] < rows[0.0]["off_topic_memories_returned"]


def test_random_arm_is_deterministic_for_a_given_seed():
    assert _results(k=3, seed=7)["summary"]["random_k"] == (
        _results(k=3, seed=7)["summary"]["random_k"]
    )
