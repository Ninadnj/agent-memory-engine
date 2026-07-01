import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import evaluate  # noqa: E402


def _dataset():
    return json.loads((ROOT / "eval" / "dataset.json").read_text())


def test_semantic_recall_uses_fewer_tokens_than_baseline():
    r = evaluate(_dataset(), k=3)
    sem = r["summary"]["semantic_recall"]
    base = r["summary"]["full_context"]
    assert sem["avg_context_tokens"] < base["avg_context_tokens"]
    assert sem["token_reduction_vs_baseline"] >= 50


def test_recall_is_high_enough_to_be_useful():
    r = evaluate(_dataset(), k=3)
    assert r["summary"]["semantic_recall"]["recall"] >= 0.8


def test_semantic_recall_beats_baseline_precision():
    r = evaluate(_dataset(), k=3)
    sem = r["summary"]["semantic_recall"]
    base = r["summary"]["full_context"]
    assert sem["precision"] > base["precision"]


def test_budget_arm_respects_cap_and_stays_useful():
    r = evaluate(_dataset(), k=3, budget=120)
    bud = r["summary"]["budget_recall"]
    assert bud["avg_context_tokens"] <= 120
    assert bud["recall"] >= 0.7
