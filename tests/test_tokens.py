"""Memory content is ordinary text, even when it contains tokenizer markers."""

import pytest

from agent_memory import HashingEmbedder, MemoryStore, tokens
from agent_memory.rendering import boot_context, recall_context


@pytest.fixture
def exact_encoder(monkeypatch):
    tiktoken = pytest.importorskip("tiktoken", reason="needs the exact tokenizer")
    encoder = tiktoken.get_encoding("cl100k_base")
    monkeypatch.setattr(tokens, "_encoder", lambda: encoder)
    return encoder


@pytest.mark.parametrize(
    "marker",
    [
        "<|endoftext|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|endofprompt|>",
    ],
)
def test_special_markers_are_counted_as_literal_text(exact_encoder, marker):
    text = f"The parser must preserve the literal {marker} marker."
    assert tokens.count_tokens(text) == len(exact_encoder.encode_ordinary(text))


def test_literal_markers_work_in_persisted_memory_and_budgeted_context(
    tmp_path, exact_encoder
):
    path = tmp_path / "memory.json"
    store = MemoryStore(path, HashingEmbedder())
    handoff = store.write(
        "Next: verify the parser preserves <|endoftext|>.", type="handoff"
    )
    fact = store.write(
        "The parser preserves <|fim_prefix|> and <|fim_suffix|> as literal text.",
        source={"path": "fixtures/<|endofprompt|>.txt"},
    )
    reopened = MemoryStore(path, HashingEmbedder())
    expected = sum(
        len(exact_encoder.encode_ordinary(entry.text)) for entry in (handoff, fact)
    )
    assert reopened.stats()["total_tokens"] == expected
    hits = reopened.recall("parser", budget_tokens=expected)
    assert {hit.entry.id for hit in hits} == {handoff.id, fact.id}
    for render in (boot_context, recall_context):
        full = render(reopened, "parser", budget=1000)
        assert handoff.text in full and fact.text in full
        size = len(exact_encoder.encode_ordinary(full))
        assert render(reopened, "parser", budget=size) == full
        smaller = render(reopened, "parser", budget=size - 1)
        assert len(exact_encoder.encode_ordinary(smaller)) <= size - 1
        assert render(reopened, "parser", budget=0) == ""


def test_approximate_counting_also_accepts_literal_markers(monkeypatch):
    monkeypatch.setattr(tokens, "_encoder", lambda: None)
    text = "The parser preserves <|endoftext|>."
    assert tokens.count_tokens(text) > 0
    assert tokens.count_tokens("") == 0
