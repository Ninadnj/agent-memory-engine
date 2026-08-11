"""The optional sentence-transformers backend.

Skipped unless the `real` extra is installed, so CI stays offline and fast.
Run locally with:  pip install -e ".[real]" && pytest tests/test_sentence_transformers.py
"""

import pytest

pytest.importorskip(
    "sentence_transformers", reason='needs the optional "real" extra'
)

from agent_memory import MemoryStore, SentenceTransformerEmbedder  # noqa: E402


@pytest.fixture(scope="module")
def embedder():
    return SentenceTransformerEmbedder()


@pytest.fixture(scope="module")
def store(embedder):
    s = MemoryStore(embedder=embedder)
    s.write("Bookings are stored in UTC and converted in the UI layer.", type="decision")
    s.write("The customer chatbot uses Google Gemini in server/gemini-chat.ts.", type="decision")
    s.write("Admin routes are guarded by requireAdmin in server/auth.ts.", type="decision")
    s.write("Uploaded images are private unless they sit under the gallery/ prefix.", type="project")
    return s


def test_dimension_is_discovered_across_library_versions(embedder):
    """`get_sentence_embedding_dimension` was renamed in 5.x; both must work."""
    assert embedder.dim == 384
    assert embedder.embed(["hello"]).shape == (1, 384)


def test_vectors_are_unit_norm(embedder):
    vecs = embedder.embed(["one sentence", "another entirely different sentence"])
    norms = (vecs**2).sum(axis=1) ** 0.5
    assert all(abs(n - 1.0) < 1e-4 for n in norms)


@pytest.mark.parametrize(
    "paraphrase, expected",
    [
        ("a client saw the wrong hour for their appointment", "UTC"),
        ("make sure only administrators can reach the new settings page", "requireAdmin"),
        ("how do we restrict a route to administrators", "requireAdmin"),
        ("which AI model answers customer questions", "Gemini"),
    ],
)
def test_recall_survives_a_paraphrase(store, paraphrase, expected):
    """The whole reason this backend exists: no shared vocabulary."""
    hits = store.recall(paraphrase, k=1)
    assert expected in hits[0].entry.text


def test_a_genuinely_ambiguous_query_is_not_answered_confidently(store):
    """Two memories are about access control, and this query fits both equally.

    Both score ~0.16 — below the configured floor — so the honest outcome is
    nothing rather than a coin-flip presented to the agent as fact.
    """
    floor = SentenceTransformerEmbedder.recommended_min_score
    assert store.recall("only staff should be able to open this screen",
                        k=2, min_score=floor) == []


def test_configured_floor_keeps_real_matches_and_drops_junk(store):
    floor = SentenceTransformerEmbedder.recommended_min_score
    assert store.recall("how are timezones handled", k=3, min_score=floor)
    assert store.recall("best sourdough bread recipe for beginners", k=3, min_score=floor) == []


def test_a_store_written_by_this_backend_reloads(tmp_path, embedder):
    path = tmp_path / "st.json"
    first = MemoryStore(path=path, embedder=embedder)
    first.write("Calendar sync must never block a booking from saving.", type="decision")

    reopened = MemoryStore(path=path, embedder=embedder)
    assert reopened.stats()["count"] == 1
    hits = reopened.recall("what happens if the calendar integration fails", k=1)
    assert "Calendar sync" in hits[0].entry.text
