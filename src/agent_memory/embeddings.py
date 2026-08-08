"""Pluggable text embedders.

The engine never imports a heavy model directly. It depends on the small
``Embedder`` protocol below, which lets us swap implementations:

* ``HashingEmbedder`` — deterministic, dependency-free, offline. Used by
  default so tests and CI run anywhere with no model download. It is a
  feature-hashing bag of word/char n-grams: good enough for lexical recall,
  and fully reproducible.
* ``SentenceTransformerEmbedder`` — real semantic embeddings via
  ``sentence-transformers`` (optional dependency). Drops in unchanged and
  improves recall on paraphrases/synonyms.

Select one with ``default_embedder()``, which prefers the real model when it
is installed and not disabled via ``AGENT_MEMORY_EMBEDDER=hashing``.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Filtering very common words stops them from dominating the feature-hashed
# vector, so content words ("admin", "timezone", "gemini") drive similarity
# instead of "the"/"a"/"is". Standard practice; matters a lot for short queries.
_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have if in into is it its like make
    sure of on or other so than that the their then there these they this to up was
    will with you your new add added""".split()
)


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into unit-norm vectors of a fixed dimension."""

    dim: int
    # Cosine score below which a match should be treated as unrelated. Every
    # embedder has its own scale, so this travels with the backend rather than
    # being a single constant in the retrieval code.
    recommended_min_score: float

    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim) float32
        ...


def _features(text: str) -> list[str]:
    """Word unigrams + bigrams + 3-char n-grams.

    Char n-grams give the hashing embedder some robustness to morphology
    (``booking`` ~ ``bookings``) that pure word matching would miss.
    """
    words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS]
    feats: list[str] = list(words)
    feats += [f"{a}_{b}" for a, b in zip(words, words[1:])]
    for w in words:
        padded = f"#{w}#"
        feats += [padded[i : i + 3] for i in range(len(padded) - 2)]
    return feats


class HashingEmbedder:
    """Deterministic feature-hashing embedder. No dependencies, no network."""

    # Calibrated on eval/dataset.json: see the `min_score` sweep in
    # eval/results.md. At 0.15 labelled recall is untouched while half the
    # matches for off-topic queries are dropped. A higher floor cuts more noise
    # on that benchmark, but the lowest-scoring gold memory in it sits at 0.13,
    # so 0.15 is deliberately close to the observed floor of "genuinely
    # relevant" rather than as aggressive as the sweep alone would allow.
    recommended_min_score = 0.15

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _hash(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "big")
        index = h % self.dim
        sign = 1.0 if (h >> 63) & 1 else -1.0
        return index, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in _features(text):
                index, sign = self._hash(feature)
                out[row, index] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    """Real semantic embeddings. Optional: needs ``sentence-transformers``."""

    # NOT calibrated in this repo — the offline benchmark runs on the hashing
    # embedder, and MiniLM cosines sit on a higher scale (unrelated pairs often
    # score 0.1-0.3). Treat this as a conservative starting point and measure on
    # your own store before relying on it.
    recommended_min_score = 0.25

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return vecs.astype(np.float32)


def default_min_score(embedder: Embedder) -> float:
    """Relevance floor to use for an agent-facing call.

    ``AGENT_MEMORY_MIN_SCORE`` overrides; otherwise the backend's own
    calibrated value is used. Library callers of ``MemoryStore.recall`` get no
    floor unless they ask for one — this is applied at the CLI/MCP boundary,
    where the caller is an agent that cannot see the scores.
    """
    override = os.environ.get("AGENT_MEMORY_MIN_SCORE")
    if override is not None:
        return float(override)
    return float(getattr(embedder, "recommended_min_score", 0.0))


def default_embedder() -> Embedder:
    """Prefer the real model when available; fall back to hashing.

    ``AGENT_MEMORY_EMBEDDER`` selects explicitly: ``hashing`` forces the offline
    embedder (CI does this so results are byte-stable), ``sentence-transformers``
    demands the real one and raises if it cannot be loaded. The default is
    ``auto``, which tries the real model and warns — loudly, on stderr — before
    falling back, because the two produce incompatible vectors and a silent
    switch is how a store ends up half-embedded by each.
    """
    choice = os.environ.get("AGENT_MEMORY_EMBEDDER", "auto").lower()
    if choice == "hashing":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder()
    except Exception as exc:
        if choice in {"sentence-transformers", "sentence_transformers", "real"}:
            raise RuntimeError(
                f"AGENT_MEMORY_EMBEDDER={choice} but sentence-transformers could "
                f'not be loaded: {exc}. Install it with: pip install "agent-memory-engine[real]"'
            ) from exc
        print(
            f"[agent-memory] sentence-transformers unavailable ({exc.__class__.__name__}); "
            "using the offline HashingEmbedder. Set AGENT_MEMORY_EMBEDDER=hashing to "
            "silence this.",
            file=sys.stderr,
        )
    return HashingEmbedder()
