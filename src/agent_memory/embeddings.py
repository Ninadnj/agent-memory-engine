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


def default_embedder() -> Embedder:
    """Prefer the real model when available; fall back to hashing.

    Force the offline embedder with ``AGENT_MEMORY_EMBEDDER=hashing`` (CI does
    this so results are byte-stable).
    """
    if os.environ.get("AGENT_MEMORY_EMBEDDER", "").lower() != "hashing":
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            pass
    return HashingEmbedder()
