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

New stores use hashing unless ``AGENT_MEMORY_EMBEDDER=sentence-transformers``
is explicitly selected. Installing optional packages does not change defaults.
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
    # Cosine score below which a match should be treated as unrelated. Every
    # embedder has its own scale, so this travels with the backend rather than
    # being a single constant in the retrieval code.
    recommended_min_score: float

    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim) float32
        ...


def validated_embed(embedder: Embedder, texts: list[str]) -> np.ndarray:
    """Reject invalid backend output before it can reach live or persisted state."""
    vectors = np.asarray(embedder.embed(texts), dtype=np.float32)
    if vectors.shape != (len(texts), embedder.dim) or not np.isfinite(vectors).all():
        raise ValueError("embedder returned invalid vectors")
    return vectors


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
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ValueError("embedding dimension must be a positive integer")
        self.dim = dim

    @property
    def configuration(self) -> dict:
        return {
            "backend": "hashing",
            "features_version": 1,
            "dim": self.dim,
            "normalization": "l2",
            "hash": "blake2b-64",
        }

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

    # Calibrated on eval/dataset.json — see eval/results_sentence_transformers.md.
    # Every off-topic query is already rejected at 0.15, and labelled recall is
    # flat until it falls at 0.40, so anything in 0.15-0.35 scores identically on
    # the benchmark. The tie is broken by headroom rather than by the sweep:
    # genuine paraphrases can land low ("which AI model answers customer
    # questions" scores 0.22 against a memory that answers it), so the floor sits
    # just under that rather than in the middle of the safe band. MiniLM cosines
    # run higher than the hashing embedder's, which is why this differs from it.
    recommended_min_score = 0.20

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", revision: str | None = None
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(
            model_name, **({"revision": revision} if revision else {})
        )
        # Renamed in sentence-transformers 5.x; support both so an upgrade of an
        # optional dependency cannot break the backend.
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            getattr(self._model, "get_sentence_embedding_dimension", None),
        )
        if get_dim is None:  # pragma: no cover - depends on the installed version
            raise RuntimeError(
                "could not determine the embedding dimension from "
                f"sentence-transformers model {model_name!r}"
            )
        self.dim = int(get_dim())
        self.model_name = model_name
        self.revision = revision

    @property
    def configuration(self) -> dict:
        return {
            "backend": "sentence-transformers",
            "model": self.model_name,
            "revision": self.revision,
            "dim": self.dim,
            "normalization": "l2",
        }

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


def embedding_config(embedder: Embedder) -> dict:
    """Stable persisted configuration; custom embedders may expose a dict.

    Model revisions should be immutable commit IDs. A floating model name
    cannot detect a remote weight change that retains the same name.
    """
    config = getattr(embedder, "configuration", None)
    if config is not None:
        return dict(config)
    return {
        "backend": f"{type(embedder).__module__}.{type(embedder).__qualname__}",
        "dim": embedder.dim,
        "model": getattr(embedder, "model_name", None),
        "revision": getattr(embedder, "revision", None),
    }


def default_embedder() -> Embedder:
    """Offline by default; semantic retrieval is an explicit choice.

    MemoryStore resolves ``auto`` from a saved store's configuration first.
    For new stores (or standalone use here), ``auto`` means hashing.
    """
    choice = os.environ.get("AGENT_MEMORY_EMBEDDER", "auto").strip().lower()
    if choice in {"auto", "hashing"}:
        return HashingEmbedder()
    if choice not in {"sentence-transformers", "sentence_transformers", "real"}:
        raise ValueError(
            f"unknown AGENT_MEMORY_EMBEDDER={choice!r}; "
            "use hashing, sentence-transformers or auto"
        )
    try:
        return SentenceTransformerEmbedder()
    except Exception as exc:
        raise RuntimeError(
            f"AGENT_MEMORY_EMBEDDER={choice} but sentence-transformers could "
            f'not be loaded: {exc}. Install it with: pip install "agent-memory-engine[real]"'
        ) from exc
