"""Agent Memory Engine — durable memory and handoffs for AI coding agents."""

from .embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    default_embedder,
    default_min_score,
)
from .store import (
    MEMORY_TYPES,
    STORE_FORMAT,
    MemoryEntry,
    MemoryStore,
    RecallHit,
)
from .tokens import count_tokens

__version__ = "0.3.0"

__all__ = [
    "MemoryStore",
    "MemoryEntry",
    "RecallHit",
    "MEMORY_TYPES",
    "STORE_FORMAT",
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "default_embedder",
    "default_min_score",
    "count_tokens",
]
