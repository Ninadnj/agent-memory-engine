"""Agent Memory Engine — durable memory and handoffs for AI coding agents."""

from .embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    default_embedder,
    default_min_score,
)
from .store import (
    GLOBAL_STORE,
    HALF_LIFE_DAYS,
    MEMORY_TYPES,
    STORE_FORMAT,
    MemoryEntry,
    MemoryStore,
    RecallHit,
    age_in_days,
    decay_factor,
    default_store_path,
    find_project_root,
)
from .tokens import count_tokens

__version__ = "0.3.0"

__all__ = [
    "MemoryStore",
    "MemoryEntry",
    "RecallHit",
    "MEMORY_TYPES",
    "STORE_FORMAT",
    "GLOBAL_STORE",
    "HALF_LIFE_DAYS",
    "age_in_days",
    "decay_factor",
    "default_store_path",
    "find_project_root",
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "default_embedder",
    "default_min_score",
    "count_tokens",
]
