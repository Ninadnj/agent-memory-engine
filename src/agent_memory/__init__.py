"""Agent Memory Engine — semantic memory + handoffs for AI coding agents."""

from .embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    default_embedder,
)
from .store import MEMORY_TYPES, MemoryEntry, MemoryStore, RecallHit
from .tokens import count_tokens

__version__ = "0.1.0"

__all__ = [
    "MemoryStore",
    "MemoryEntry",
    "RecallHit",
    "MEMORY_TYPES",
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "default_embedder",
    "count_tokens",
]
