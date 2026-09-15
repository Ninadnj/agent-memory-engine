"""The format-3 JSON boundary: validated reads and atomic snapshot replacement.

The store owns transaction locks and rollback. These functions know only how
to read and write a snapshot; they never decide which memories to change.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ._locking import _replace_file
from .embeddings import Embedder, embedding_config, validated_embed
from .models import MemoryEntry, _entry_from_raw

STORE_FORMAT = 3
MAX_STORE_BYTES = 128 * 1024 * 1024


class StoreFormatError(ValueError):
    """An unreadable or invalid store was left untouched."""


def file_stamp(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size, stat.st_ino


def read_payload(path: Path) -> dict:
    """Read supported JSON without repairing or rewriting the source file."""
    try:
        if path.stat().st_size > MAX_STORE_BYTES:
            raise ValueError("store exceeds the supported local file size")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(
            payload.get("entries"), list
        ):
            raise ValueError("store must contain an entries array")
        version = payload.get("format", 1)
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or not 1 <= version <= STORE_FORMAT
        ):
            raise ValueError(f"unsupported store format {version!r}")
        return payload
    except (ValueError, TypeError, UnicodeError) as exc:
        raise StoreFormatError(
            f"cannot load {path}: {exc}; original file left untouched"
        ) from exc


def _encode_vector(vector: np.ndarray) -> str:
    """Float16 + base64 keeps vectors compact inside an inspectable JSON store."""
    return base64.b64encode(np.asarray(vector, dtype=np.float16).tobytes()).decode(
        "ascii"
    )


def _decode_vector(raw: str | list[float]) -> np.ndarray:
    if isinstance(raw, str):
        vector = np.frombuffer(
            base64.b64decode(raw, validate=True), dtype=np.float16
        ).astype(np.float32)
    else:  # Format 1 stored plain JSON float lists.
        vector = np.asarray(raw, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def read_snapshot(
    path: Path, embedder: Embedder
) -> tuple[list[MemoryEntry], np.ndarray, tuple[int, int, int] | None]:
    # A concurrent replacement during the read must trigger another reload.
    stamp = file_stamp(path)
    payload = read_payload(path)
    try:
        reembed = (
            payload.get("dim") != embedder.dim
            or payload.get("embedder") != type(embedder).__name__
            or payload.get("embedding_config") != embedding_config(embedder)
        )
        entries, vectors, seen = [], [], set()
        for raw in payload["entries"]:
            entry = _entry_from_raw(raw)
            if entry.id in seen:
                raise ValueError(f"duplicate stored memory id {entry.id!r}")
            seen.add(entry.id)
            entries.append(entry)
            embedding = raw.get("embedding")
            # Validate old vectors even when changing the embedding backend.
            vector = None if embedding is None else _decode_vector(embedding)
            if vector is not None and (
                vector.ndim != 1
                or not np.isfinite(vector).all()
                or len(vector) != payload.get("dim")
            ):
                raise ValueError(f"invalid embedding for {entry.id}")
            vectors.append(None if reembed else vector)
        missing = [i for i, vector in enumerate(vectors) if vector is None]
        if missing:
            fresh = validated_embed(embedder, [entries[i].text for i in missing])
            for slot, i in enumerate(missing):
                vectors[i] = fresh[slot]
        matrix = (
            np.array(vectors, dtype=np.float32)
            if vectors
            else np.zeros((0, embedder.dim), dtype=np.float32)
        )
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        raise StoreFormatError(
            f"cannot load {path}: {exc}; original file left untouched"
        ) from exc
    return entries, matrix, stamp


def write_snapshot(
    path: Path, entries: list[MemoryEntry], matrix: np.ndarray, embedder: Embedder
) -> None:
    """Write under the caller's OS lock; never truncate the existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for i, entry in enumerate(entries):
        raw = asdict(entry)
        _entry_from_raw(raw)
        raw["embedding"] = _encode_vector(matrix[i])
        records.append(raw)
    payload = {
        "format": STORE_FORMAT,
        "embedder": type(embedder).__name__,
        "embedding_config": embedding_config(embedder),
        "dim": embedder.dim,
        "entries": records,
    }
    content = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    if len(content.encode("utf-8")) > MAX_STORE_BYTES:
        raise ValueError("store exceeds the supported local file size")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_file(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
