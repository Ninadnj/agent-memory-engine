"""Token counting used by the evaluation harness.

The whole pitch of the engine is *context efficiency*, so we measure the cost
of what gets loaded into the model in tokens, not characters. Uses ``tiktoken``
(OpenAI's BPE) when installed for an exact count; otherwise falls back to a
word/punctuation approximation that is close enough to compare arms fairly.
"""

from __future__ import annotations

import re
from functools import lru_cache

_APPROX_RE = re.compile(r"\w+|[^\w\s]")


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    # Approximation: count word and punctuation chunks, then nudge up ~30%
    # because BPE typically splits long words into multiple tokens.
    return round(len(_APPROX_RE.findall(text)) * 1.3)


def using_exact_tokenizer() -> bool:
    return _encoder() is not None
