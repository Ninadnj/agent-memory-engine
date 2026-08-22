import sys
from pathlib import Path

import pytest

from agent_memory import count_tokens

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from ingest_markdown import sections  # noqa: E402


def test_a_long_single_paragraph_is_split_under_the_token_limit():
    chunks = sections("## Notes\n" + "word " * 500, max_tokens=120)

    assert len(chunks) > 1
    assert all(count_tokens(chunk) <= 120 for chunk in chunks)


def test_a_long_unbroken_string_is_split_under_the_token_limit():
    chunks = sections("## Notes\n" + "abcdefghij" * 500, max_tokens=25)

    assert len(chunks) > 1
    assert all(count_tokens(chunk) <= 25 for chunk in chunks)


def test_token_limit_must_be_positive():
    with pytest.raises(ValueError, match="greater than zero"):
        sections("## Notes\nanything", max_tokens=0)
