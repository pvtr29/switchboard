# mypy: ignore-errors
# ruff: noqa
"""Tiny text statistics helpers."""

import re


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def word_count(text: str) -> int:
    return len(tokenize(text))


def top_words(text: str, n: int) -> list[tuple[str, int]]:
    """Return the n most common words (case-insensitive), most common first.

    Ties are broken alphabetically.
    """
    counts: dict[str, int] = {}
    for word in tokenize(text):
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[: n - 1]
