"""Most frequent words."""

from collections import Counter


def top_words(words: list[str], k: int) -> list[str]:
    """The k most frequent words, most frequent first; ties are broken alphabetically."""
    if k <= 0:
        return []
    counts = Counter(words)
    return sorted(counts, key=lambda w: (-counts[w], w))[:k]
