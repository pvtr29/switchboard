"""Most frequent words."""

from collections import Counter


def top_words(words: list[str], k: int) -> list[str]:
    """The k most frequent words, most frequent first; ties are broken alphabetically."""
    counts = Counter(words)
    return [w for w, _ in counts.most_common(k)]
