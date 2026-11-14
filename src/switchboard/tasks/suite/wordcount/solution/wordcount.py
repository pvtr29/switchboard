"""Word frequencies."""

import re


def count_words(text: str) -> dict[str, int]:
    """How often each word occurs. Words are case-insensitive runs of letters and apostrophes
    ("Don't" and "don't" are the same word); everything else separates words."""
    counts: dict[str, int] = {}
    for word in re.findall(r"[a-z']+", text.lower()):
        word = word.strip("'")
        if word:
            counts[word] = counts.get(word, 0) + 1
    return counts
