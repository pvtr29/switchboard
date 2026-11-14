"""Bracket matching for (), [] and {}."""

PAIRS = {")": "(", "]": "[", "}": "{"}


def balanced(text: str) -> bool:
    """True if every bracket is closed by the matching kind, in order. Other characters are
    ignored."""
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in PAIRS:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0
