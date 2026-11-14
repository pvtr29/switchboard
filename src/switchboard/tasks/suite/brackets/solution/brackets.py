"""Bracket matching for (), [] and {}."""

PAIRS = {")": "(", "]": "[", "}": "{"}


def balanced(text: str) -> bool:
    """True if every bracket is closed by the matching kind, in order. Other characters are
    ignored."""
    stack: list[str] = []
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in PAIRS:
            if not stack or stack.pop() != PAIRS[ch]:
                return False
    return not stack
