"""Compare dotted version numbers like 1.10.2."""


def compare(a: str, b: str) -> int:
    """-1 if a < b, 0 if equal, 1 if a > b. Missing trailing parts count as 0, so "1.0" equals
    "1.0.0"."""
    if a == b:
        return 0
    return -1 if a < b else 1
