"""Compare dotted version numbers like 1.10.2."""


def compare(a: str, b: str) -> int:
    """-1 if a < b, 0 if equal, 1 if a > b. Missing trailing parts count as 0, so "1.0" equals
    "1.0.0"."""
    pa = [int(x) for x in a.split(".")]
    pb = [int(x) for x in b.split(".")]
    n = max(len(pa), len(pb))
    pa += [0] * (n - len(pa))
    pb += [0] * (n - len(pb))
    return (pa > pb) - (pa < pb)
