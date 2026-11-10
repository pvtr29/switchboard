"""Roman numerals."""

_NUMERALS = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]
_VALUES = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}


def to_roman(n: int) -> str:
    """Convert 1..3999 to a Roman numeral in standard subtractive form (IV, IX, XL, CM...)."""
    if not 0 < n < 4000:
        raise ValueError("n must be in 1..3999")
    out = []
    for value, symbol in _NUMERALS:
        count, n = divmod(n, value)
        out.append(symbol * count)
    return "".join(out)


def from_roman(s: str) -> int:
    """Parse a Roman numeral (subtractive pairs allowed)."""
    total = 0
    for i, ch in enumerate(s):
        v = _VALUES[ch]
        if i + 1 < len(s) and _VALUES[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total
