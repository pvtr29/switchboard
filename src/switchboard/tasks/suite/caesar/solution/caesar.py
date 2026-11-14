"""Caesar cipher."""


def shift(text: str, k: int) -> str:
    """Shift letters by k places in the alphabet (wrapping around, any k, negative allowed),
    keeping case; every other character is unchanged."""
    out = []
    for ch in text:
        if "a" <= ch <= "z" or "A" <= ch <= "Z":
            base = ord("a") if ch.islower() else ord("A")
            out.append(chr(base + (ord(ch) - base + k) % 26))
        else:
            out.append(ch)
    return "".join(out)
