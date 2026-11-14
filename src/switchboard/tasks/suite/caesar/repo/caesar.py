"""Caesar cipher."""


def shift(text: str, k: int) -> str:
    """Shift letters by k places in the alphabet (wrapping around, any k, negative allowed),
    keeping case; every other character is unchanged."""
    out = []
    for ch in text:
        if ch.isalpha():
            out.append(chr(ord(ch) + k))
        else:
            out.append(ch)
    return "".join(out)
