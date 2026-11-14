"""Parse durations like "1h30m" or "45s"."""

import re

_PART = re.compile(r"(\d+)([hms])")
_UNIT = {"h": 3600, "m": 3600, "s": 1}


def parse_duration(text: str) -> int:
    """Seconds in a duration made of <number><unit> parts (units h, m, s, each at most once,
    in that order). Anything else raises ValueError."""
    if not re.fullmatch(r"(\d+h)?(\d+m)?(\d+s)?", text) or not text:
        raise ValueError(f"bad duration: {text!r}")
    return sum(int(n) * _UNIT[u] for n, u in _PART.findall(text))
