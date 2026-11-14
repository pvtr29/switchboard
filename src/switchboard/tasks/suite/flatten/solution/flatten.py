"""Flatten nested lists."""

from typing import Any


def flatten(items: list[Any]) -> list[Any]:
    """All non-list values of an arbitrarily nested list, in order. Only lists are flattened:
    strings, tuples and dicts are values."""
    out: list[Any] = []
    for x in items:
        if isinstance(x, list):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out
