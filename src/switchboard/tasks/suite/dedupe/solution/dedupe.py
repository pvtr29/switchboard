"""Remove duplicates."""

from collections.abc import Callable, Hashable
from typing import Any


def dedupe(items: list[Any], key: Callable[[Any], Hashable] | None = None) -> list[Any]:
    """Items without repeats, keeping the FIRST occurrence and the original order. With `key`,
    two items are the same when their keys are equal."""
    seen: set[Hashable] = set()
    out: list[Any] = []
    for x in items:
        k = x if key is None else key(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out
