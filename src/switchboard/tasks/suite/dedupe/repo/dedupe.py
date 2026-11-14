"""Remove duplicates."""

from collections.abc import Callable, Hashable
from typing import Any


def dedupe(items: list[Any], key: Callable[[Any], Hashable] | None = None) -> list[Any]:
    """Items without repeats, keeping the FIRST occurrence and the original order. With `key`,
    two items are the same when their keys are equal."""
    if key is None:
        return list(set(items))
    seen: dict[Hashable, Any] = {}
    for x in items:
        seen[key(x)] = x
    return list(seen.values())
