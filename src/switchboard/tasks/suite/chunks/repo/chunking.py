"""Split a list into fixed-size pieces."""


def chunks(items: list[int], size: int) -> list[list[int]]:
    """Consecutive pieces of `size` items; the last one may be shorter. size must be >= 1."""
    if size < 1:
        raise ValueError("size must be >= 1")
    return [items[i : i + size] for i in range(0, len(items) - size + 1, size)]
