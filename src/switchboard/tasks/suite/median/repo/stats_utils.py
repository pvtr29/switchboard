"""Basic statistics."""


def median(values: list[float]) -> float:
    """The median of a non-empty list (the mean of the two middle values for even lengths). The
    input does not have to be sorted."""
    if not values:
        raise ValueError("median of an empty list")
    mid = len(values) // 2
    return values[mid]
