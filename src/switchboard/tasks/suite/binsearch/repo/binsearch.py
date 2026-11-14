"""Binary search over a sorted list."""


def find(items: list[int], target: int) -> int:
    """Index of `target` in the sorted list `items`, or -1 if it is not there."""
    lo, hi = 0, len(items) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
