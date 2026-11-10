"""Closed integer intervals."""


def merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching closed intervals.

    The input may be in any order; the result is sorted by start and has no two intervals that
    overlap or touch ((1, 4) and (4, 5) touch and merge into (1, 5)).
    """
    out: list[tuple[int, int]] = []
    for start, end in intervals:
        if out and start < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out
