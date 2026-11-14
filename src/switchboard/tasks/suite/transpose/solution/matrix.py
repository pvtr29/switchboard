"""Small matrix helpers on lists of rows."""


def transpose(m: list[list[int]]) -> list[list[int]]:
    """Rows become columns. All rows have the same length."""
    if not m:
        return []
    return [[m[r][c] for r in range(len(m))] for c in range(len(m[0]))]
