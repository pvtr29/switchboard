from matrix import transpose


def test_tall() -> None:
    assert transpose([[1], [2], [3]]) == [[1, 2, 3]]


def test_empty() -> None:
    assert transpose([]) == []


def test_twice_is_identity() -> None:
    m = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    assert transpose(transpose(m)) == m
