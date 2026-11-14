from chunking import chunks


def test_even_split() -> None:
    assert chunks([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_remainder() -> None:
    assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
