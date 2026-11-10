from intervals import merge


def test_unsorted_input() -> None:
    assert merge([(8, 10), (1, 3), (2, 6)]) == [(1, 6), (8, 10)]


def test_nested_and_chained() -> None:
    assert merge([(1, 10), (2, 3), (10, 12), (15, 15)]) == [(1, 12), (15, 15)]


def test_empty_and_single() -> None:
    assert merge([]) == []
    assert merge([(5, 7)]) == [(5, 7)]


def test_disjoint_stay_apart() -> None:
    assert merge([(1, 2), (3, 4)]) == [(1, 2), (3, 4)]
