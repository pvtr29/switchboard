from stats_utils import median


def test_odd() -> None:
    assert median([1, 2, 3]) == 2


def test_even() -> None:
    assert median([1, 2, 3, 4]) == 2.5
