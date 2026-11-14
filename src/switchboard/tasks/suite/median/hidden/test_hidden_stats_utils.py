import pytest

from stats_utils import median


def test_unsorted() -> None:
    assert median([9, 1, 5]) == 5
    assert median([4, 1, 3, 2]) == 2.5


def test_does_not_modify_the_input() -> None:
    xs = [3, 1, 2]
    median(xs)
    assert xs == [3, 1, 2]


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        median([])
