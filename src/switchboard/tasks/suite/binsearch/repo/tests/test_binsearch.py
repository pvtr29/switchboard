from binsearch import find


def test_finds_middle() -> None:
    assert find([1, 3, 5, 7, 9], 5) == 2


def test_finds_last() -> None:
    assert find([1, 3, 5, 7, 9], 9) == 4


def test_missing() -> None:
    assert find([1, 3, 5], 4) == -1
