import pytest

from chunking import chunks


def test_size_larger_than_list() -> None:
    assert chunks([1, 2], 5) == [[1, 2]]


def test_empty() -> None:
    assert chunks([], 3) == []


def test_bad_size() -> None:
    with pytest.raises(ValueError):
        chunks([1], 0)


def test_nothing_lost() -> None:
    items = list(range(17))
    for size in range(1, 20):
        assert [x for c in chunks(items, size) for x in c] == items
