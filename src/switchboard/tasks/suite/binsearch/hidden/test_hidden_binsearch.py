from binsearch import find


def test_every_element_is_found() -> None:
    items = list(range(0, 40, 3))
    for i, x in enumerate(items):
        assert find(items, x) == i


def test_single_and_empty() -> None:
    assert find([7], 7) == 0
    assert find([7], 8) == -1
    assert find([], 1) == -1


def test_values_outside_the_range() -> None:
    assert find([2, 4, 6], 1) == -1
    assert find([2, 4, 6], 7) == -1
