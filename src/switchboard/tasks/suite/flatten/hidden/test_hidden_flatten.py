from flatten import flatten


def test_other_iterables_are_values() -> None:
    assert flatten(["ab", (1, 2), [{"k": 1}]]) == ["ab", (1, 2), {"k": 1}]


def test_empty_lists_vanish() -> None:
    assert flatten([[], [[]], 1]) == [1]


def test_input_unchanged() -> None:
    xs = [1, [2, [3]]]
    flatten(xs)
    assert xs == [1, [2, [3]]]
