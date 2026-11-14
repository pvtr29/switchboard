from dedupe import dedupe


def test_key_keeps_the_first_occurrence() -> None:
    assert dedupe(["Apple", "apple", "Pear"], key=str.lower) == ["Apple", "Pear"]


def test_key_order() -> None:
    assert dedupe([10, 21, 12, 31], key=lambda x: x % 10) == [10, 21, 12]


def test_empty() -> None:
    assert dedupe([]) == []
