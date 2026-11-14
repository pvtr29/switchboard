from dedupe import dedupe


def test_keeps_order() -> None:
    assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]


def test_strings() -> None:
    assert dedupe(["b", "a", "b"]) == ["b", "a"]
