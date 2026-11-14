from brackets import balanced


def test_nested() -> None:
    assert balanced("{[()()]}")


def test_wrong_kind() -> None:
    assert not balanced("(]")
