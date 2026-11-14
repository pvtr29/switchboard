from brackets import balanced


def test_crossed() -> None:
    assert not balanced("([)]")


def test_unclosed_and_extra_closer() -> None:
    assert not balanced("((")
    assert not balanced("())")


def test_other_characters_ignored() -> None:
    assert balanced("f(x[0]) + {a: b}")
    assert balanced("")
