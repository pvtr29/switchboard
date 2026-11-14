from versions import compare


def test_numeric_not_lexical() -> None:
    assert compare("1.10.0", "1.9.0") == 1


def test_equal() -> None:
    assert compare("2.0.1", "2.0.1") == 0
