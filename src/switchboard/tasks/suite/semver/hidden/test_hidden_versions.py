from versions import compare


def test_missing_parts_are_zero() -> None:
    assert compare("1.0", "1.0.0") == 0
    assert compare("1.0.1", "1.0") == 1
    assert compare("1", "1.0.1") == -1


def test_antisymmetric() -> None:
    pairs = [("0.9", "0.10"), ("3.2.1", "3.12"), ("10", "9.9.9")]
    for a, b in pairs:
        assert compare(a, b) == -compare(b, a)
