from caesar import shift


def test_negative_and_large_shifts() -> None:
    assert shift("abc", -1) == "zab"
    assert shift("abc", 27) == "bcd"


def test_roundtrip() -> None:
    s = "The quick brown fox, 42."
    assert shift(shift(s, 11), -11) == s


def test_non_ascii_letters_unchanged() -> None:
    assert shift("café", 1) == "dbgé"
