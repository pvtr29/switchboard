from rle import decode, encode


def test_long_runs() -> None:
    assert encode("x" * 12) == "12x"
    assert decode("12x") == "x" * 12


def test_roundtrip() -> None:
    for s in ["", "a", "abba", "aaaaaaaaaaab", "mississippi"]:
        assert decode(encode(s)) == s


def test_empty() -> None:
    assert encode("") == ""
