from rle import encode


def test_encode() -> None:
    assert encode("aaabcc") == "3a1b2c"


def test_single_char() -> None:
    assert encode("z") == "1z"
