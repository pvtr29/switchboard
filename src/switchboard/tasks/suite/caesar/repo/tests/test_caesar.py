from caesar import shift


def test_wraps() -> None:
    assert shift("xyz", 3) == "abc"


def test_case_and_punctuation() -> None:
    assert shift("Hello, World!", 1) == "Ifmmp, Xpsme!"
