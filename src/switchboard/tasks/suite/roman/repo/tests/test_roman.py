from roman import from_roman, to_roman


def test_simple_numbers() -> None:
    assert to_roman(3) == "III"
    assert to_roman(8) == "VIII"


def test_four_and_nine() -> None:
    assert to_roman(4) == "IV"
    assert to_roman(9) == "IX"


def test_parse() -> None:
    assert from_roman("XIV") == 14
