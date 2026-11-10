import pytest

from roman import from_roman, to_roman


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (14, "XIV"),
        (40, "XL"),
        (90, "XC"),
        (400, "CD"),
        (944, "CMXLIV"),
        (1994, "MCMXCIV"),
        (3999, "MMMCMXCIX"),
    ],
)
def test_subtractive_forms(n: int, expected: str) -> None:
    assert to_roman(n) == expected


def test_round_trip_every_number() -> None:
    for n in range(1, 4000):
        assert from_roman(to_roman(n)) == n


def test_out_of_range() -> None:
    with pytest.raises(ValueError):
        to_roman(0)
