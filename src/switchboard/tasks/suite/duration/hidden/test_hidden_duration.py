import pytest

from duration import parse_duration


def test_all_units() -> None:
    assert parse_duration("2h5m7s") == 7507
    assert parse_duration("90m") == 5400


@pytest.mark.parametrize("bad", ["", "5", "1x", "1m1h", "h", "1h 2m"])
def test_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_duration(bad)
