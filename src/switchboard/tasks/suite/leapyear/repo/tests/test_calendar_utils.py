from calendar_utils import days_in_month, is_leap


def test_ordinary_leap_year() -> None:
    assert is_leap(2024)
    assert not is_leap(2023)


def test_century_is_not_leap() -> None:
    assert not is_leap(1900)
    assert days_in_month(1900, 2) == 28
