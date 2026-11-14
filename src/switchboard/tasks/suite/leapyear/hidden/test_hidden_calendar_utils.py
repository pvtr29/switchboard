from calendar_utils import days_in_month, is_leap


def test_every_400_years_is_leap() -> None:
    assert is_leap(2000)
    assert is_leap(1600)
    assert days_in_month(2000, 2) == 29


def test_other_months() -> None:
    assert days_in_month(2023, 4) == 30
    assert days_in_month(2023, 12) == 31
