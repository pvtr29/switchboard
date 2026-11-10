from slug import slugify


def test_leading_and_trailing_separators() -> None:
    assert slugify("  --Already--Slugged--  ") == "already-slugged"


def test_accents_are_stripped() -> None:
    assert slugify("Café au lait") == "cafe-au-lait"
    assert slugify("Ünïcödé") == "unicode"


def test_digits_kept() -> None:
    assert slugify("Top 10 of 2026") == "top-10-of-2026"


def test_nothing_left() -> None:
    assert slugify("!!!") == ""
