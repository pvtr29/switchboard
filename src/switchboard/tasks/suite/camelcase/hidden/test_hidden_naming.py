from naming import snake_to_camel


def test_single_word() -> None:
    assert snake_to_camel("name") == "name"


def test_leading_underscore_kept() -> None:
    assert snake_to_camel("_private_value") == "_privateValue"


def test_doubled_underscores() -> None:
    assert snake_to_camel("a__b") == "aB"
