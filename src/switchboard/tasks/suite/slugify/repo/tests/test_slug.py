from slug import slugify


def test_simple() -> None:
    assert slugify("Hello World") == "hello-world"


def test_punctuation_collapses() -> None:
    assert slugify("Hello,  World!") == "hello-world"
