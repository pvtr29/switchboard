from wordcount import count_words


def test_case_insensitive() -> None:
    assert count_words("The cat and the hat") == {"the": 2, "cat": 1, "and": 1, "hat": 1}


def test_punctuation_separates() -> None:
    assert count_words("one,two;one") == {"one": 2, "two": 1}
