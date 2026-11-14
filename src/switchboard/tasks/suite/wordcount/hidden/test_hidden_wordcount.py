from wordcount import count_words


def test_apostrophes_inside_words() -> None:
    assert count_words("Don't stop. don't!") == {"don't": 2, "stop": 1}


def test_quotes_around_words_are_not_part_of_them() -> None:
    assert count_words("'hello' said 'Hello'") == {"hello": 2, "said": 1}


def test_empty() -> None:
    assert count_words("") == {}
    assert count_words("... !!!") == {}
