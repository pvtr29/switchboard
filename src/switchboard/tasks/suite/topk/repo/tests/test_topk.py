from topk import top_words


def test_clear_winner() -> None:
    assert top_words(["a", "b", "a", "c", "a", "b"], 2) == ["a", "b"]


def test_ties_alphabetical() -> None:
    assert top_words(["pear", "apple", "fig", "pear", "fig", "apple"], 2) == ["apple", "fig"]
