from topk import top_words


def test_k_larger_than_vocabulary() -> None:
    assert top_words(["b", "a", "b"], 10) == ["b", "a"]


def test_k_zero_and_empty() -> None:
    assert top_words(["a"], 0) == []
    assert top_words([], 3) == []


def test_tie_after_the_leader() -> None:
    words = ["z"] * 3 + ["m", "b", "m", "b"]
    assert top_words(words, 3) == ["z", "b", "m"]
