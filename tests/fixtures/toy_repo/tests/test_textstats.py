# mypy: ignore-errors
# ruff: noqa
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textstats import top_words, word_count


def test_word_count():
    assert word_count("the quick brown fox") == 4


def test_word_count_empty():
    assert word_count("") == 0


def test_top_words_basic():
    assert top_words("a b b c c c", 2) == [("c", 3), ("b", 2)]


def test_top_words_case_insensitive():
    assert top_words("The the THE cat Cat dog", 2) == [("the", 3), ("cat", 2)]


def test_top_words_tie_alphabetical():
    assert top_words("pear apple pear apple fig", 2) == [("apple", 2), ("pear", 2)]
