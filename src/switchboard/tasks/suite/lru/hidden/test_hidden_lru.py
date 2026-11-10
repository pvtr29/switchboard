import pytest

from lru import LRUCache


def test_put_of_existing_key_counts_as_a_use() -> None:
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("a", 10)
    c.put("c", 3)
    assert c.get("a") == 10
    assert "b" not in c
    assert len(c) == 2


def test_missing_key_returns_default_and_changes_nothing() -> None:
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("zzz", "none") == "none"
    c.put("c", 3)
    assert "a" not in c


def test_capacity_one() -> None:
    c = LRUCache(1)
    c.put("a", 1)
    c.put("b", 2)
    assert len(c) == 1
    assert c.get("b") == 2


def test_bad_capacity() -> None:
    with pytest.raises(ValueError):
        LRUCache(0)
