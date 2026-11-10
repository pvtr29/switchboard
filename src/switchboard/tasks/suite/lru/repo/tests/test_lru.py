from lru import LRUCache


def test_evicts_oldest() -> None:
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    assert "a" not in c
    assert c.get("b") == 2


def test_get_counts_as_a_use() -> None:
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")
    c.put("c", 3)
    assert "a" in c
    assert "b" not in c
