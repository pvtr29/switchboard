"""A small least-recently-used cache."""

from collections import OrderedDict


class LRUCache:
    """Holds at most `capacity` entries; adding one more evicts the least recently used.

    Both `get` (of a present key) and `put` count as a use of that key.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._data: OrderedDict[str, object] = OrderedDict()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: object = None) -> object:
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: object) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)
