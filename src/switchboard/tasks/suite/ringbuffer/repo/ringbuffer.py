"""A fixed-size buffer that overwrites its oldest item when full."""


class RingBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._items: list[int] = []
        self._start = 0

    def __len__(self) -> int:
        return len(self._items)

    def push(self, x: int) -> None:
        if len(self._items) < self.capacity:
            self._items.append(x)
        else:
            self._items[self._start] = x
            self._start = (self._start + 1) % self.capacity

    def items(self) -> list[int]:
        """Oldest first."""
        return list(self._items)
