import pytest

from ringbuffer import RingBuffer


def test_many_wraps() -> None:
    r = RingBuffer(4)
    for x in range(23):
        r.push(x)
    assert r.items() == [19, 20, 21, 22]
    assert len(r) == 4


def test_capacity_one() -> None:
    r = RingBuffer(1)
    r.push(5)
    r.push(6)
    assert r.items() == [6]


def test_bad_capacity() -> None:
    with pytest.raises(ValueError):
        RingBuffer(0)
