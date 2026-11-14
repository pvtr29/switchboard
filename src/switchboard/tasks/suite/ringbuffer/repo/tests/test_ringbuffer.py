from ringbuffer import RingBuffer


def test_not_full() -> None:
    r = RingBuffer(3)
    r.push(1)
    r.push(2)
    assert r.items() == [1, 2]


def test_overwrites_oldest() -> None:
    r = RingBuffer(3)
    for x in [1, 2, 3, 4]:
        r.push(x)
    assert r.items() == [2, 3, 4]
