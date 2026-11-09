import dataclasses

import pytest

from switchboard.obs import LatencyWindow, percentile


def test_percentile_is_exact_on_known_data() -> None:
    data = [float(i) for i in range(1, 101)]  # 1..100
    assert percentile(data, 50) == 50.0
    assert percentile(data, 99) == 99.0
    assert percentile(data, 100) == 100.0
    assert percentile(data, 1) == 1.0
    assert percentile(data, 90) == 90.0
    assert percentile(data, 99.9) == 100.0  # ceil(99.9) -> rank 100


def test_percentile_nearest_rank_small_samples() -> None:
    assert percentile([7.0], 50) == 7.0
    assert percentile([7.0], 99) == 7.0
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0  # unsorted input, rank ceil(1.5) = 2
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0  # rank 2, an observed value
    assert percentile([1.0, 2.0, 3.0, 4.0], 75) == 3.0
    assert percentile([], 50) == 0.0


def test_percentile_rejects_bad_p() -> None:
    for p in (0, -1, 100.5):
        with pytest.raises(ValueError, match="p must be"):
            percentile([1.0], p)


def test_window_snapshot_values() -> None:
    w = LatencyWindow(1000)
    for i in range(1, 101):
        w.record(i / 1000)
    s = w.snapshot()
    assert (s.count, s.window) == (100, 100)
    assert s.p50 == pytest.approx(0.050)
    assert s.p99 == pytest.approx(0.099)
    assert s.max == pytest.approx(0.100)
    assert s.mean == pytest.approx(0.0505)
    assert s.max_ever == pytest.approx(0.100)


def test_window_is_bounded_and_remembers_all_time_figures() -> None:
    w = LatencyWindow(10)
    w.record(100.0)  # falls out of the window below
    for _ in range(10):
        w.record(1.0)
    s = w.snapshot()
    assert s.window == 10
    assert s.count == 11
    assert s.max == 1.0  # the window forgot the outlier ...
    assert s.max_ever == 100.0  # ... the all-time max did not
    assert s.mean == pytest.approx(110.0 / 11)
    w.reset()
    assert dataclasses.astuple(w.snapshot()) == (0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_empty_window_and_bad_size() -> None:
    assert LatencyWindow(5).snapshot().p99 == 0.0
    with pytest.raises(ValueError, match="size"):
        LatencyWindow(0)
