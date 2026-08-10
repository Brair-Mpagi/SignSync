from __future__ import annotations

import pytest

from signsync.perf import FpsMeter, LatencyTracker


def test_fps_meter_needs_two_ticks():
    meter = FpsMeter()
    assert meter.fps == 0.0
    meter.tick(0.0)
    assert meter.fps == 0.0
    meter.tick(0.5)
    assert meter.fps == pytest.approx(2.0)


def test_fps_meter_uses_a_sliding_window():
    """A stall must show up, not be averaged away by an earlier fast stretch."""
    meter = FpsMeter(window=4)
    for i in range(10):
        meter.tick(i * 0.01)  # 100 FPS
    assert meter.meets(25)

    now = 0.09
    for _ in range(4):
        now += 0.2  # 5 FPS
        meter.tick(now)
    assert not meter.meets(25)


def test_fps_meter_rejects_a_useless_window():
    with pytest.raises(ValueError):
        FpsMeter(window=1)


def test_latency_tracker_summarises_stages():
    tracker = LatencyTracker()
    for ms in (10, 12, 11, 90):
        tracker.record("recognition", ms)
    tracker.record("translation", 5)

    stage = tracker.stage("recognition")
    assert stage is not None
    assert stage.count == 4
    assert stage.max_ms == 90
    assert stage.p50_ms <= stage.p95_ms <= stage.max_ms


def test_latency_report_totals_and_flags_the_bottleneck():
    tracker = LatencyTracker()
    tracker.record("vision", 20)
    tracker.record("recognition", 300)
    tracker.record("tts", 40)

    report = tracker.report()
    assert report.total_p95_ms == pytest.approx(360)
    assert report.meets(2000)  # objective O11
    assert not report.meets(100)
    slowest = report.slowest()
    assert slowest is not None and slowest.stage == "recognition"


def test_measure_context_manager_records_a_stage():
    tracker = LatencyTracker()
    with tracker.measure("vision"):
        pass
    assert tracker.stage("vision") is not None


def test_unknown_stage_is_none_not_an_error():
    assert LatencyTracker().stage("nope") is None
