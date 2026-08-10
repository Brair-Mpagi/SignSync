from __future__ import annotations

import pytest

from signsync.vision.tracker import ReplayTracker, Tracker


def test_replay_tracker_satisfies_the_tracker_protocol(clip):
    assert isinstance(ReplayTracker(clip), Tracker)


def test_replay_tracker_yields_every_frame_once(clip):
    tracker = ReplayTracker(clip)
    frames = list(tracker.frames())
    assert len(frames) == len(clip)
    assert tracker.exhausted

    with pytest.raises(StopIteration):
        tracker.track()


def test_replay_tracker_loops_when_asked(clip):
    tracker = ReplayTracker(clip, loop=True)
    for _ in range(len(clip) * 2 + 3):
        tracker.track()
    assert not tracker.exhausted


def test_replay_tracker_reset(clip):
    tracker = ReplayTracker(clip)
    tracker.track()
    tracker.reset()
    assert len(list(tracker.frames())) == len(clip)


def test_replay_tracker_can_restamp_timestamps(clip):
    frame = ReplayTracker(clip).track(None, timestamp=4.25)
    assert frame.timestamp == 4.25


def test_replay_tracker_from_file(tmp_path, clip):
    path = clip.save(tmp_path / "c.npz")
    assert len(list(ReplayTracker.from_file(path).frames())) == len(clip)


def test_mediapipe_tracker_reports_the_missing_extra():
    pytest.importorskip
    from signsync.capabilities import available
    from signsync.errors import MissingDependencyError
    from signsync.vision.tracker import MediaPipeHolisticTracker

    if available("mediapipe"):
        pytest.skip("mediapipe installed; nothing to assert about its absence")
    with pytest.raises(MissingDependencyError, match="vision"):
        MediaPipeHolisticTracker()
