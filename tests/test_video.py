import cv2
import pytest
import numpy as np

from swish_or_brick.video import read_frame, read_frames, resize_for_display, read_selected_frames, find_video

def make_test_video(path, n_frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64,48))
    for i in range(n_frames):
        frame = np.full((48,64,3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

def test_read_frame_returns_requested_frame(tmp_path):
    path = tmp_path / "test.avi"
    make_test_video(path, 20)

    frame = read_frame(str(path), 7)
    assert abs(frame.mean()- 70) < 3

def test_read_frames_gives_timestamps(tmp_path):
    path = tmp_path / "test.avi"
    make_test_video(path, 20)

    times = [t_ms for _, t_ms, _ in read_frames(str(path))]
    assert len(times) == 20
    assert abs(times[10] - 10 * 1000 / 30) < 1

def test_resize_for_display_keeps_aspect_ratio():
    frame = np.zeros((1080, 1440, 3), dtype=np.uint8)
    small, scale = resize_for_display(frame)
    assert small.shape[:2] == (720, 960)
    assert scale == pytest.approx(1.5)

def test_read_selected_frames(tmp_path):
    path = tmp_path / "test.avi"
    make_test_video(path, 20)

    result = [(index, frame.mean()) for index, _, frame in read_selected_frames(str(path), [12, 3, 3, 7])]
    assert [index for index, _ in result] == [3, 7, 12]
    assert [mean for _, mean in result] == pytest.approx([30, 70, 120], abs=3)

def test_read_selected_frames_stops_at_end(tmp_path):
    path = tmp_path / "test.avi"
    make_test_video(path, 20)

    indices = [index for index, _, _ in read_selected_frames(str(path), [5, 50])]
    assert indices == [5]

def test_find_video_searches_subfolders(tmp_path):
    (tmp_path / "2026-11-12").mkdir()
    target = tmp_path / "2026-11-12" / "a.mp4"
    target.write_bytes(b"")

    assert find_video("a.mp4", tmp_path) == target
    with pytest.raises(FileNotFoundError):
        find_video("b.mp4", tmp_path)