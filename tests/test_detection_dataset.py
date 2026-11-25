"""Unit tests for the SUN RGB-D -> YOLO conversion helpers (no torch, no real data needed)."""

import pytest

from latent_crew.data.detection_dataset import _to_yolo_lines, sun_key


def test_sun_key_strips_prefix_and_trailing_slash():
    assert sun_key("/n/fs/sun3d/data/SUNRGBD/kv1/b3dodata/img_0063/") == "SUNRGBD/kv1/b3dodata/img_0063"
    assert sun_key("xxx/SUNRGBD/kv2/align") == "SUNRGBD/kv2/align"


def test_to_yolo_lines_normalises_centre_and_size():
    # One 100x50 box at top-left (20, 10) in a 200x100 image.
    boxes = [("chair", 20.0, 10.0, 100.0, 50.0)]
    lines = _to_yolo_lines(boxes, {"chair": 0}, img_w=200, img_h=100, min_box_size=4)
    assert len(lines) == 1
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "0"
    # centre = (20 + 100/2, 10 + 50/2) = (70, 35) -> normalised (0.35, 0.35); size (0.5, 0.5)
    assert float(cx) == pytest.approx(0.35, abs=1e-4)
    assert float(cy) == pytest.approx(0.35, abs=1e-4)
    assert float(w) == pytest.approx(0.5, abs=1e-4)
    assert float(h) == pytest.approx(0.5, abs=1e-4)


def test_to_yolo_lines_drops_unknown_classes():
    boxes = [("chair", 0.0, 0.0, 40.0, 40.0), ("teapot", 0.0, 0.0, 40.0, 40.0)]
    lines = _to_yolo_lines(boxes, {"chair": 0}, img_w=100, img_h=100, min_box_size=4)
    assert len(lines) == 1
    assert lines[0].startswith("0 ")


def test_to_yolo_lines_clips_to_image_bounds():
    # Box runs past the right/bottom edge; after clipping it should stay within [0, 1].
    boxes = [("door", 80.0, 80.0, 60.0, 60.0)]  # x+w=140 > 100, y+h=140 > 100
    lines = _to_yolo_lines(boxes, {"door": 0}, img_w=100, img_h=100, min_box_size=4)
    assert len(lines) == 1
    _cls, cx, cy, w, h = (float(v) for v in lines[0].split())
    # Clipped box is (80,80)-(100,100): centre (90,90)->0.9, size 20->0.2
    assert cx == pytest.approx(0.9, abs=1e-4)
    assert cy == pytest.approx(0.9, abs=1e-4)
    assert w == pytest.approx(0.2, abs=1e-4)
    assert h == pytest.approx(0.2, abs=1e-4)
    assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0


def test_to_yolo_lines_drops_tiny_boxes():
    boxes = [("cup", 10.0, 10.0, 2.0, 2.0)]  # 2px < min_box_size
    lines = _to_yolo_lines(boxes, {"cup": 0}, img_w=100, img_h=100, min_box_size=4)
    assert lines == []
