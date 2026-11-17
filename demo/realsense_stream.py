"""Demo: live RGB + depth stream from an Intel RealSense camera (e.g. the D435i).

Shows the colour image and the colourised depth side by side in one OpenCV window, with the measured FPS.
Press q (or Esc) to quit.

Demo tool: it lives in demo/, outside the pytest suite.

Run from the repository root (the RealSense SDK comes from the optional `camera` extra in pyproject.toml):
    demo/realsense_stream.sh
    demo/realsense_stream.sh --width 848 --height 480 --max-depth 4
    demo/realsense_stream.sh --sudo        # macOS: if the camera is not found or USB access is refused

Supported: macOS 15+ on Apple Silicon (pyrealsense2-macosx), Linux x86_64 and Windows (pyrealsense2), Python 3.14.
Linux aarch64 (e.g. the DGX Spark) has no pyrealsense2 wheel; there librealsense must be built from source.

The camera needs a USB 3 connection (USB-C port and a USB 3 capable cable, ideally no hub); over USB 2 the
depth stream either fails to start or only runs at low resolution and frame rate.
"""

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as error:  # pragma: no cover - only hit on machines without the SDK
    raise SystemExit("pyrealsense2 is missing: run demo/realsense_stream.sh (uses the `camera` extra), "
                     "or uv run --extra camera python demo/realsense_stream.py") from error

WINDOW = "RealSense: RGB | depth (q to quit)"


@dataclass
class StreamSettings:
    """Resolution and frame rate of both streams, and the depth range mapped to the colour scale."""

    width: int = 640
    height: int = 480
    fps: int = 30
    max_depth_m: float = 6.0


class RealSenseStream:
    """Starts the colour and depth streams and returns depth aligned to the colour image.

    Use as a context manager, so the camera is always released:
        with RealSenseStream(settings) as camera:
            color, depth_m = camera.read()
    """

    def __init__(self, settings: StreamSettings):
        self.settings = settings
        self.pipeline = rs.pipeline()
        # Depth and colour come from different sensors; align maps every depth pixel onto the colour image
        self.align = rs.align(rs.stream.color)
        self.depth_scale = 0.001  # metres per depth unit, replaced by the device's value on start

    def __enter__(self) -> "RealSenseStream":
        s = self.settings
        config = rs.config()
        config.enable_stream(rs.stream.color, s.width, s.height, rs.format.bgr8, s.fps)
        config.enable_stream(rs.stream.depth, s.width, s.height, rs.format.z16, s.fps)
        profile = self.pipeline.start(config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        device = profile.get_device()
        print(f"Streaming from {device.get_info(rs.camera_info.name)} "
              f"(serial {device.get_info(rs.camera_info.serial_number)}), {s.width}x{s.height} @ {s.fps} fps")
        return self

    def __exit__(self, *exc) -> None:
        self.pipeline.stop()

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Waits for the next frame pair; returns (BGR uint8 [H, W, 3], depth in metres float32 [H, W], 0 = no data)."""
        frames = self.align.process(self.pipeline.wait_for_frames())
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * self.depth_scale
        return color, depth


def colorize_depth(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    """Depth in metres -> BGR image (near = warm, far = cold); pixels without a measurement are black."""
    scaled = np.clip(depth_m / max_depth_m, 0.0, 1.0)
    color = cv2.applyColorMap(((1.0 - scaled) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[depth_m == 0] = 0
    return color


class FpsCounter:
    """Frames per second, smoothed with an exponential moving average so the number does not flicker."""

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self.last = time.perf_counter()

    def tick(self) -> float:
        """Registers one frame and returns the smoothed FPS."""
        now = time.perf_counter()
        current = 1.0 / max(now - self.last, 1e-6)
        self.last = now
        self.fps = current if self.fps == 0.0 else self.smoothing * self.fps + (1 - self.smoothing) * current
        return self.fps


def draw_label(image: np.ndarray, text: str) -> None:
    """Writes `text` in the top-left corner, white on a black outline so it is readable on any background."""
    for color, thickness in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness, cv2.LINE_AA)


def parse_args() -> StreamSettings:
    """Command-line arguments -> stream settings."""
    parser = argparse.ArgumentParser(description="Live RGB + depth view of a RealSense camera.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-depth", type=float, default=6.0, help="Depth (m) shown as the far end of the colour scale")
    args = parser.parse_args()
    return StreamSettings(args.width, args.height, args.fps, args.max_depth)


def main() -> None:
    """Shows the stream until q or Esc is pressed or the window is closed."""
    settings = parse_args()
    fps = FpsCounter()
    with RealSenseStream(settings) as camera:
        while True:
            color, depth_m = camera.read()
            depth_view = colorize_depth(depth_m, settings.max_depth_m)

            draw_label(color, f"RGB  {fps.tick():.1f} fps")
            center = depth_m[depth_m.shape[0] // 2, depth_m.shape[1] // 2]
            draw_label(depth_view, f"Depth  centre {center:.2f} m" if center > 0 else "Depth  centre: no data")
            cv2.imshow(WINDOW, np.hstack([color, depth_view]))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
