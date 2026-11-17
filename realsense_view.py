"""Minimal live RGB + depth viewer for an Intel RealSense D435i."""
import numpy as np, cv2, pyrealsense2 as rs

pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipe.start(cfg)
align = rs.align(rs.stream.color)  # align depth to the RGB frame
try:
    while True:
        f = align.process(pipe.wait_for_frames())
        color = np.asanyarray(f.get_color_frame().get_data())
        depth = np.asanyarray(f.get_depth_frame().get_data())
        depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("RealSense RGB | depth", np.hstack([color, depth_vis]))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
finally:
    pipe.stop(); cv2.destroyAllWindows()
