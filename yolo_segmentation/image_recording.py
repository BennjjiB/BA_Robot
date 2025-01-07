import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
pipeline.start(config)

try:
    while True:
        align_to = rs.stream.color
        align = rs.align(align_to)
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        cv2.imshow('RealSense Color Image', color_image)

        key = cv2.waitKey(1)
        if key & 0xFF == ord(' '):  # Space bar to capture the image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            color_path = os.path.join("Images", f"color_{timestamp}.png")

            cv2.imwrite(color_path, color_image)

        elif key & 0xFF == ord('q') or key & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
