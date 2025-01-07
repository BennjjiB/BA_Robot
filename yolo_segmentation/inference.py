import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO

@torch.no_grad()
def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    pipeline.start(config)

    model = YOLO('/home/panda3/Desktop/Robot_BA/best.pt')

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        color_image = np.asanyarray(color_frame.get_data())

        results = model(color_image)     
        annotated_frame = results[0].plot()

        if(results[0].masks is not None):
            cv2.imshow('Estimating poses ...', annotated_frame)
        else:
            cv2.imshow('Estimating poses ...', color_image)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        if cv2.getWindowProperty('Estimating poses ...', cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
