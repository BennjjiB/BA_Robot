from foundation_pose.PoseEstimationApp import PoseEstimatorApp
from foundation_pose.real_sense_reader import RealSenseReader
import numpy as np
from ultralytics import YOLO
import cv2


class RobotInterface:
    def __init__(self) -> None:
        self.webcam = RealSenseReader()
        self.maskModel = YOLO('/home/panda3/Desktop/Robot_BA/best.pt')
        self.robot = PoseEstimatorApp(
            reader=self.webcam, maskModel=self.maskModel)
        self.is_sorting = False
        
    def stop_sorting(self):
        self.robot.stop()
        self.is_sorting = False

    def get_images(self):
        try:
            color_image, depth_image, depth_colormap, annotated_frame, _ = self.__get_images_and_brick_poses()
            # adjust color bgr to rgb
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            # Normalize depth image to [0, 255] range for visualization
            depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_image_uint8 = depth_image_normalized.astype(np.uint8)
            depth_image_colored = cv2.applyColorMap(depth_image_uint8, cv2.COLORMAP_JET)  # Visualize depth with color
            depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            return color_image, depth_image_colored, depth_colormap, annotated_frame
        except:
            print("camera is busy")
            return None, None, None, None

    def get_3d_image(self):
        _, _, image_3d = self.get_3d_bricks_and_image()
        return cv2.cvtColor(image_3d, cv2.COLOR_BGR2RGB)

    def sort_bricks(self, registered_bricks, bricks, by_color: bool = False):
        self.is_sorting = True
        self.robot.start_sort_pipeline(registered_bricks, bricks, by_color)
        self.is_sorting = False

    def __get_images_and_brick_poses(self):
        color_image, depth_image, depth_colormap = self.webcam.capture_image()
        registered_bricks = self.maskModel(
            color_image, iou=0.9, verbose=False)[0]
        annotated_frame = registered_bricks.plot()
        return color_image, depth_image, depth_colormap, annotated_frame, registered_bricks

    def get_3d_bricks_and_image(self):
        color_image, depth_image, _, _, registered_bricks = self.__get_images_and_brick_poses()
        bricks, image_3d = self.robot.get_brick_poses(
            registered_bricks,
            color_image,
            depth_image
        )
        return registered_bricks, bricks, image_3d

    def get_collision_free_bricks(self):
        _, bricks, _ = self.get_3d_bricks_and_image()
        self.robot.get_collision_free_bricks(bricks)