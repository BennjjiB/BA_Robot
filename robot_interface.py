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

    def get_images(self):
        color_image, depth_image, depth_colormap, annotated_frame, _ = self.__get_images_and_brick_poses()
        # adjust color bgr to rgb
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_image = np.clip(depth_image, -1, 1)
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
        depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        return color_image, depth_image, depth_colormap, annotated_frame

    def get_3d_image(self):
        color_image, depth_image, _, _, registered_bricks = self.__get_images_and_brick_poses()
        bricks, image_3d = self.robot.get_brick_poses(
            registered_bricks,
            color_image,
            depth_image
        )
        return cv2.cvtColor(image_3d, cv2.COLOR_BGR2RGB)

    def __get_images_and_brick_poses(self):
        color_image, depth_image, depth_colormap = self.webcam.capture_image()
        registered_bricks = self.maskModel(
            color_image, iou=0.9, verbose=False)[0]
        annotated_frame = registered_bricks.plot()
        return color_image, depth_image, depth_colormap, annotated_frame, registered_bricks
