import pyrealsense2 as rs
import numpy as np
import cv2
import atexit

class RealSenseReader:
    def __init__(self):
        # Setup the webcam
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        atexit.register(self.cleanup)
        self.profile = self.pipeline.get_active_profile()
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_sensor.set_option(rs.option.laser_power, 250)
        self.depth_scale = self.depth_sensor.get_depth_scale()

        self.intrinsics = self.profile.get_stream(
            rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[self.intrinsics.fx, 0, self.intrinsics.ppx],
                           [0, self.intrinsics.fy, self.intrinsics.ppy],
                           [0, 0, 1]])

        self.align = rs.align(rs.stream.color)

    def cleanup(self):
        self.pipeline.stop()

    def capture_image(self):
        """
        Captures color, depth, and depth colormap images using the RealSense camera.

        Returns:
            tuple: A tuple containing:
                - color_image (numpy.ndarray): The captured color image.
                - depth_image (numpy.ndarray): The depth image scaled by the depth scale.
                - depth_colormap (numpy.ndarray): The depth image visualized as a colormap.
        """
        success, frame = self.pipeline.try_wait_for_frames()
        if not success:
            print("real sense reader could not read frames")
            return None, None, None
        aligned_frames = self.align.process(frame)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if color_frame and depth_frame:
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.18), cv2.COLORMAP_JET
            )

            depth_image = depth_image * self.depth_scale

            return color_image, depth_image, depth_colormap
