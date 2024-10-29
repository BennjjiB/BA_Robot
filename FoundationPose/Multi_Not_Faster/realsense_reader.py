import pyrealsense2 as rs
import numpy as np

class RealSenseReader:
    def __init__(self):
        self.setup_camera()

    def setup_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

        self.pipeline.start(config)
        self.profile = self.pipeline.get_active_profile()
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_sensor.set_option(rs.option.laser_power, 250)
        self.depth_scale = self.depth_sensor.get_depth_scale()

        self.camera_depth_to_disparity = rs.disparity_transform(True)
        self.camera_disparity_to_depth = rs.disparity_transform(False)
        self.camera_spatial = rs.spatial_filter()
        self.camera_spatial.set_option(rs.option.filter_magnitude, 5)
        self.camera_spatial.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.camera_spatial.set_option(rs.option.filter_smooth_delta, 1)
        self.camera_spatial.set_option(rs.option.holes_fill, 1)
        self.camera_temporal = rs.temporal_filter()
        self.camera_temporal.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.camera_temporal.set_option(rs.option.filter_smooth_delta, 1)

        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[self.intrinsics.fx, 0, self.intrinsics.ppx],
                           [0, self.intrinsics.fy, self.intrinsics.ppy],
                           [0, 0, 1]])

        self.align = rs.align(rs.stream.color)

    def capture_image(self):
        frame = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frame)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if color_frame and depth_frame:
            color_image = np.asanyarray(color_frame.get_data())
            filtered_depth_frame = self.camera_depth_to_disparity.process(depth_frame)
            filtered_depth_frame = self.camera_spatial.process(filtered_depth_frame)
            filtered_depth_frame = self.camera_temporal.process(filtered_depth_frame)
            filtered_depth_frame = self.camera_disparity_to_depth.process(filtered_depth_frame)

            depth_image = np.asanyarray(filtered_depth_frame.get_data()) * self.depth_scale
            
            return color_image, depth_image
