import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# Start the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

device = pipeline.get_active_profile().get_device()
depth_sensor = device.first_depth_sensor()
depth_sensor.set_option(rs.option.laser_power, 360)

# Declare RealSense filters
decimation = rs.decimation_filter()
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()

try:
    # Get frames
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Check frame validity
    if not depth_frame or not color_frame:
        raise RuntimeError("Could not acquire depth or color frames.")

    # Apply filters to the depth frame
    depth_frame = decimation.process(depth_frame)
    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Create and map point cloud
    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)

    # Convert points to numpy array
    vtx = np.asanyarray(points.get_vertices())
    vtx = np.array(vtx.tolist())  # Convert to float32 for Open3D

    # Handle color information
    tex_coords = np.asanyarray(points.get_texture_coordinates())
    tex_coords = np.array(tex_coords.tolist())

    colors = np.zeros((len(tex_coords), 3), dtype=np.uint8)
    height, width, _ = color_image.shape
    for i, coord in enumerate(tex_coords):
        u, v = int(coord[0] * width), int(coord[1] * height)
        if 0 <= u < width and 0 <= v < height:
            colors[i] = color_image[v, u, ::-1]  # Convert BGR to RGB

    # Prepare point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vtx)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D', width=800, height=600)
    vis.add_geometry(pcd)
    view_control = vis.get_view_control()
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, -1, 0])
    view_control.set_front([0, 0, -1])
    view_control.set_zoom(0.5)
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    vis.run()

finally:
    pipeline.stop()
    vis.destroy_window()
