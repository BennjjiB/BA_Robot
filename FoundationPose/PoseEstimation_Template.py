import cv2
import numpy as np
from estimater import *
from datareader import *
import pyrealsense2 as rs
from ultralytics import YOLO

class PoseEstimatorApp:
    def __init__(self):
        self.center_pose = None
        self.code_dir = os.path.dirname(os.path.realpath(__file__))
        self.test_scene_dir = f'{self.code_dir}/out'
        self.est_refine_iter = 5
        self.track_refine_iter = 2
        self.debug = 1
        self.debug_dir = f'{code_dir}/debug'
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.reader = YcbineoatReader(video_dir=self.test_scene_dir, shorter_side=None, zfar=np.inf)
        self.maskModel = YOLO('/home/panda3/Desktop/Robot_BA/best.pt')
        self.track = False
        self.start_estimate = False

        self.mesh_4x2 = trimesh.load(f'{self.code_dir}/out/mesh/4x2_brick.obj')
        self.est_4x2 = FoundationPose(model_pts=self.mesh_4x2.vertices, model_normals=self.mesh_4x2.vertex_normals, mesh=self.mesh_4x2, scorer=self.scorer, refiner=self.refiner, debug_dir=self.debug_dir, debug=self.debug, glctx=self.glctx)
        self.mesh_2x2 = trimesh.load(f'{self.code_dir}/out/mesh/2x2_brick.obj')
        self.est_2x2 = FoundationPose(model_pts=self.mesh_2x2.vertices, model_normals=self.mesh_2x2.vertex_normals, mesh=self.mesh_2x2, scorer=self.scorer, refiner=self.refiner, debug_dir=self.debug_dir, debug=self.debug, glctx=self.glctx)

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        print("[INFO] Setting up camera ...")
        self.pipeline.start(config)
        profile = self.pipeline.get_active_profile()
        sensor = profile.get_device().query_sensors()[1]
        sensor.set_option(rs.option.enable_auto_exposure, 1)
        print("[INFO] Camera ready.")

    def draw_button(self, img, text, width):
        text_color = (255, 255, 255)
        button_color = (0, 0, 0)
        button_text = text
        button_x, button_y = (50, 50)
        button_width, button_height = (width, 50)
        cv2.rectangle(img, (button_x, button_y), (button_x + button_width, button_y + button_height), button_color, -1)
        cv2.putText(img, button_text, (button_x + 10, button_y + button_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            button_x, button_y = (50, 50)
            if(self.track):
                button_width, button_height = (100, 50)
            else:
                button_width, button_height = (150, 50)
            if button_x < x < button_x + button_width and button_y < y < button_y + button_height:
                if(self.center_pose is not None):
                    self.grip_brick(self.center_pose)
                else:
                    self.start_estimate = True

    def run(self):
        while True:
            align_to = rs.stream.color
            align = rs.align(align_to)
            frames = self.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.17), cv2.COLORMAP_JET
            )
            depth_image = depth_image / 1e3
            color_image = np.asanyarray(color_frame.get_data())
            color_image_copy = color_image.copy()

            results = self.maskModel(color_image, conf=0.6)     

            annotated_frame = results[0].plot()

            if(not self.track):
                try:
                    obj = results[0].boxes[0]
                    size = self.maskModel.names[int(obj.cls)][:3]

                    if size == "4x2":
                        est = self.est_4x2
                        mesh = self.mesh_4x2
                        to_origin = trimesh.bounds.oriented_bounds(mesh)[0]
                        extents = trimesh.bounds.oriented_bounds(mesh)[1]
                        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
                    elif size == "2x2":
                        est = self.est_2x2
                        mesh = self.mesh_2x2
                        to_origin = trimesh.bounds.oriented_bounds(mesh)[0]
                        extents = trimesh.bounds.oriented_bounds(mesh)[1]
                        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
                except:
                    pass

                cv2.namedWindow("Estimating poses ...")
                cv2.setMouseCallback("Estimating poses ...", self.on_mouse)

                images_top = np.hstack((color_image_copy, annotated_frame))
                images_bottom = np.hstack((color_image, depth_colormap))
                images = np.vstack((images_top, images_bottom))
                if(results[0].masks is not None):
                    self.draw_button(images, "Estimate pose", 150)
                cv2.imshow('Estimating poses ...', images)

                if(results[0].masks is not None and self.start_estimate):
                    mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1, 2, 0)
                    mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))
                    h2, w2, _ = results[0].orig_img.shape
                    mask = cv2.resize(mask_3channel, (w2, h2))
                    lower_black = np.array([0,0,0])
                    upper_black = np.array([0,0,1])
                    mask = cv2.inRange(mask, lower_black, upper_black)
                    mask = cv2.bitwise_not(mask)
                    pose = est.register(K=self.reader.K, rgb=color_image, depth=depth_image, ob_mask=mask, iteration=self.est_refine_iter)
                    self.track = True
            else:
                pose = est.track_one(rgb=color_image, depth=depth_image, K=self.reader.K, iteration=self.track_refine_iter)
                self.center_pose = pose@np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(self.reader.K, img=color_image, ob_in_cam=self.center_pose, bbox=bbox)
                vis = draw_xyz_axis(color_image, ob_in_cam=self.center_pose, scale=0.1, K=self.reader.K, thickness=3, transparency=0, is_input_rgb=True)
                
                images_top = np.hstack((vis, annotated_frame))
                images_bottom = np.hstack((color_image_copy, depth_colormap))
                images = np.vstack((images_top, images_bottom))
                self.draw_button(images, "Grip brick", 100)

                cv2.imshow('Estimating poses ...', images)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break

        cv2.destroyAllWindows()

PoseEstimatorApp().run()
