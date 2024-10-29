import cv2
import numpy as np
from estimater import *
from datareader import *
import pyrealsense2 as rs
from ultralytics import YOLO
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
import roboticstoolbox as rtb
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.experimental.motion_utils import reset_joints_to
from pybullet_collision_check import get_gripping_points
import queue
import threading
from skimage.metrics import structural_similarity as ssim
from RealSenseReader import *

class PoseEstimatorApp:
    def __init__(self):
        self.code_dir = os.path.dirname(os.path.realpath(__file__))
        self.test_scene_dir = f'{self.code_dir}/out'
        self.est_refine_iter = 5
        self.track_refine_iter = 2
        self.debug = 1
        self.debug_dir = f'{code_dir}/debug'
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.maskModel = YOLO('/home/mrenz/runs/segment/train15/weights/best.pt')
        self.start_estimate = False
        self.robot_interface = FrankaInterface(config_root + "/charmander.yml", use_visualizer=False)
        self.glctx = dr.RasterizeCudaContext()
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.reader = RealSenseReader()

        self.mesh_4x2 = trimesh.load(f'{self.code_dir}/out/mesh/4x2_brick.obj')
        self.mesh_2x2 = trimesh.load(f'{self.code_dir}/out/mesh/2x2_brick.obj')

        self.gpu_lock = threading.Lock()
    
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
            button_width, button_height = (150, 50)
            if button_x < x < button_x + button_width and button_y < y < button_y + button_height:
                self.start_estimate = True

    def adjust_object_frame(self, T_BO):
        R_y_neg_90 = np.array([
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1]
                    ])
        T_BO_prime = T_BO @ R_y_neg_90
        return T_BO_prime
    
    def grasp(self, grasp):
        self.robot_interface.control(
            controller_type="OSC_POSE",
            action=[0.0]*6 + [grasp],
            controller_cfg=get_default_controller_config("OSC_POSE")
        )

    def move_last_bit(self, scale_x, scale_y, scale_z, gripper_open):
        def should_continue(current, target, scale):
            if scale > 0:
                return current < target
            else:
                return current > target
            
        has_failed = False

        i = 0
        pos = self.robot_interface.last_eef_rot_and_pos[1].flatten()
        target_pos = pos + [scale_x, scale_y, scale_z]

        while should_continue(pos[2], target_pos[2], scale_z) or should_continue(pos[1], target_pos[1], scale_y) or should_continue(pos[0], target_pos[0], scale_x):
            vx = scale_x * (1 - np.abs(np.cos(np.pi * i / 100))) * 0.55
            vy = scale_y * (1 - np.abs(np.cos(np.pi * i / 100))) * 0.55
            vz = scale_z * (1 - np.abs(np.cos(np.pi * i / 100))) * 0.55

            action = [vx, vy, vz, 0, 0, 0, gripper_open]

            self.robot_interface.control(
                controller_type="CARTESIAN_VELOCITY",
                action=action,
                controller_cfg=get_default_controller_config("CARTESIAN_VELOCITY"),
            )

            #print(self.robot_interface.last_dq)

            if(self.robot_interface.last_gripper_q < 0.001):
                has_failed = True
                break

            if i == 100:
                break

            i += 1
            _, pos = self.robot_interface.last_eef_rot_and_pos
        
        return has_failed

    def grip_brick(self, T_base2object, wide_grip):

        ### gripper is actually not upside down in pybullet but transformation matrix is always... So this is necessary for now
        rotation_matrix = T_base2object[:3, :3]
        z_axis = rotation_matrix[:, 2]
        # check if upside down 
        if z_axis[2] > 0:
            rotation_180_x = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            T_base2object = T_base2object @ rotation_180_x

        translate_z = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -0.1],
            [0, 0, 0, 1]
        ])
        T_base2object = T_base2object @ translate_z

        rotation_180_z = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        T_base2object_mirrored = T_base2object @ rotation_180_z

        panda = rtb.models.Panda()
        ets = panda.ets()
        q_regular = ets.ik_LM(Tep=T_base2object, q0=self.robot_interface.last_q)[0]
        q_rotated = ets.ik_LM(Tep=T_base2object_mirrored, q0=self.robot_interface.last_q)[0]
        distance_regular = np.linalg.norm(np.array(self.robot_interface.last_q) - np.array(q_regular))
        distance_rotated = np.linalg.norm(np.array(self.robot_interface.last_q) - np.array(q_rotated))
        best_q = q_regular if distance_regular < distance_rotated else q_rotated

        translate_z = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.08],
            [0, 0, 0, 1]
        ])
        T_base2object = T_base2object @ translate_z
        translation_vector = T_base2object[:3, 3:]

        grip_width = 0.99 if wide_grip else 0.6

        if len(best_q) > 0:
            reset_joints_to(self.robot_interface, best_q, gripper_open=True, gripper_width=grip_width)
            _, current_pos = self.robot_interface.last_eef_rot_and_pos
            diff = (translation_vector - current_pos).flatten()
            self.move_last_bit(diff[0], diff[1], diff[2], -grip_width)
            has_failed = self.move_last_bit(-diff[0], -diff[1], -diff[2], grip_width)
            if(not has_failed):
                reset_joints_to(self.robot_interface, [
                                    0.55867454,
                                    0.32977858,
                                    0.46889284,
                                    -1.88649096,
                                    -0.28273366,
                                    2.19680637,
                                    1.47928339
                                ], gripper_open=False, gripper_width=grip_width)
                self.grasp(-1.0)
                reset_joints_to(self.robot_interface, [0.0704015130, 
                                                       -0.5273743065,
                                                       -0.0405501909,
                                                       -2.5585170688,
                                                       -0.0288302432, 
                                                       2.0835442059, 
                                                       0.8050347492], 
                                gripper_open=True, gripper_width=grip_width)
            else:
                reset_joints_to(self.robot_interface, [0.0704015130, 
                                                       -0.5273743065,
                                                       -0.0405501909,
                                                       -2.5585170688,
                                                       -0.0288302432, 
                                                       2.0835442059, 
                                                       0.8050347492], 
                                gripper_open=True, gripper_width=grip_width)
            
        return has_failed

    def worker_thread(self, tid, task_queue, result_queue, init_event):

        print(f"Thread {tid} initializing...")
        if tid == 0:
            mesh = self.mesh_4x2
            est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=self.scorer, refiner=self.refiner, debug_dir=self.debug_dir, debug=self.debug, glctx=self.glctx)
            to_origin = trimesh.bounds.oriented_bounds(mesh)[0]
            extents = trimesh.bounds.oriented_bounds(mesh)[1]
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
            size = "4x2"
        elif tid == 1:
            mesh = self.mesh_2x2
            est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=self.scorer, refiner=self.refiner, debug_dir=self.debug_dir, debug=self.debug, glctx=self.glctx)
            to_origin = trimesh.bounds.oriented_bounds(mesh)[0]
            extents = trimesh.bounds.oriented_bounds(mesh)[1]
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
            size = "2x2"
        else:
            raise ValueError("task id must be either 0 or 1.") 
         
        init_event.set()

        while True:
            task, completion_event = task_queue.get()
            if task is None: 
                task_queue.task_done()
                task_queue.put((None, None))
                break
            
            shape, mask, color_image, depth_image, color = task
            mask_raw = mask.cpu().data.numpy().transpose(1, 2, 0)
            mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))
            h2, w2, _ = shape
            mask = cv2.resize(mask_3channel, (w2, h2))
            mask = cv2.inRange(mask, np.array([0,0,0]), np.array([0,0,1]))
            mask = cv2.bitwise_not(mask)
            with self.gpu_lock:
                pose = est.register(K=self.reader.K, rgb=color_image, depth=depth_image, ob_mask=mask, iteration=self.est_refine_iter)
            center_pose = pose@np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(self.reader.K, img=color_image, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color_image, ob_in_cam=center_pose, scale=0.1, K=self.reader.K, thickness=3, transparency=0, is_input_rgb=True)
           
            completion_event.set()
            result_queue.put(([vis, center_pose, size, color, mask], completion_event))
            print(f"{color} {size} brick processed")

            task_queue.task_done()

    def compute_image_difference(self, image1, image2, mask):
        kernel = np.ones((8, 8), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image1 = np.where(mask == 255, 0, image1)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        image2 = np.where(mask == 255, 0, image2)

        epsilon = 1e-10  
        ssim_score, diff = ssim(image1, image2, full=True, data_range=image1.max() - image1.min() + epsilon)
        diff = (diff * 255).astype(np.uint8)

        _, ssim_image_thresholded = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        white_image = np.full_like(image1, 255, dtype=np.uint8)
        ssim_score, diff = ssim(ssim_image_thresholded, white_image, full=True)

        print("SSIM score: {:.4f}".format(ssim_score))

        return ssim_score
    
    def draw_image_grid(self, image1, image2, image3, image4, draw_button):
        images_top = np.hstack((image1, image2))
        images_bottom = np.hstack((image3, image4))
        images = np.vstack((images_top, images_bottom))
        if(draw_button):
            self.draw_button(images, "Estimate pose", 150)
        cv2.imshow('Estimating poses ...', images)
        cv2.waitKey(25)

    def initialize_threads(self, num_threads):
        queues = [queue.Queue() for _ in range(num_threads)]
        result_queue = queue.Queue()
        init_events = [threading.Event() for _ in range(num_threads)]
        threads = [threading.Thread(target=self.worker_thread, args=(i, queues[i], result_queue, init_events[i])) for i in range(num_threads)]
        for t in threads:
            t.start()

        for event in init_events:
            event.wait()

        print(f"Pose estimation threads initialized.")

        return queues, result_queue

    def run(self):

        queues, result_queue = self.initialize_threads(num_threads=2)

        while True:
            color_image, depth_image, depth_colormap = self.reader.capture_image()
            color_image_copy = color_image.copy()

            registered_bricks = self.maskModel(color_image, iou=0.9)[0]    

            annotated_frame = registered_bricks.plot()

            cv2.namedWindow("Estimating poses ...")
            cv2.setMouseCallback("Estimating poses ...", self.on_mouse)

            if not self.start_estimate:
                draw_button = registered_bricks.masks is not None
                self.draw_image_grid(color_image_copy, annotated_frame, color_image, depth_colormap, draw_button)

            else:
                print("Bricks registered: ", len(registered_bricks))

                self.draw_image_grid(color_image_copy, annotated_frame, color_image, depth_colormap, False)

                if not registered_bricks:
                    self.start_estimate = False
                else:
                    shape = registered_bricks.orig_img.shape

                    completion_events = [threading.Event() for _ in registered_bricks]

                    for i in range(len(registered_bricks)):
                        brick_class = self.maskModel.names[int(registered_bricks.boxes[i].cls)]
                        size = brick_class[:3]
                        color = brick_class[4:]
                        mask = registered_bricks.masks[i]

                        if size == "4x2":
                            queues[0].put(([shape, mask, color_image, depth_image, color], completion_events[i]))
                        elif size == "2x2":
                            queues[1].put(([shape, mask, color_image, depth_image, color], completion_events[i]))

                    T_base2gripper = None
                    while T_base2gripper is None:
                        T_base2gripper = self.robot_interface.last_eef_pose

                    T_cam2gripper = np.load('T_cam2gripper.npy')

                    bricks = []

                    for event in completion_events:
                        event.wait() 
                        result, _ = result_queue.get()
                        vis, center_pose, size, color, mask = result
                        center_pose = np.dot(T_base2gripper, np.dot(np.array(T_cam2gripper), np.array(center_pose)))
                        bricks.append([center_pose, size, color, mask])
                        self.draw_image_grid(vis, annotated_frame, color_image_copy, depth_colormap, False)

                    print("Estimated all bricks")

                    ssim_score = 1

                    while(ssim_score > 0.9925):
                        image, _, _ = self.reader.capture_image()

                        if not bricks:
                            break
                        
                        # Sort so the highest z-values come first
                        bricks = sorted(
                            bricks,
                            key=lambda x: x[0][2, 3],
                            reverse=True 
                        )

                        collision_free_brick = get_gripping_points(bricks)

                        if not collision_free_brick:
                            break

                        index = collision_free_brick[4]
                        del bricks[index]

                        needs_wide_grip = collision_free_brick[1]
                        
                        has_failed = self.grip_brick(collision_free_brick[0], needs_wide_grip)
                        if has_failed:
                            break

                        image_after_grip, _, _ = self.reader.capture_image()

                        mask = collision_free_brick[3][3]
                        ssim_score = self.compute_image_difference(image, image_after_grip, mask)


if __name__ == '__main__':
    PoseEstimatorApp().run()

