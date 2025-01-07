import cv2
import numpy as np
from estimater import *
from ultralytics import YOLO
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
import roboticstoolbox as rtb
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.experimental.motion_utils import reset_joints_to
from helper_functions import *
from pybullet_collision_check import get_gripping_points
from foundation_pose.real_sense_reader import *

class PoseEstimatorApp:
    def __init__(self):
        self.code_dir = os.path.dirname(os.path.realpath(__file__))
        self.test_scene_dir = f'{self.code_dir}/out'
        self.debug_dir = f'{code_dir}/debug'
        self.maskModel = YOLO('/home/panda3/Desktop/Robot_BA/best.pt')
        self.start_estimate = False
        self.robot_interface = FrankaInterface(config_root + "/charmander.yml", use_visualizer=False)
        self.glctx = dr.RasterizeCudaContext()
        self.reader = RealSenseReader()

        # sets the sorting offset relative to the base sorting position for each color
        self.offset_red = 0
        self.offset_orange = 0
        self.offset_yellow = 0
        self.offset_green = 0
        self.offset_blue = 0

        # sets the sorting offset relative to the base sorting position for both sizes
        self.offset_left = 0
        self.offset_right = 0

        self.sort_by_color = True

        self.mesh_4x2 = trimesh.load(f'{self.code_dir}/out/mesh/4x2_brick.obj')
        self.est4x2 = FoundationPose(model_pts=self.mesh_4x2.vertices, model_normals=self.mesh_4x2.vertex_normals, mesh=self.mesh_4x2, debug_dir=self.debug_dir, glctx=self.glctx)
        self.to_origin4x2 = trimesh.bounds.oriented_bounds(self.mesh_4x2)[0]
        self.extents4x2 = trimesh.bounds.oriented_bounds(self.mesh_4x2)[1]
        self.bbox4x2 = np.stack([-self.extents4x2 /2, self.extents4x2 /2], axis=0).reshape(2,3)

        self.mesh_2x2 = trimesh.load(f'{self.code_dir}/out/mesh/2x2_brick.obj')
        self.extents2x2 = trimesh.bounds.oriented_bounds(self.mesh_2x2)[1]
        self.bbox2x2 = np.stack([-self.extents2x2 /2, self.extents2x2 /2], axis=0).reshape(2,3)
        self.to_origin2x2 = trimesh.bounds.oriented_bounds(self.mesh_2x2)[0]
        self.est2x2 = FoundationPose(model_pts=self.mesh_2x2.vertices, model_normals=self.mesh_2x2.vertex_normals, mesh=self.mesh_2x2, debug_dir=self.debug_dir, glctx=self.glctx)

        self.T_cam2gripper = np.load('T_cam2gripper.npy')

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            button_est_x, button_est_y = (20, 20)
            button_switch_x, button_switch_y = (20, 80)
            button_width, button_height = (150, 50)
            if button_est_x < x < button_est_x + button_width and button_est_y < y < button_est_y + button_height:
                self.start_estimate = True
            if button_switch_x < x < button_switch_x + button_width and button_switch_y < y < button_switch_y + button_height:
                self.sort_by_color = not self.sort_by_color

    def draw_button(self, img, text, width, y):
        text_color = (255, 255, 255)
        button_color = (0, 0, 0)
        button_text = text
        button_x, button_y = (20, y)
        button_width, button_height = (width, 50)
        cv2.rectangle(img, (button_x, button_y), (button_x + button_width, button_y + button_height), button_color, -1)
        cv2.putText(img, button_text, (button_x + 10, button_y + button_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    def draw_image_grid(self, image1, image2, image3, image4, draw_button):
        images_top = np.hstack((image1, image2))
        images_bottom = np.hstack((image3, image4))
        images = np.vstack((images_top, images_bottom))
        if draw_button:
            self.draw_button(images, "Estimate pose", 150, 20)
            if self.sort_by_color:
                self.draw_button(images, "Sorting by color", 150, 80)
            else:
                self.draw_button(images, "Sorting by size", 150, 80)
        cv2.imshow('Estimating poses ...', images)
        cv2.waitKey(25)
    
    def grasp(self, grasp):
        self.robot_interface.control(
            controller_type="CARTESIAN_VELOCITY",
            action=[0.0]*6 + [grasp],
            controller_cfg=get_default_controller_config("CARTESIAN_VELOCITY")
        )

    #lets gripper move in straight line
    def move_last_bit(self, scale_x, scale_y, scale_z, gripper_open):
        has_failed = False

        dt = 0.05 
        T = 4.5
        N = int(T / dt)
        
        v_max_x = (2 * scale_x) / T
        v_max_y = (2 * scale_y) / T
        v_max_z = (2 * scale_z) / T
        
        t_array = np.linspace(0, T, N+1)

        for t in t_array:
            vx = v_max_x * (1 - np.cos(2 * np.pi * t / T)) / 2
            vy = v_max_y * (1 - np.cos(2 * np.pi * t / T)) / 2
            vz = v_max_z * (1 - np.cos(2 * np.pi * t / T)) / 2

            action = [vx, vy, vz, 0, 0, 0, gripper_open]

            self.robot_interface.control(
                controller_type="CARTESIAN_VELOCITY",
                action=action,
                controller_cfg=get_default_controller_config("CARTESIAN_VELOCITY"),
            )

            if self.robot_interface.last_gripper_q < 0.001 :
                has_failed = True
                break

            time.sleep(0.05)
        
        return has_failed
    
    # computes ditance between the translation vectors of two trnasformation matrices
    def compute_transformation_distance(self, T1, T2):
        t1 = T1[:3, 3]
        t2 = T2[:3, 3]

        translation_distance = np.linalg.norm(t1 - t2)
        
        return translation_distance

    def sort_brick(self, collsion_free_brick):
        T_base2object = collsion_free_brick[0]
        wide_grip = collsion_free_brick[1]
        color = collsion_free_brick[3][2]
        size = collsion_free_brick[3][1]
        brick_is_upright = collsion_free_brick[5]
        original_pose = collsion_free_brick[6]
        x_offset = collsion_free_brick[8]

        rotation_matrix = T_base2object[:3, :3]
        z_axis = rotation_matrix[:, 2]

        # check if upside down 
        if z_axis[2] > 0:
            T_base2object = T_base2object @ rotation_matrix_x(180)

        T_base2object = T_base2object @ translation_matrix(0, 0, -0.1)

        T_base2object_mirrored = T_base2object @ rotation_matrix_z(180)

        # inverse kienamtics
        panda = rtb.models.Panda()
        ets = panda.ets()
        q_regular = ets.ik_LM(Tep=T_base2object, q0=self.robot_interface.last_q)[0]
        q_rotated = ets.ik_LM(Tep=T_base2object_mirrored, q0=self.robot_interface.last_q)[0]
        distance_regular = np.linalg.norm(np.array(self.robot_interface.last_q) - np.array(q_regular))
        distance_rotated = np.linalg.norm(np.array(self.robot_interface.last_q) - np.array(q_rotated))
        if distance_regular < distance_rotated:
            best_q = q_regular
        else:
            best_q = q_rotated
            x_offset = -x_offset

        T_base2object = T_base2object @ translation_matrix(0, 0, 0.095)
        translation_vector_target = T_base2object[:3, 3:]

        grip_width = 0.99 if wide_grip else 0.6

        sorting_pose_right_color = [[ 0.11089964,  1.0, -0.02814022,  0.22],
                            [ 1.0, -0.11044599,  0.0171983,   0.49670671],
                            [ 0.01397722, -0.02987089, -1.0,  0.12],
                            [ 0.,          0.,          0.,          1.]]
    
        sorting_pose_left_color = [[ 0.0229358,  -1.0,  0.00338916,  0.25516519],
                            [-1.0, -0.02307881, -0.05096378, -0.48184099],
                            [ 0.05102781, -0.00221492, -1.0,  0.124],
                            [ 0. ,         0. ,         0. ,         1.  ]]
        
        sorting_pose_right_size = [[ 0.11089964,  1.0, -0.02814022,  0.22],
                            [ 1.0, -0.11044599,  0.0171983,   0.40],
                            [ 0.01397722, -0.02987089, -1.0,  0.12],
                            [ 0.,          0.,          0.,          1.]]
        sorting_pose_left_size = [[ 0.0229358,  -1.0,  0.00338916,  0.25516519],
                            [-1.0, -0.02307881, -0.05096378, -0.42],
                            [ 0.05102781, -0.00221492, -1.0,  0.124],
                            [ 0. ,         0. ,         0. ,         1.  ]]
        
        init_pose = [0.1232563209, -0.0022815667, -0.0932499246, -2.1356946695, -0.0281708669, 2.0154906706, 0.8219482610]

        if len(best_q) > 0:
            reset_joints_to(self.robot_interface, best_q, gripper_open=True, gripper_width=grip_width)

            # move straight down to brick
            _, current_pos = self.robot_interface.last_eef_rot_and_pos
            diff = translation_vector_target.flatten() - current_pos.flatten()
            target_pos_up = current_pos.flatten() - 0.5 * diff
            distance = np.linalg.norm(diff)
            while distance > 0.01:
                has_failed = self.move_last_bit(diff[0], diff[1], diff[2], -grip_width)
                _, current_pos = self.robot_interface.last_eef_rot_and_pos
                diff = translation_vector_target.flatten() - current_pos.flatten()
                distance = np.linalg.norm(diff)
                if has_failed:
                    break

            self.grasp(grip_width)
            time.sleep(0.3)

            # move straight up from brick
            _, current_pos = self.robot_interface.last_eef_rot_and_pos
            diff = target_pos_up - current_pos.flatten()
            distance = np.linalg.norm(diff)
            while distance > 0.01:
                has_failed = self.move_last_bit(diff[0], diff[1], diff[2], grip_width)
                _, current_pos = self.robot_interface.last_eef_rot_and_pos
                diff = target_pos_up - current_pos.flatten()
                distance = np.linalg.norm(diff)
                if has_failed:
                    break

            if not has_failed:
                if self.sort_by_color: 
                    if brick_is_upright:
                        dist_to_center = self.compute_transformation_distance(T_base2object, original_pose)
                        if color == "green" or color == "blue":
                            dist_to_center += 0.012
                        else:
                            dist_to_center += 0.01
                        sorting_pose_left_color[2][3] += dist_to_center
                        sorting_pose_right_color[2][3] += dist_to_center
                    if color == "red":
                        if self.offset_red > 0:
                            self.offset_red += 0.03 if size == "2x2" or wide_grip else 0.05
                        sorting_pose_right_color[1][3] -= self.offset_red
                        sorting_pose_right_color[1][3] += x_offset

                        self.offset_red += 0.017 if size == "2x2" or wide_grip else 0.035
                    elif color == "orange":
                        sorting_pose_right_color[0][3] += 0.15
                        if self.offset_orange > 0:
                            self.offset_orange += 0.03 if size == "2x2" or wide_grip else 0.05
                        sorting_pose_right_color[1][3] -= self.offset_orange
                        sorting_pose_right_color[1][3] += x_offset

                        self.offset_orange += 0.017 if size == "2x2" or wide_grip else 0.035
                    elif color == "yellow":
                        sorting_pose_right_color[0][3] += 0.3
                        if self.offset_yellow > 0:
                            self.offset_yellow += 0.03 if size == "2x2" or wide_grip else 0.05
                        sorting_pose_right_color[1][3] -= self.offset_yellow
                        sorting_pose_right_color[1][3] += x_offset

                        self.offset_yellow += 0.017 if size == "2x2" or wide_grip else 0.035
                    elif color == "green":
                        if self.offset_green > 0:
                            self.offset_green += 0.03 if size == "2x2" or wide_grip else 0.05
                        sorting_pose_left_color[1][3] += self.offset_green
                        sorting_pose_left_color[1][3] -= x_offset

                        self.offset_green += 0.017 if size == "2x2" or wide_grip else 0.035
                    elif color == "blue":
                        sorting_pose_left_color[0][3] += 0.2
                        if self.offset_blue > 0:
                            self.offset_blue += 0.03 if size == "2x2" or wide_grip else 0.05
                        sorting_pose_left_color[1][3] += self.offset_blue
                        sorting_pose_left_color[1][3] -= x_offset

                        self.offset_blue += 0.017 if size == "2x2" or wide_grip else 0.035
                    
                    sort_pose = sorting_pose_left_color if color == "blue" or color == "green" else sorting_pose_right_color
                else:
                    if brick_is_upright:
                        dist_to_center = self.compute_transformation_distance(T_base2object, original_pose)
                        dist_to_center += 0.012
                        sorting_pose_left_size[2][3] += dist_to_center
                        sorting_pose_right_size[2][3] += dist_to_center
                    if size == "2x2":
                        if self.offset_left > 0:
                            self.offset_left += 0.045
                        sorting_pose_left_size[0][3] += self.offset_left
                        sorting_pose_left_size[1][3] -= x_offset

                        self.offset_left += 0.015
                    elif size == "4x2":
                        if self.offset_right > 0:
                            self.offset_right += 0.045 if not wide_grip else 0.065
                        sorting_pose_right_size[0][3] += self.offset_right
                        sorting_pose_right_size[1][3] += x_offset

                        self.offset_right += 0.015 if not wide_grip else 0.04
                    
                    sort_pose = sorting_pose_left_size if size == "2x2" else sorting_pose_right_size

                # move to sorting position
                q_target = ets.ik_LM(Tep=sort_pose, q0=self.robot_interface.last_q)[0]
                reset_joints_to(self.robot_interface, q_target, gripper_open=False, gripper_width=grip_width)

                # move straight down
                _, current_pos = self.robot_interface.last_eef_rot_and_pos
                target_pos = current_pos.flatten() + [0,0,-0.1]
                diff = target_pos - current_pos.flatten()
                distance = np.linalg.norm(diff)
                while distance > 0.01:
                    self.move_last_bit(diff[0], diff[1], diff[2], grip_width)
                    _, current_pos = self.robot_interface.last_eef_rot_and_pos
                    diff = target_pos - current_pos.flatten()
                    distance = np.linalg.norm(diff)
                
                # move straight up
                _, current_pos = self.robot_interface.last_eef_rot_and_pos
                target_pos = current_pos.flatten() + [0,0,0.1]
                diff = target_pos - current_pos.flatten()
                distance = np.linalg.norm(diff)
                open_grip = -0.9 if wide_grip else -0.7
                while distance > 0.01:
                    self.move_last_bit(diff[0], diff[1], diff[2], open_grip)
                    _, current_pos = self.robot_interface.last_eef_rot_and_pos
                    diff = target_pos - current_pos.flatten()
                    distance = np.linalg.norm(diff)

                # move to starting poisiton
                reset_joints_to(self.robot_interface, init_pose, 
                                gripper_open=True, gripper_width=grip_width)
            else:
                reset_joints_to(self.robot_interface, init_pose, 
                                gripper_open=True, gripper_width=grip_width)
            
        return has_failed

    def get_brick_poses(self, registered_bricks, shape, color_image, depth_image, annotated_frame, color_image_copy, depth_colormap):
        bricks = []

        h2, w2, _ = shape

        T_base2gripper = None
        while T_base2gripper is None:
            T_base2gripper = self.robot_interface.last_eef_pose
        
        vis = color_image.copy()

        for i in range(len(registered_bricks)):
            brick_class = self.maskModel.names[int(registered_bricks.boxes[i].cls)]
            brick_class_id = registered_bricks.boxes[i].cls
            size = brick_class[:3]
            color = brick_class[4:]

            mask = registered_bricks.masks[i].cpu().data.numpy().transpose(1, 2, 0)
            mask = cv2.merge((mask,mask,mask))
            mask = cv2.resize(mask, (w2, h2))
            mask = cv2.inRange(mask, np.array([0,0,0]), np.array([0,0,1]))
            mask = cv2.bitwise_not(mask)

            if size == "4x2":
                pose = self.est4x2.register(K=self.reader.K, rgb=color_image, depth=depth_image, ob_mask=mask)
                center_pose = pose@np.linalg.inv(self.to_origin4x2)
                draw_posed_3d_box(self.reader.K, img=vis, ob_in_cam=center_pose, bbox=self.bbox4x2)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.06, K=self.reader.K, thickness=3, transparency=0, is_input_rgb=True)
            elif size == "2x2":
                pose = self.est2x2.register(K=self.reader.K, rgb=color_image, depth=depth_image, ob_mask=mask)
                center_pose = pose@np.linalg.inv(self.to_origin2x2)
                draw_posed_3d_box(self.reader.K, img=vis, ob_in_cam=center_pose, bbox=self.bbox2x2)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.06, K=self.reader.K, thickness=3, transparency=0, is_input_rgb=True)

            center_pose = np.dot(T_base2gripper, np.dot(np.array(self.T_cam2gripper), np.array(center_pose)))      

            bricks.append([center_pose, size, color, mask, brick_class_id])

            self.draw_image_grid(vis, annotated_frame, color_image_copy, depth_colormap, False)

        return bricks 

    def run(self):
        while True:
            color_image, depth_image, depth_colormap = self.reader.capture_image()
            color_image_clear = color_image.copy()

            registered_bricks = self.maskModel(color_image, iou=0.9)[0]    

            annotated_frame = registered_bricks.plot()
            print("Hey")
            cv2.setMouseCallback("Estimating poses ...", self.on_mouse)

            if cv2.getWindowProperty("Estimating poses ...", cv2.WND_PROP_VISIBLE) < 1:
                break

            if not self.start_estimate:
                draw_button = registered_bricks.masks is not None
                self.draw_image_grid(color_image_clear, annotated_frame, color_image, depth_colormap, draw_button)

            else:
                print("Bricks registered: ", len(registered_bricks))

                self.draw_image_grid(color_image_clear, annotated_frame, color_image, depth_colormap, False)

                if not registered_bricks:
                    self.start_estimate = False
                    self.offset_red = 0
                    self.offset_orange = 0
                    self.offset_yellow = 0
                    self.offset_green = 0
                    self.offset_blue = 0
                    self.offset_left = 0
                    self.offset_right = 0

                else:
                    bricks = self.get_brick_poses(registered_bricks, 
                                                  registered_bricks.orig_img.shape, 
                                                  color_image, depth_image, 
                                                  annotated_frame, 
                                                  color_image_clear, 
                                                  depth_colormap)

                    print("Estimated all bricks")

                    ssim_score = 1
                    detections_coherent = True
                    bricks_before = registered_bricks.boxes.cls

                    while(ssim_score > 0.994 and detections_coherent):
                        image, _, _ = self.reader.capture_image()

                        if not bricks:
                            break

                        collision_free_brick = get_gripping_points(bricks)

                        if not collision_free_brick:
                            break

                        index = collision_free_brick[4]
                        del bricks[index]

                        
                        has_failed = self.sort_brick(collision_free_brick)
                        if has_failed:
                            break

                        start_time = time.time()
                        while time.time() - start_time < 0.2:
                            image_after_grip, _, _ = self.reader.capture_image()

                        mask = collision_free_brick[3][3]
                        ssim_score = compute_image_difference(image, image_after_grip, mask)
                        registered_bricks_after = self.maskModel(image_after_grip, iou=0.9, conf=0.6)[0] 
                        removed_brick = collision_free_brick[3][4].item()
                        bricks_before = bricks_before.tolist()
                        bricks_before.remove(removed_brick)
                        bricks_before = torch.tensor(sorted(bricks_before))
                        bricks_after = torch.sort(registered_bricks_after.boxes.cls).values
                        detections_coherent = torch.equal(bricks_before, bricks_after)


        cv2.destroyAllWindows()

if __name__ == '__main__':
    PoseEstimatorApp().run()

