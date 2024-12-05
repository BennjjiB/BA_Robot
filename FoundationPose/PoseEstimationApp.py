from RealSenseReader import *
import cv2
import numpy as np
from estimater import *
from pose_estimation_helper import *
from ultralytics import YOLO
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to
from pybullet_collision_check import get_gripping_points


class PoseEstimatorApp:
    def __init__(self, maskModelPath='/home/panda3/Desktop/Robot_BA/best.pt', sort_by_color: bool = True):
        # Directories
        self.code_dir = os.path.dirname(os.path.realpath(__file__))
        self.debug_dir = f'{code_dir}/debug'
        self.test_scene_dir = f'{self.code_dir}/out'

        # Models
        self.maskModel = YOLO(maskModelPath)
        self.robot_interface = FrankaInterface(
            config_root + "/charmander.yml", use_visualizer=False)
        self.glctx = dr.RasterizeCudaContext()
        self.reader = RealSenseReader()

        # Reset and init all offsets
        self.reset_estimation()
        self.sort_by_color = sort_by_color

        # Meshes and boxes
        self.mesh_4x2 = trimesh.load(f'{self.code_dir}/out/mesh/4x2_brick.obj')
        self.est4x2 = FoundationPose(model_pts=self.mesh_4x2.vertices, model_normals=self.mesh_4x2.vertex_normals,
                                     mesh=self.mesh_4x2, debug_dir=self.debug_dir, glctx=self.glctx)
        self.to_origin4x2 = trimesh.bounds.oriented_bounds(self.mesh_4x2)[0]
        self.extents4x2 = trimesh.bounds.oriented_bounds(self.mesh_4x2)[1]
        self.bbox4x2 = np.stack(
            [-self.extents4x2 / 2, self.extents4x2 / 2], axis=0).reshape(2, 3)

        self.mesh_2x2 = trimesh.load(f'{self.code_dir}/out/mesh/2x2_brick.obj')
        self.extents2x2 = trimesh.bounds.oriented_bounds(self.mesh_2x2)[1]
        self.bbox2x2 = np.stack(
            [-self.extents2x2 / 2, self.extents2x2 / 2], axis=0).reshape(2, 3)
        self.to_origin2x2 = trimesh.bounds.oriented_bounds(self.mesh_2x2)[0]
        self.est2x2 = FoundationPose(model_pts=self.mesh_2x2.vertices, model_normals=self.mesh_2x2.vertex_normals,
                                     mesh=self.mesh_2x2, debug_dir=self.debug_dir, glctx=self.glctx)

        self.T_cam2gripper = np.load('T_cam2gripper.npy')

    def reset_estimation(self):
        self.start_estimate = False
        self.offset_red = 0
        self.offset_orange = 0
        self.offset_yellow = 0
        self.offset_green = 0
        self.offset_blue = 0
        self.offset_left = 0
        self.offset_right = 0

    # --------------------------- UI ---------------------------
    def on_mouse(self, event, x, y):
        """
        Handles button presses to start the estimation or switch mode
        """
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
        cv2.rectangle(img, (button_x, button_y), (button_x +
                      button_width, button_y + button_height), button_color, -1)
        cv2.putText(img, button_text, (button_x + 10, button_y +
                    button_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

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

    # --------------------------- Sorting Bricks ---------------------------
    def get_sort_pose_by_color(
        self,
        color,
        T_base2object,
        original_pose,
        brick_is_upright,
        sorting_pose_left_color,
        sorting_pose_right_color,
        size,
        wide_grip,
        x_offset
    ):
        """
        Adjusts the sorting pose based on the brick's color, orientation, and size.
        This function modifies the sorting position matrices for bricks of different colors,
        taking into account offsets and adjustments for upright orientation, brick size, and grip width.
        """
        if brick_is_upright:
            dist_to_center = compute_transformation_distance(
                T_base2object, original_pose)
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

        return sorting_pose_left_color if color == "blue" or color == "green" else sorting_pose_right_color

    def get_sort_pose_by_default(
        self,
        T_base2object,
        original_pose,
        sorting_pose_left_size,
        sorting_pose_right_size,
        brick_is_upright,
        size,
        wide_grip,
        x_offset
    ):
        """
        Adjusts the sorting pose based on the brick's size, orientation, and grip configuration.
        """
        if brick_is_upright:
            dist_to_center = compute_transformation_distance(
                T_base2object, original_pose)
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

        return sorting_pose_left_size if size == "2x2" else sorting_pose_right_size

    def sort_brick(self, collision_free_brick):
        """
        Sorts a brick by picking it up and placing it in the correct position 
        based on its color, size, and orientation.

        Parameters:
            collision_free_brick (list): 
                A list containing information about the brick to be sorted, including:
                - collision_free_brick[0]: (4x4 numpy array) Transformation matrix of the brick in the base frame.
                - collision_free_brick[1]: (bool) Indicator if the brick requires a wide grip.
                - collision_free_brick[3][1]: (str) Size of the brick (e.g., "2x2" or "4x2").
                - collision_free_brick[3][2]: (str) Color of the brick (e.g., "red", "blue").
                - collision_free_brick[5]: (bool) Indicator if the brick is upright.
                - collision_free_brick[6]: (4x4 numpy array) Original pose of the brick.
                - collision_free_brick[8]: (float) Offset in the x-direction for sorting.

        Returns:
            bool: `True` if the operation fails (e.g., due to gripper issues), `False` otherwise.
        """
        T_base2object = collision_free_brick[0]
        wide_grip = collision_free_brick[1]
        color = collision_free_brick[3][2]
        size = collision_free_brick[3][1]
        brick_is_upright = collision_free_brick[5]
        original_pose = collision_free_brick[6]
        x_offset = collision_free_brick[8]

        T_base2object = adjust_brick_orientation(T_base2object)
        T_base2object_mirrored = T_base2object @ rotation_matrix_z(180)
        best_q, x_offset_sign, ets = compute_best_inverse_kinematic_solution(
            self.robot_interface, T_base2object, T_base2object_mirrored)
        x_offset *= x_offset_sign

        grip_width = adjust_gripper_and_position(
            self.robot_interface, wide_grip, best_q)

        T_base2object = T_base2object @ translation_matrix(0, 0, 0.095)
        translation_vector_target = T_base2object[:3, 3:]

        sorting_pose_right_color, sorting_pose_left_color, sorting_pose_right_size, sorting_pose_left_size, init_pose = getSortMatrices()
        if len(best_q) > 0:
            reset_joints_to(self.robot_interface, best_q,
                            gripper_open=True, gripper_width=grip_width)
            # Move down to brick
            target_pos_up, has_failed = move_straight_from_brick(
                self.robot_interface, translation_vector_target, -grip_width)
            do_grasp_action(self.robot_interface,  grip_width)
            time.sleep(0.3)
            # Move up again
            _, has_failed = move_straight_from_brick(
                self.robot_interface, target_pos_up, grip_width)

            if has_failed:
                print("failed moving to or from brick")
                reset_joints_to(self.robot_interface, init_pose,
                                gripper_open=True, gripper_width=grip_width)
                return has_failed

            if self.sort_by_color:
                sort_pose = self.get_sort_pose_by_color(color, T_base2object, original_pose, brick_is_upright,
                                                        sorting_pose_left_color, sorting_pose_right_color, size, wide_grip, x_offset)
            else:
                sort_pose = self.get_sort_pose_by_default(
                    T_base2object, original_pose, sorting_pose_left_size, sorting_pose_right_size, brick_is_upright, size, wide_grip, x_offset)

            # move to sorting position
            q_target = ets.ik_LM(
                Tep=sort_pose, q0=self.robot_interface.last_q)[0]
            reset_joints_to(self.robot_interface, q_target,
                            gripper_open=False, gripper_width=grip_width)

            move_straight_from_brick(
                self.robot_interface, target_pos=None, target_offset=[0, 0, -0.1], grip_width=grip_width)
            move_straight_from_brick(
                self.robot_interface, target_pos=None, target_offset=[0, 0, 0.1], grip_width=-0.9 if wide_grip else -0.7)
            # move to starting position
            reset_joints_to(self.robot_interface, init_pose,
                            gripper_open=True, gripper_width=grip_width)
        return has_failed

    # --------------------------- Brick Poses ---------------------------
    def get_brick_poses(
        self,
        registered_bricks,
        shape,
        color_image,
        depth_image,
        annotated_frame,
        color_image_copy,
        depth_colormap
    ):
        """
        Extract and estimate the 3D poses of bricks detected in the scene.

        Returns:
        list: A list of detected bricks, where each brick is represented as:
            [center_pose, size, color, mask, brick_class_id].
        """
        bricks = []
        T_base2gripper = None
        while T_base2gripper is None:
            T_base2gripper = self.robot_interface.last_eef_pose
        vis = color_image.copy()

        for i in range(len(registered_bricks)):
            mask, size, color, brick_class_id = create_brick_mask(
                self.maskModel, registered_bricks, i, shape[0], shape[1])
            if size == "4x2":
                pose = self.est4x2.register(
                    K=self.reader.K, rgb=color_image, depth=depth_image, ob_mask=mask)
                center_pose = pose@np.linalg.inv(self.to_origin4x2)
                draw_posed_3d_box(self.reader.K, img=vis,
                                  ob_in_cam=center_pose, bbox=self.bbox4x2)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.06,
                                    K=self.reader.K, thickness=3, transparency=0, is_input_rgb=True)
            elif size == "2x2":
                pose = self.est2x2.register(
                    K=self.reader.K, rgb=color_image, depth=depth_image, ob_mask=mask)
                center_pose = pose@np.linalg.inv(self.to_origin2x2)
                draw_posed_3d_box(self.reader.K, img=vis,
                                  ob_in_cam=center_pose, bbox=self.bbox2x2)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.06,
                                    K=self.reader.K, thickness=3, transparency=0, is_input_rgb=True)

            center_pose = np.dot(T_base2gripper, np.dot(
                np.array(self.T_cam2gripper), np.array(center_pose)))

            bricks.append([center_pose, size, color, mask, brick_class_id])

            self.draw_image_grid(vis, annotated_frame,
                                 color_image_copy, depth_colormap, False)

        return bricks


     # --------------------------- Sorting Pipeline ---------------------------
    def start_sorting_bricks_pipeline(self, bricks, registered_bricks):
        ssim_score = 1
        detections_coherent = True
        bricks_before = registered_bricks.boxes.cls

        while (ssim_score > 0.994 and detections_coherent):
            image, _, _ = self.reader.capture_image()

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
            ssim_score = compute_image_difference(
                image, image_after_grip, mask)
            registered_bricks_after = self.maskModel(
                image_after_grip, iou=0.9, conf=0.6)[0]
            removed_brick = collision_free_brick[3][4].item()
            bricks_before = bricks_before.tolist()
            bricks_before.remove(removed_brick)
            bricks_before = torch.tensor(sorted(bricks_before))
            bricks_after = torch.sort(
                registered_bricks_after.boxes.cls).values
            detections_coherent = torch.equal(
                bricks_before, bricks_after)

    def run(self):
        while True:
            color_image, depth_image, depth_colormap = self.reader.capture_image()
            color_image_clear = color_image.copy()

            registered_bricks = self.maskModel(color_image, iou=0.9)[0]

            annotated_frame = registered_bricks.plot()

            cv2.namedWindow("Estimating poses ...")
            cv2.setMouseCallback("Estimating poses ...", self.on_mouse)

            if cv2.getWindowProperty("Estimating poses ...", cv2.WND_PROP_VISIBLE) < 1:
                break

            if not self.start_estimate:
                draw_button = registered_bricks.masks is not None
                self.draw_image_grid(
                    color_image_clear, annotated_frame, color_image, depth_colormap, draw_button)
            else:
                self.draw_image_grid(
                    color_image_clear, annotated_frame, color_image, depth_colormap, False)

                if not registered_bricks:
                    self.reset_estimation()
                else:
                    bricks = self.get_brick_poses(
                        registered_bricks, registered_bricks.orig_img.shape, color_image,
                        depth_image, annotated_frame, color_image_clear, depth_colormap)
                    if not bricks:
                        break
                    self.start_sorting_bricks_pipeline(
                        bricks, registered_bricks)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    PoseEstimatorApp().run()
