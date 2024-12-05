import time
import numpy as np
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.experimental.motion_utils import reset_joints_to
import roboticstoolbox as rtb
import cv2


def move_last_bit(robot_interface: FrankaInterface, scale_x, scale_y, scale_z, gripper_open):
    """
    Moves the robot's gripper along a straight line using a smooth velocity profile.

    The function generates a sinusoidal velocity profile for motion in the x, y, and z directions,
    ensuring smooth acceleration and deceleration over the duration of the motion. It also controls
    the state of the gripper (open or closed) during the motion.

    Parameters:
    ----------
    scale_x : float
        Scaling factor for the motion along the x-axis, representing the total displacement in meters.
    scale_y : float
        Scaling factor for the motion along the y-axis, representing the total displacement in meters.
    scale_z : float
        Scaling factor for the motion along the z-axis, representing the total displacement in meters.
    gripper_open : bool or float
        The desired state of the gripper during motion. A value of `True` or `1.0` typically indicates
        an open gripper, while `False` or `0.0` indicates a closed gripper.

    Returns:
    -------
    bool
        Returns `True` if the motion fails (e.g., gripper state becomes invalid), otherwise `False`.

    Example:
    -------
    To move the gripper in a straight line by 0.1 m in the x-direction, 0.05 m in the y-direction,
    and 0.2 m in the z-direction while keeping the gripper open:

    >>> move_last_bit(scale_x=0.1, scale_y=0.05, scale_z=0.2, gripper_open=True)
    """
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

        robot_interface.control(
            controller_type="CARTESIAN_VELOCITY",
            action=action,
            controller_cfg=get_default_controller_config("CARTESIAN_VELOCITY")
        )

        if robot_interface.last_gripper_q < 0.001:
            has_failed = True
            break

        time.sleep(0.05)

    return has_failed


def do_grasp_action(robot_interface, grasp):
    """
    Command the robot to perform a grasping action using Cartesian velocity control.
    """
    robot_interface.control(
        controller_type="CARTESIAN_VELOCITY",
        action=[0.0]*6 + [grasp],
        controller_cfg=get_default_controller_config("CARTESIAN_VELOCITY")
    )


def adjust_gripper_and_position(robot_interface, wide_grip, best_q):
    """
    Adjust the robot's gripper width and position based on the provided configuration.

    Parameters:
    robot_interface (object): Interface for controlling the robot.
    wide_grip (bool): Indicates whether to use a wide grip (True) or a narrow grip (False).
    best_q (list): Joint configuration to reset the robot to.

    Returns:
    float: The gripper width set for the operation.
    """
    grip_width = 0.99 if wide_grip else 0.6
    if len(best_q) > 0:
        reset_joints_to(robot_interface, best_q,
                        gripper_open=True, gripper_width=grip_width)
    return grip_width


def move_straight_from_brick(robot_interface, target_pos, grip_width, target_offset=None):
    """
    Move the robot's end-effector straight from a brick to a target position.

    Parameters:
    robot_interface (FrankaInterface): The robot interface used to control the robot.
    target_pos (array-like): The target position for the end-effector.
    grip_width (float): The width of the gripper during movement.
    target_offset (array-like, optional): Offset to apply to the current position to calculate the target position.

    Returns:
    tuple:
        - opposite_target (array-like): The position opposite to the target position relative to the current position.
        - has_failed (bool): Indicates whether the movement operation failed.
    """
    _, current_pos = robot_interface.last_eef_rot_and_pos
    if target_offset:
        target_pos = current_pos.flatten() + target_offset
    diff = target_pos - current_pos.flatten()
    distance = np.linalg.norm(diff)
    opposite_target = current_pos.flatten() - 0.5 * diff
    while distance > 0.01:
        has_failed = move_last_bit(
            robot_interface, diff[0], diff[1], diff[2], grip_width)
        _, current_pos = robot_interface.last_eef_rot_and_pos
        diff = target_pos - current_pos.flatten()
        distance = np.linalg.norm(diff)
        if has_failed:
            break
    return opposite_target, has_failed


def compute_best_inverse_kinematic_solution(robot_interface: FrankaInterface, T_base2object, T_base2object_mirrored):
    """
    Compute the best inverse kinematic solution for reaching an object.

    Parameters:
    robot_interface (FrankaInterface): The robot interface used for accessing the robot's state and controls.
    T_base2object (array-like): The transformation matrix from the base to the object.
    T_base2object_mirrored (array-like): The mirrored transformation matrix from the base to the object.

    Returns:
    tuple: A tuple containing:
        - best_q (array-like): The joint configuration that minimizes the distance to the target.
        - x_offset_sign (int): A sign indicator (+1 or -1) based on the closer solution.
        - ets (rtb.ETS): The Elementary Transform Sequence representation of the robot.
    """
    panda = rtb.models.Panda()
    ets = panda.ets()
    q_regular = ets.ik_LM(
        Tep=T_base2object, q0=robot_interface.last_q)[0]
    q_rotated = ets.ik_LM(Tep=T_base2object_mirrored,
                          q0=robot_interface.last_q)[0]

    distance_regular = np.linalg.norm(
        np.array(robot_interface.last_q) - np.array(q_regular))
    distance_rotated = np.linalg.norm(
        np.array(robot_interface.last_q) - np.array(q_rotated))
    best_q = q_regular if distance_regular < distance_rotated else q_rotated
    x_offset_sign = 1 if distance_regular < distance_rotated else -1
    return best_q, x_offset_sign, ets


def get_sort_matrices():
    """
    Provides predefined transformation matrices and an initial pose for sorting operations.

    Returns:
    tuple: Contains the following:
        - sorting_pose_right_color (list[list[float]]): 4x4 transformation matrix for sorting objects by color on the right side.
        - sorting_pose_left_color (list[list[float]]): 4x4 transformation matrix for sorting objects by color on the left side.
        - sorting_pose_right_size (list[list[float]]): 4x4 transformation matrix for sorting objects by size on the right side.
        - sorting_pose_left_size (list[list[float]]): 4x4 transformation matrix for sorting objects by size on the left side.
        - init_pose (list[float]): A 7-element list defining the initial pose.
    """
    sorting_pose_right_color = [[0.11089964,  1.0, -0.02814022,  0.22],
                                [1.0, -0.11044599,  0.0171983,   0.49670671],
                                [0.01397722, -0.02987089, -1.0,  0.12],
                                [0.,          0.,          0.,          1.]]

    sorting_pose_left_color = [[0.0229358,  -1.0,  0.00338916,  0.25516519],
                               [-1.0, -0.02307881, -
                                   0.05096378, -0.48184099],
                               [0.05102781, -0.00221492, -1.0,  0.124],
                               [0.,         0.,         0.,         1.]]

    sorting_pose_right_size = [[0.11089964,  1.0, -0.02814022,  0.22],
                               [1.0, -0.11044599,  0.0171983,   0.40],
                               [0.01397722, -0.02987089, -1.0,  0.12],
                               [0.,          0.,          0.,          1.]]
    sorting_pose_left_size = [[0.0229358,  -1.0,  0.00338916,  0.25516519],
                              [-1.0, -0.02307881, -0.05096378, -0.42],
                              [0.05102781, -0.00221492, -1.0,  0.124],
                              [0.,         0.,         0.,         1.]]

    init_pose = [0.1232563209, -0.0022815667, -0.0932499246, -
                 2.1356946695, -0.0281708669, 2.0154906706, 0.8219482610]

    return sorting_pose_right_color, sorting_pose_left_color, sorting_pose_right_size, sorting_pose_left_size, init_pose


def compute_transformation_distance(T1, T2):
    """
    Computes distance between the translation vectors of two transformation matrices
    """
    return np.linalg.norm(T1[:3, 3] - T2[:3, 3])


def adjust_brick_orientation(T_base2object):
    """Adjusts the orientation of the brick to ensure it is upright."""
    rotation_matrix_x = T_base2object[:3, :3]
    z_axis = rotation_matrix_x[:, 2]
    if z_axis[2] > 0:
        T_base2object = T_base2object @ rotation_matrix_x(180)
    return T_base2object


def create_brick_mask(maskModel, registered_bricks, index, w2, h2):
    """
    Generate a binary mask for a specific brick from registered detection data.

    Parameters:
    maskModel (object): The model used for detecting and labeling bricks.
    registered_bricks (object): The object containing detected brick information, including classes and masks.
    index (int): The index of the brick to generate the mask for.
    w2 (int): The width to which the mask should be resized.
    h2 (int): The height to which the mask should be resized.

    Returns:
    numpy.ndarray: A binary mask for the specified brick with dimensions (w2, h2).
    """
    brick_class = maskModel.names[int(
        registered_bricks.boxes[index].cls)]
    brick_class_id = registered_bricks.boxes[index].cls
    size = brick_class[:3]
    color = brick_class[4:]

    mask = registered_bricks.masks[index].cpu(
    ).data.numpy().transpose(1, 2, 0)
    mask = cv2.merge((mask, mask, mask))
    mask = cv2.resize(mask, (w2, h2))
    mask = cv2.inRange(mask, np.array([0, 0, 0]), np.array([0, 0, 1]))
    mask = cv2.bitwise_not(mask)
    return mask, size, color, brick_class_id
