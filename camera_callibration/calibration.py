import os
import numpy as np
import cv2
from cv2 import aruco
import pyrealsense2 as rs
from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.log_utils import get_deoxys_example_logger

ARUCO_DICT = 5
SQUARES_VERTICALLY = 14
SQUARES_HORIZONTALLY = 9
SQUARE_LENGTH = 0.0188
MARKER_LENGTH = 0.0146
PATH_TO_IMAGES = '/home/panda3/Desktop/Robot_BA/camera_callibration/images'

calibration_positions = [
        [-0.199427,-0.303889,0.0692703,-2.43281,0.0363531,1.98839,0.840368],
        [-0.338833,-0.111088,-0.288847,-2.20117,0.307392,1.73006,0.645364],
        [-0.22489,0.532639,-0.192103,-1.35517,0.13621,1.31752,0.0115306],
        [0.277271,0.54268,-0.209407,-1.42414,-0.0531952,1.39126,1.88619],
        [0.317017,-0.139508,0.109578,-2.12047,-0.397975,1.83889,2.40216],
        [0.27664,-0.600108,0.0578287,-2.37475,-0.554525,1.72298,-0.0554051],
        [0.0382363,-0.37407,-0.0369735,-1.61362,-0.139072,1.22564,0.876148],
        [1.04209,-0.82438,-0.866144,-1.86998,-0.341921,1.16218,0.7334],
        [1.34447,-1.05191,-1.01728,-1.80784,-0.622365,1.19451,1.57682],
        [1.42212,-0.691195,-1.4098,-2.11549,-0.692469,1.53,1.58114],
        [0.997678,0.422432,-1.19506,-1.82503,0.218938,1.51332,1.65147],
        [0.513488,-0.0925881,-0.612378,-1.90421,-0.101932,1.45205,1.12216],
        [-0.395858,-0.523988,0.200459,-2.04288,0.0372969,1.48128,0.889949]
    ]

logger = get_deoxys_example_logger()

def capture_frames_and_eef():
    robot_interface = FrankaInterface(
        config_root + "/charmander.yml", use_visualizer=False
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    pipeline.start(config)

    T_gripper2base = []

    for idx, position in enumerate(calibration_positions):

        reset_joints_to(robot_interface, position)

        # doch eher Transformationsmatrix R^T_H, also base2gripper?
        last_eef_pose = robot_interface.last_eef_pose

        T_gripper2base.append(last_eef_pose)

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        if idx < 10:
            idx = f"0{idx}"
            cv2.imwrite(f'{PATH_TO_IMAGES}/pos_{idx}.png', color_image)
        else:
            cv2.imwrite(f'{PATH_TO_IMAGES}/pos_{idx}.png', color_image)

    np.save('T_gripper2base.npy', T_gripper2base)


def calibrate():
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    board = aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    all_charuco_corners = []
    all_charuco_ids = []

    image_files = [os.path.join(PATH_TO_IMAGES, f) for f in os.listdir(PATH_TO_IMAGES) if f.endswith(".png")]
    image_files.sort()

    for image_file in image_files:
        
        img = cv2.imread(image_file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        image_size = img_gray.shape[::-1]
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(img_gray, dictionary)

        if len(marker_ids) > 0:
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, img_gray, board)
            if charuco_retval:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
            
    _, camera_matrix, dist_coeffs, _, _ = aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, image_size, None, None)

    all_charuco_corners = []
    all_charuco_ids = []

    for image_file in image_files:
        img = cv2.imread(image_file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h,  w = img_gray.shape[:2]
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
        img_gray = cv2.undistort(img_gray, camera_matrix, dist_coeffs, None, newcameramtx)

        marker_corners, marker_ids, _ = detector.detectMarkers(img_gray)
        if len(marker_ids) > 0:
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, img_gray, board)
            if charuco_retval:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)

    return aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, image_size, None, None)

def main():
    np.set_printoptions(suppress=True, precision=8)

    capture_frames_and_eef()

    T_gripper2base = np.load('T_gripper2base.npy')

    _, camera_matrix, dist_coeffs, R_target2cam, t_target2cam = calibrate()

    T_cam2target = [np.concatenate((cv2.Rodrigues(R)[0], T), axis=1) for R, T in zip(R_target2cam, t_target2cam)]
    for i in range(len(T_cam2target)):
        T_cam2target[i] = np.concatenate((T_cam2target[i], np.array([[0, 0, 0, 1]])), axis=0)

    R_gripper2base = [T[:3,:3] for T in T_gripper2base]
    t_gripper2base = [T[:3,3] for T in T_gripper2base]

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)

    T_cam2gripper = np.concatenate((np.concatenate((R_cam2gripper, t_cam2gripper), axis=1), [[0, 0, 0, 1]]), axis=0)

    print("T_cam2gripper: \n", T_cam2gripper)
    print(camera_matrix)

    np.save('camera_matrix.npy', camera_matrix)
    np.save('dist_coeffs.npy', dist_coeffs)
    np.save('T_cam2gripper.npy', T_cam2gripper)

if __name__ == "__main__":
    main()
