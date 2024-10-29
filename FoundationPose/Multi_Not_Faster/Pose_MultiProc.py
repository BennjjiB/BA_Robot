import redis
import numpy as np
import subprocess
import signal
import os
import pickle
import cv2
from ultralytics import YOLO
from RealSenseReader import *
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
import roboticstoolbox as rtb
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.experimental.motion_utils import reset_joints_to
from estimater import *
from datareader import *

HOST = "127.0.0.1"
PORT = 6379

class MultiObjectPoseController:
    def __init__(self, est_refine_iter, track_refine_iter, debug):
        self.est_refine_iter = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self.debug = debug
        self.code_dir = os.path.dirname(os.path.realpath(__file__))
        self.objects = {
            '4x2_brick': f'{self.code_dir}/out/mesh/4x2_brick.obj',
            '2x2_brick': f'{self.code_dir}/out/mesh/2x2_brick.obj',
        }
        self.redis_client = redis.Redis(host=HOST, port=PORT, db=0)
        self.processes = []
        self.should_terminate = False
        self.maskModel = YOLO('/home/mrenz/runs/segment/train15/weights/best.pt')
        self.should_terminate = False
        self.get_estimate = False

    def start_estimator(self):

        for obj_name in self.objects.keys():
            self.redis_client.delete(f"pose_{obj_name}")
            print(f"Cleared Redis key: {obj_name}")
            
        for obj_name, mesh_file in self.objects.items():
            process = subprocess.Popen([
                "python", "pose_estimation_worker.py",
                "--obj_name", obj_name,
                "--mesh_file", mesh_file,
                "--est_refine_iter", str(self.est_refine_iter),
                "--track_refine_iter", str(self.track_refine_iter),
                "--debug", str(self.debug),
                "--debug_dir", f'{self.code_dir}/debug'
            ])
            self.processes.append(process)

        print("All PoseEstimator processes started.")

    def stop_tracking(self):
        print("Terminating all PoseEstimator processes...")
        for process in self.processes:
            process.terminate()
        for process in self.processes:
            process.wait()
        print("All PoseEstimator processes terminated.")

    def signal_handler(self, signum, frame):
        self.should_terminate = True
        self.stop_tracking()

    def draw_button(self, img, button_text, button_size):
        text_color = (255, 255, 255)
        button_color = (0, 0, 0)
        button_x, button_y = (20, 20)
        button_width, button_height = button_size
        cv2.rectangle(img, (button_x, button_y), (button_x + button_width, button_y + button_height), button_color, -1)
        cv2.putText(img, button_text, (button_x + 10, button_y + button_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            button_x, button_y = (20, 20)
            button_width, button_height = (150, 50)
            if button_x < x < button_x + button_width and button_y < y < button_y + button_height:
                self.get_estimate = True

    def run(self):
        reader = RealSenseReader()

        while True:
            color_image, depth = reader.capture_image()
            color_image_copy = color_image.copy()
            registered_bricks = self.maskModel(color_image)[0]     
            annotated_frame = registered_bricks.plot()
            cv2.namedWindow("Estimating poses ...")
            cv2.setMouseCallback("Estimating poses ...", self.on_mouse)
            images = np.vstack((annotated_frame, color_image_copy))
            if registered_bricks[0].masks is not None:
                self.draw_button(images, "Estimate pose", (150,50))

            if self.get_estimate:
                long_bricks = []
                short_bricks = []

                for i in range(len(registered_bricks)):
                    brick_class = self.maskModel.names[int(registered_bricks.boxes[i].cls)]
                    brick_size = brick_class[:3]
                    brick_color = brick_class[4:]
                    brick_mask = registered_bricks.masks[i]
                    shape = registered_bricks.orig_img.shape

                    if brick_size == "4x2":
                        long_bricks.append([brick_mask, brick_color])
                    elif brick_size == "2x2":
                        short_bricks.append([brick_mask, brick_color])

                bricks_long = {
                    'K': reader.K,
                    'shape': shape,
                    'color': color_image,
                    'depth': depth,
                    'bricks': long_bricks
                }
                bricks_short = {
                    'K': reader.K,
                    'shape': shape,
                    'color': color_image,
                    'depth': depth,
                    'bricks': short_bricks
                }

                serialized_data_long = pickle.dumps(bricks_long)
                self.redis_client.set("pose_4x2_brick", serialized_data_long)
                serialized_data_short = pickle.dumps(bricks_short)
                self.redis_client.set("pose_2x2_brick", serialized_data_short)
                self.get_estimate = False

                while True: 
                    serialized_data_long = self.redis_client.get("pose_4x2_brick")
                    serialized_data_short = self.redis_client.get("pose_2x2_brick")
                    
                    if serialized_data_long and serialized_data_short:
                        bricks_long = pickle.loads(serialized_data_long)
                        bricks_short = pickle.loads(serialized_data_short)

                        if isinstance(bricks_long, list) and isinstance(bricks_short, list):
                            for brick in bricks_long:
                                vis = draw_posed_3d_box(reader.K, img=color_image, ob_in_cam=brick['center_pose'], bbox=brick['bbox'])
                            for brick in bricks_short:
                                vis = draw_posed_3d_box(reader.K, img=color_image, ob_in_cam=brick['center_pose'], bbox=brick['bbox'])
                            #vis = draw_xyz_axis(color_image, ob_in_cam=bricks_long['center_pose'], scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)

                            images = np.vstack((annotated_frame, vis))
                            cv2.imshow('Estimating poses ...', images)

                            key=cv2.waitKey(0)
                            if key & 0xFF == ord('q') or key == 27:
                                break    

                            break
                

            cv2.imshow('Estimating poses ...', images)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break


        self.stop_tracking()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    multi_tracker = MultiObjectPoseController(est_refine_iter=5, track_refine_iter=2, debug=1)
    
    signal.signal(signal.SIGINT, multi_tracker.signal_handler)
    multi_tracker.start_estimator()

    multi_tracker.run()