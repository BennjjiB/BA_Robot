import argparse
import redis
import numpy as np
import trimesh
import pickle
from estimater import *
from datareader import *
from RealSenseReader import *
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
import roboticstoolbox as rtb
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.experimental.motion_utils import reset_joints_to

HOST = "127.0.0.1"
PORT = 6379

class PoseEstimator:
    def __init__(self, obj_name, mesh_file, est_refine_iter, track_refine_iter, debug, debug_dir):
        self.object_name = obj_name
        self.est_refine_iter = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self.debug = debug
        self.debug_dir = debug_dir
        self.mesh = trimesh.load(mesh_file)

        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0).reshape(2, 3)

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()  
        self.est = FoundationPose(model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals,
                                  mesh=self.mesh, scorer=self.scorer, refiner=self.refiner,
                                  debug_dir=self.debug_dir, debug=self.debug, glctx=self.glctx)

        self.redis_controller = redis.Redis(host=HOST, port=PORT, db=0)

    def get_pose(self, K, color, depth, mask):
        pose = self.est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=self.est_refine_iter)
        center_pose = pose@np.linalg.inv(self.to_origin)

        return center_pose

    def get_results(self):
        redis_key = f"pose_{self.object_name}"
        serialized_data = self.redis_controller.get(redis_key)

        if serialized_data:
            bricks = pickle.loads(serialized_data)

            if isinstance(bricks, dict):
                k = bricks['K']
                h2, w2, _  = bricks['shape']
                color = bricks['color']
                depth = bricks['depth']
                bricks = bricks['bricks']
                h2 = int(h2 * 0.3)
                w2 = int(w2 * 0.3)
                k[:2] *= 0.3

                result = []
                for brick in bricks:
                    print(f"estimating pose ({self.object_name})")
                    mask_raw = brick[0].cpu().data.numpy().transpose(1, 2, 0)
                    mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))
                    mask = cv2.resize(mask_3channel, (w2, h2))
                    mask = cv2.inRange(mask, np.array([0,0,0]), np.array([0,0,1]))
                    mask = cv2.bitwise_not(mask)
                    color = cv2.resize(color, (w2, h2))
                    depth = cv2.resize(depth, (w2, h2))

                    center_pose = self.get_pose(k, color, depth, mask)

                    result.append({
                        'center_pose': center_pose,
                        'bbox': self.bbox
                    })

                serialized_data = pickle.dumps(result)

                self.redis_controller.set(f"pose_{self.object_name}", serialized_data)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_name", type=str, default='')
    parser.add_argument("--mesh_file", type=str, default='')
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug_dir", type=str, default="")
    args = parser.parse_args()

    estimator = PoseEstimator(args.obj_name, args.mesh_file, args.est_refine_iter, args.track_refine_iter, args.debug, args.debug_dir)

    while True:
        estimator.get_results()
        time.sleep(1)
            