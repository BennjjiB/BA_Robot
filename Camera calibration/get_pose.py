from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.experimental.motion_utils import reset_joints_to, follow_joint_traj
import numpy as np

def main():
    robot_interface = FrankaInterface(
        config_root + "/charmander.yml", use_visualizer=False
    )

    current_pos = None
    while current_pos is None:
        current_pos = robot_interface.last_q

    print(np.array2string(current_pos, formatter={'float_kind': lambda x: f"{x:.10f}"}))


if __name__ == "__main__":
    main()
