# Panda Robot Arm Controller

Object Detection and Grasp Planning of Duplo Bricks from a Pile with a Robotic Arm.

## Dependecies 
This project uses Foundation Pose for pose estimation and tracking of the bricks: 
https://github.com/NVlabs/FoundationPose

To interface with the robot a wrapper of franka lib called deoxys control is used:
https://github.com/UT-Austin-RPL/deoxys_control  

## Setup
Install the requiremnts.txt file:
pip install -r requirements.txt

### Previous challenges:
- As pytorch3d does not support pip, we had to build and install it from a local clone
  https://github.com/facebookresearch/pytorch3d
  Install Infos: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md 
- Adjust all path pointing to the yolo model weighs: "/home/panda3/Desktop/Robot_BA/best.pt"

### Changed Files
Some files from external libraries have been modified: 
- The deoxys [charmander.yml](../../ws/3rd_party_lib/deoxys_control/deoxys/config/charmander.yml) and [control_config.yml](../../ws/3rd_party_lib/deoxys_control/deoxys/config/control_config.yml)
- The deoxys [motion_utils.py](../../ws/3rd_party_lib/deoxys_control/deoxys/deoxys/experimental/motion_utils.py) has been modified to include the gripper width 
- The Yolo [results.py](../../ws/venv38/lib/python3.8/site-packages/ultralytics/engine/results.py) and [plotting.py](../../ws/venv38/lib/python3.8/site-packages/ultralytics/utils/plotting.py) has been modified to show the match the bounding box color to the brick color 


## How to run the sorting pipeline 
1. Go to the franka web interface: https://172.16.0.2
2. Unlock all joints
3. Activate FCI
4. Run the scripts in deoxys/auto_scripts.   
  ./auto_arm.sh ../config/charmander.yml ../config/control_config.yml  
  ./auto_gripper.sh ../config/charmander.yml ../config/control_config.yml  
  You can use the helper scripts: [run_arm](../run_arm.sh) and [run_gripper](../run_gripper.sh) make sure to change the target directory to your version of deoxys.
5. Run the gradio app_ui.py

## How to calibrate the camera
Camera calibration should be done about once a month or if the grasping becomes inaccurate. 
1. In calibration.py Change the PATH_TO_IMAGES to the correct path of the image folder
2. Execute calibration.py 
3. Move T_cam2gripper.npy into the FoundationPose folder

## How to run image generation
- Execute in BlenderProc-main: "blenderproc download cc_textures" and copy cc_textures folder into Brick_Pile_Generation
- Copy Brick_Pile_Generation in BlenderProc-main and execute python Brick_Pile_Generation/loop.py
<br>
<br>
# Chatbot and robot interface 
The chatbot interface is done using [Gradio](https://www.gradio.app/). 

Following components are used:
- Client: 
  - Sends audio chunks to ther server for translation. 
  - Sends prompts and function responses to the server.
- ToolService:
  - Parses and executes tool calls made from the Llama model 
- RobotInterface: 
  - A small wrapper above the PoseEstimationApp, providing the tool call api. 



