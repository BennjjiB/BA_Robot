# Thesis

Lageerkennung und Greifplanung von Duplo-Steinen aus einem Haufen mit einem Roboterarm

## How to run sorting pipeline 
If not already done:
- Follow first step of "Data prepare" from https://github.com/NVlabs/FoundationPose
- train a yolo-seg model using Yolo_Seg/train.py (models can be downloaded from https://docs.ultralytics.com/tasks/segment/)
  - Dataset for the segmentation model can be labeled and properly exported with Roboflow
- change the path to the trained model in PoseEstimationApp.py

If everything done:

cd home/mrenz/anaconda3/lib/python3.11/site-packages/deoxys/auto_scripts
./auto_arm.sh ../config/charmander.yml ../config/control_config.yml
./auto_gripper.sh ../config/charmander.yml ../config/control_config.yml

-> python PoseEstimationApp.py

## How to run image generation

- Execute in BlenderProc-main: "blenderproc download cc_textures" and copy cc_textures folder into Brick_Pile_Generation

- Copy Brick_Pile_Generation in BlenderProc-main and execute python Brick_Pile_Generation/loop.py
