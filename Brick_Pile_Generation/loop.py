import subprocess
import random

num_scenes = 1
imgs_per_scene = 3
cnt = 1

for _ in range(num_scenes):
    random_num_bricks = random.randint(1, 10)
    print(f"Generating {imgs_per_scene} images for scene {cnt} with {random_num_bricks} bricks.")
    command = [
        'blenderproc', 'run',
        'Brick_Pile_Generation/main.py',
        'Brick_Pile_Generation/datasets',
        'bricks', 
        'Brick_Pile_Generation/cc_textures',
        'Brick_Pile_Generation/output',
        str(random_num_bricks),
        str(imgs_per_scene)
    ]
    subprocess.run(command, capture_output=True, text=True)
    cnt += 1 