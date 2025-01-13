import pybullet as p
import math
import numpy as np
import pybullet_data

def get_gripping_points(T_base2bricks): 

    def rotation_matrix_to_quaternion(R):
        yaw = math.atan2(R[1, 0], R[0, 0])
        pitch = math.asin(-R[2, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
        return p.getQuaternionFromEuler([roll, pitch, yaw])
    
    def quaternion_to_matrix(position, quaternion):
        rotation_matrix_flat = p.getMatrixFromQuaternion(quaternion)

        rotation_matrix = np.array(rotation_matrix_flat).reshape(3, 3)

        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = position

        return T
    
    def maxtrix_to_translation_quaternion(transformation_matrix):
        translation = transformation_matrix[:3, 3]
        rotation_matrix = transformation_matrix[:3, :3]
        quaternion = rotation_matrix_to_quaternion(rotation_matrix)
        return translation, quaternion

    def create_rotation_transformation(angle_degrees, axis):
        angle_radians = np.radians(angle_degrees)
        if axis == 'x':
            return np.array([
                [1, 0, 0, 0],
                [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
                [0, np.sin(angle_radians), np.cos(angle_radians), 0],
                [0, 0, 0, 1]
            ])
        elif axis == 'y':
            return np.array([
                [np.cos(angle_radians), 0, np.sin(angle_radians), 0],
                [0, 1, 0, 0],
                [-np.sin(angle_radians), 0, np.cos(angle_radians), 0],
                [0, 0, 0, 1]
            ])
        elif axis == 'z':
            return np.array([
                [np.cos(angle_radians), -np.sin(angle_radians), 0, 0],
                [np.sin(angle_radians), np.cos(angle_radians), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        
    def grip_should_widen(finger_1, finger_2, T_base2brick):
        vector = np.array(finger_1[0]) - np.array(finger_2[0])

        rotation_matrix = T_base2brick[:3, :3]
        x_axis = rotation_matrix[:, 0]
        y_axis = rotation_matrix[:, 1]
        z_axis = rotation_matrix[:, 2]

        proj_length_x = np.abs(np.dot(vector, x_axis) / np.linalg.norm(x_axis))
        proj_length_y = np.abs(np.dot(vector, y_axis) / np.linalg.norm(y_axis))
        proj_length_z = np.abs(np.dot(vector, z_axis) / np.linalg.norm(z_axis))

        return proj_length_x > proj_length_y and proj_length_x > proj_length_z
    
    def grips_along_z_axis(finger_1, finger_2, T_base2brick):
        vector = np.array(finger_1[0]) - np.array(finger_2[0])

        rotation_matrix = T_base2brick[:3, :3]
        x_axis = rotation_matrix[:, 0]
        y_axis = rotation_matrix[:, 1]
        z_axis = rotation_matrix[:, 2]

        proj_length_x = np.abs(np.dot(vector, x_axis) / np.linalg.norm(x_axis))
        proj_length_y = np.abs(np.dot(vector, y_axis) / np.linalg.norm(y_axis))
        proj_length_z = np.abs(np.dot(vector, z_axis) / np.linalg.norm(z_axis))

        return proj_length_z > proj_length_y and proj_length_z > proj_length_x
    
    def z_axis_gripper_matches_x_axis_brick(T_gripper, T_brick):
        body_trans = T_gripper[0]
        body_quat = T_gripper[1]

        T_gripper = quaternion_to_matrix(body_trans, body_quat)

        z_axis_gripper = T_gripper[:3, 2]
        x_axis_brick = T_brick[:3, 0]

        z_axis_gripper = z_axis_gripper / np.linalg.norm(z_axis_gripper)
        x_axis_brick = x_axis_brick / np.linalg.norm(x_axis_brick)

        dot_product = np.dot(z_axis_gripper, x_axis_brick)

        return np.abs(dot_product) > 0.9

    
    def choose_best_grip(grips):
        global_z = np.array([0, 0, 1])     
        max_score = -np.inf
        best_grip = None

        # First filter out grips with a dot product less than 0.1 (cases where gripper is parallel to table)
        filtered_grips = []
        for grip in grips:
            z_axis = grip[0][:3, 2]
            dot_product = np.dot(z_axis, global_z)
            if dot_product >= 0.4:
                filtered_grips.append(grip)

        for grip in filtered_grips:
            z_axis = grip[0][:3, 2]
            z = grip[0][2][3]
            score = np.dot(z_axis, global_z) / 2
            is_center_grip = grip[2]
            is_wide_grip = grip[1]
            grips_z_axis = grip[7]
            bonus = 0.1
            score = score + z*15
            if is_center_grip:
                score += bonus
            if not is_wide_grip:
                score += bonus
            if grips_z_axis:
                score -= bonus
            if score > max_score:
                max_score = score
                best_grip = grip

        return best_grip
    
    connection_mode = p.DIRECT
    cid = p.connect(connection_mode)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, -0.01])

    ### BRICKS ###

    colors = {
        "red": [1.0, 0.0, 0.0, 1.0],
        "green": [0.0, 1.0, 0.0, 1.0],
        "blue": [0.0, 0.0, 1.0, 1.0],
        "yellow": [1.0, 1.0, 0.0, 1.0],
        "orange": [1.0, 0.5, 0.0, 1.0]
    }

    brick_ids = []

    for brick in T_base2bricks:
        brick_color = brick[2] 
        brick_size = brick[1]
        if brick_size == "4x2":
            brick_dimensions = [0.063, 0.031, 0.024]
        elif brick_size == "2x2":
            brick_dimensions = [0.031, 0.031, 0.024]
            
        brick_pose = brick[0] @ np.array([
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [-1, 0, 0, 0],
                    [0, 0, 0, 1]
                ])
        translation, quaternion = maxtrix_to_translation_quaternion(brick_pose)

        # Create brick with both visual and collision properties
        brick_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[d / 2 for d in brick_dimensions], rgbaColor=colors[brick_color])
        brick_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[d / 2 for d in brick_dimensions])
        brick_id = p.createMultiBody(baseMass=0, 
                                    baseVisualShapeIndex=brick_visual_shape_id, 
                                    baseCollisionShapeIndex=brick_collision_shape_id, 
                                    basePosition=translation, 
                                    baseOrientation=quaternion)
        brick_ids.append(brick_id)

    ### GRIPPER ###

    initial_position = [0.0, 0.0, 0.0]
    initial_orientation = p.getQuaternionFromEuler([0, 0, 0]) 

    #Base
    base_dimensions = [0.0, 0.0, 0.0]
    base_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[d / 2 for d in base_dimensions], rgbaColor=[1, 1, 1, 1])
    base_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[d / 2 for d in base_dimensions])
    
    # Body
    body_dimensions = [0.04, 0.25, 0.07]
    body_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[d / 2 for d in body_dimensions], rgbaColor=[1, 1, 1, 1])
    body_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[d / 2 for d in body_dimensions])

    # Finger
    finger_dimensions = [0.018, 0.018, 0.03]
    finger_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[d / 2 for d in finger_dimensions], rgbaColor=[0.3, 0.3, 0.3, 0.8])
    finger_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[d / 2 for d in finger_dimensions])

    # Fingertip
    fingertip_dimensions = [0.018, 0.018, 0.017]
    fingertip_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[d / 2 for d in fingertip_dimensions], rgbaColor=[0.0, 0.0, 0.0, 0.8])
    fingertip_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[d / 2 for d in fingertip_dimensions])

    # Camera
    camera_dimensions = [0.06, 0.11, 0.03]
    camera_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[d / 2 for d in camera_dimensions], rgbaColor=[0.5, 0.5, 0.5, 1])
    camera_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[d / 2 for d in camera_dimensions])

    #relative positions
    inner_left_fingertip_position = [initial_position[0], initial_position[1] - 0.032, initial_position[2]]
    inner_right_fingertip_position = [initial_position[0], initial_position[1] + 0.032, initial_position[2]]
    inner_left_finger_position = [initial_position[0], initial_position[1] - 0.032, initial_position[2] + fingertip_dimensions[2] / 2 + finger_dimensions[2] / 2]
    inner_right_finger_position = [initial_position[0], initial_position[1] + 0.032, initial_position[2] + fingertip_dimensions[2] / 2 + finger_dimensions[2]/ 2]
    outer_left_fingertip_position = [initial_position[0], initial_position[1] - 0.052, initial_position[2]]
    outer_right_fingertip_position = [initial_position[0], initial_position[1] + 0.052, initial_position[2]]
    outer_left_finger_position = [initial_position[0], initial_position[1] - 0.052, initial_position[2] + fingertip_dimensions[2] / 2 + finger_dimensions[2] / 2]
    outer_right_finger_position = [initial_position[0], initial_position[1] + 0.052, initial_position[2] + fingertip_dimensions[2] / 2 + finger_dimensions[2]/ 2]
    body_position = [initial_position[0], initial_position[1], initial_position[2] + fingertip_dimensions[2] / 2 + finger_dimensions[2] + body_dimensions[2] / 2]
    camera_position = [initial_position[0] + body_dimensions[0] / 2 + camera_dimensions[0] / 2, initial_position[1] - 0.0125, fingertip_dimensions[2] / 2 + finger_dimensions[2] + 0.04 + camera_dimensions[2] / 2]


    ## maybe create new multibody including left and right finder parts and then add it
    # Create MultiBody
    gripper_id = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=-1,
                                baseVisualShapeIndex=base_visual_shape_id,
                                basePosition=initial_position,
                                baseOrientation=initial_orientation,
                                linkMasses=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                linkCollisionShapeIndices=[fingertip_collision_shape_id, 
                                                           fingertip_collision_shape_id, 
                                                           finger_collision_shape_id, 
                                                           finger_collision_shape_id, 
                                                           fingertip_collision_shape_id, 
                                                           fingertip_collision_shape_id, 
                                                           finger_collision_shape_id, 
                                                           finger_collision_shape_id, 
                                                           body_collision_shape_id, 
                                                           camera_visual_shape_id],
                                linkVisualShapeIndices=[fingertip_visual_shape_id, 
                                                        fingertip_visual_shape_id, 
                                                        finger_visual_shape_id, 
                                                        finger_visual_shape_id, 
                                                        fingertip_visual_shape_id, 
                                                        fingertip_visual_shape_id, 
                                                        finger_visual_shape_id, 
                                                        finger_visual_shape_id,
                                                        body_visual_shape_id, 
                                                        camera_collision_shape_id],
                                linkPositions=[inner_left_fingertip_position, 
                                               inner_right_fingertip_position, 
                                               inner_left_finger_position, 
                                               inner_right_finger_position,
                                               outer_left_fingertip_position, 
                                               outer_right_fingertip_position, 
                                               outer_left_finger_position, 
                                               outer_right_finger_position, 
                                               body_position, 
                                               camera_position],
                                linkOrientations=[initial_orientation, 
                                                  initial_orientation, 
                                                  initial_orientation, 
                                                  initial_orientation,
                                                  initial_orientation, 
                                                  initial_orientation, 
                                                  initial_orientation, 
                                                  initial_orientation,  
                                                  initial_orientation, 
                                                  initial_orientation],
                                linkInertialFramePositions=[[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0],  [0,0,0], [0,0,0]],
                                linkInertialFrameOrientations=[initial_orientation, 
                                                               initial_orientation, 
                                                               initial_orientation, 
                                                               initial_orientation, 
                                                               initial_orientation, 
                                                               initial_orientation, 
                                                               initial_orientation, 
                                                               initial_orientation, 
                                                               initial_orientation, 
                                                               initial_orientation],
                                linkParentIndices=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                linkJointTypes=[p.JOINT_FIXED, 
                                                p.JOINT_FIXED, 
                                                p.JOINT_FIXED, 
                                                p.JOINT_FIXED, 
                                                p.JOINT_FIXED, 
                                                p.JOINT_FIXED, 
                                                p.JOINT_FIXED, 
                                                p.JOINT_FIXED, 
                                                p.JOINT_FIXED, 
                                                p.JOINT_FIXED],
                                linkJointAxis=[[0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0],  [0,1,0], [0,1,0]])
    
    ### FINDING BEST POSE ####

    for brick_id in brick_ids:
        p.setCollisionFilterPair(brick_id, gripper_id, -1, -1, enableCollision=1)
    
    p.setCollisionFilterPair(plane_id, gripper_id, -1, -1, enableCollision=1)

    collision_free = []

    for id, brick in enumerate(T_base2bricks):

        T_base2brick = brick[0] @ np.array([
                            [0, 0, -1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]
                        ]) @ np.array([
                        [1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]
                    ])  
        T_base2gripper = T_base2brick
        
        if brick[1] == "2x2": 
            for angle_x in [0,90,90,90]:
                T_base2gripper = T_base2gripper @ create_rotation_transformation(angle_x, 'x')
                translation, quaternion = maxtrix_to_translation_quaternion(T_base2gripper)
                for angle_y in [0,90,90,90]:
                    T_base2gripper = T_base2gripper @ create_rotation_transformation(angle_y, 'y')
                    translation, quaternion = maxtrix_to_translation_quaternion(T_base2gripper)
                    for angle_z in [0,90,90,90]:
                        T_base2gripper = T_base2gripper @ create_rotation_transformation(angle_z, 'z')
                        translation, quaternion = maxtrix_to_translation_quaternion(T_base2gripper)
                        p.resetBasePositionAndOrientation(gripper_id, translation, quaternion)

                        fingertip_1 = p.getLinkState(gripper_id, 0)
                        fingertip_2 = p.getLinkState(gripper_id, 1)
                        grips_z_axis = grips_along_z_axis(fingertip_1, fingertip_2, T_base2brick)

                        for _ in range(10):
                            p.stepSimulation()
                        
                        has_collision = p.getContactPoints(plane_id, gripper_id)
                        if has_collision:
                            continue
                        
                        collision = False
                        for brick_id in brick_ids:
                            has_collision = p.getContactPoints(brick_id, gripper_id)
                            if has_collision:
                                collision = True
                                break
                        
                        if not collision:
                            collision_free.append([T_base2gripper, False, True, brick, id, False, T_base2brick, grips_z_axis, 0])    

        wide_grip = False
        center_grip = True                

        if brick[1] == "4x2": 
            for angle_x in [0,90,90,90]:
                T_base2gripper = T_base2gripper @ create_rotation_transformation(angle_x, 'x')
                translation, quaternion = maxtrix_to_translation_quaternion(T_base2gripper)
                for angle_y in [0,90,90,90]:
                    T_base2gripper = T_base2gripper @ create_rotation_transformation(angle_y, 'y')
                    translation, quaternion = maxtrix_to_translation_quaternion(T_base2gripper)
                    for angle_z in [0,90,90,90]:
                        T_base2gripper = T_base2gripper @ create_rotation_transformation(angle_z, 'z')
                        translation, quaternion = maxtrix_to_translation_quaternion(T_base2gripper)
                        p.resetBasePositionAndOrientation(gripper_id, translation, quaternion)

                        fingertip_1 = p.getLinkState(gripper_id, 0)
                        fingertip_2 = p.getLinkState(gripper_id, 1)
                        body = p.getLinkState(gripper_id, 8)

                        if grip_should_widen(fingertip_1, fingertip_2, T_base2brick):
                            collision_check_links = [4, 5, 6, 7, 8, 9] 
                            wide_grip = True
                        else:
                            collision_check_links = [0, 1, 2, 3, 8, 9]
                            wide_grip = False
                        
                        final_x_offset = 0
                        if not wide_grip: 
                            offsets = [0.02, -0.04, 0.02]
                            x_axis_base2brick = T_base2brick[:3, 0]

                            for i, x_offset in enumerate(offsets):
                                translation_vector = x_axis_base2brick * x_offset
                                translation_matrix = np.eye(4)
                                translation_matrix[:3, 3] = translation_vector
                                T_base2gripper = np.dot(translation_matrix, T_base2gripper)
                                translation, quaternion = maxtrix_to_translation_quaternion(T_base2gripper)
                                p.resetBasePositionAndOrientation(gripper_id, translation, quaternion)

                                brick_is_upright = z_axis_gripper_matches_x_axis_brick(body, T_base2brick)
                                grips_z_axis = grips_along_z_axis(fingertip_1, fingertip_2, T_base2brick)

                                for _ in range(10):
                                    p.stepSimulation()

                                contact_points = p.getContactPoints(bodyA=gripper_id, bodyB=plane_id)
                                finger_contacts = [cp for cp in contact_points if cp[3] in collision_check_links or cp[4] in collision_check_links]
                                if finger_contacts: 
                                    continue

                                collision = False
                                for brick_id in brick_ids:
                                    contact_points = p.getContactPoints(bodyA=gripper_id, bodyB=brick_id)
                                    finger_contacts = [cp for cp in contact_points if cp[3] in collision_check_links or cp[4] in collision_check_links]
                                    if finger_contacts:
                                        collision = True
                                        break
                                
                                if not collision: 
                                    center_grip = (i == 2)
                                    if not grips_z_axis and not brick_is_upright:
                                        translation_T1 = T_base2brick[:3, 3]
                                        translation_T2 = T_base2gripper[:3, 3]
                                        x_axis_T2 = T_base2gripper[:3, 0]
                                        translation_difference = translation_T2 - translation_T1
                                        final_x_offset = np.dot(translation_difference, x_axis_T2)
                                    collision_free.append([T_base2gripper, wide_grip, center_grip, brick, id, brick_is_upright, T_base2brick, grips_z_axis, final_x_offset])
                                    final_x_offset = 0
                        else:
                            for _ in range(10):
                                p.stepSimulation()
                            
                            contact_points = p.getContactPoints(bodyA=gripper_id, bodyB=plane_id)
                            finger_contacts = [cp for cp in contact_points if cp[3] in collision_check_links or cp[4] in collision_check_links]
                            if finger_contacts: 
                                continue

                            collision = False
                            for brick_id in brick_ids:
                                contact_points = p.getContactPoints(bodyA=gripper_id, bodyB=brick_id)
                                finger_contacts = [cp for cp in contact_points if cp[3] in collision_check_links or cp[4] in collision_check_links]
                                if finger_contacts:
                                    collision = True
                                    break
                            
                            if not collision:
                                collision_free.append([T_base2gripper, wide_grip, center_grip, brick, id, False, T_base2brick, grips_z_axis, final_x_offset])
                        
                        center_grip = True

    grip = choose_best_grip(collision_free)

    if connection_mode == p.GUI:    
        translation, quaternion = maxtrix_to_translation_quaternion(grip[0])
        p.resetBasePositionAndOrientation(gripper_id, translation, quaternion)
        for _ in range(10):
            p.stepSimulation()     
        input("Press the <Enter> key on the cmd line to exit.")

    p.disconnect(cid)

    # [T_base2gripper, wide_grip, center_grip, [T_base2brick, size, color, mask, brick_class_id], id, brick_is_upright, original_pose, grips_z_axis, final_x_offset]
    return grip, collision_free