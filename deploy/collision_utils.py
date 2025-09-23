import warnings

import numpy as np
import scipy.spatial.transform as st


def solve_table_collision(ee_pose, gripper_width, finger_thickness, height_threshold):
    """
    Adjust the end-effector pose to avoid collision with the table.
    
    Args:
        ee_pose (np.ndarray): The end-effector pose in the format [pos + rotation vector].
        gripper_width (float): The width of the gripper in meters.
        finger_thickness (float): The thickness of the fingers in meters.
        height_threshold (float): The minimum height above the table to avoid collision.
    
    Returns:
        None: The function modifies the ee_pose in place to ensure it is above the table.
    """
    keypoints = [
        (dx * gripper_width / 2, dy * finger_thickness / 2, 0)
        for dx in [-1, 1] for dy in [-1, 1]
    ]
    keypoints = np.asarray(keypoints)
    
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[: 3]
    # print("[DBG] ", transformed_keypoints)
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    
    if delta > 0:
        warnings.warn(
            f"End-effector pose {ee_pose} is too low,"
            f" adjusting by {delta * 100:.3f} cm to avoid collision with the table."
        )
    
    ee_pose[..., 2] += delta


def solve_sphere_collision(ee_poses, robots_confg):
    """
    Adjust the end-effector pose to avoid collision with the sphere.
    
    Args:
        ee_poses (np.ndarray): The end-effector pose in the format [pos + rotation vector].
        robots_confg (dict): The configuration of the robots.
    
    Returns:
        None: The function modifies the ee_poses in place to ensure it is above the sphere.
    """
    pass
    