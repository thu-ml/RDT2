"""
python tests/test_ur_tracker_combine.py

"""

import sys
sys.path.append('.')

import os

from deploy.umi.real_world.rtde_interpolation_controller import RTDEInterpolationController

from deploy.calibration.tracker import (
    TrackerController, TrackerControllerConfig,
    get_all_tracker_serials
)
from scipy.spatial.transform import Rotation as R
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import time


def main():
    fps = 30
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ur5e_ip", type=str, default="192.168.56.10")
    args = parser.parse_args()
    
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    ur5e_config = RTDEInterpolationController(
        shm_manager=shm_manager,
        robot_ip=args.ur5e_ip,
        frequency=125,
        verbose=False
    )
    
    ur5e = ur5e_config
    ur5e.start()
    
    ur5e.start_wait()
    serial_number = get_all_tracker_serials()
    tracker_serial = serial_number[0]
    print(f"tracker_serials: {tracker_serial}")
    tracker_config = TrackerControllerConfig(
        
        name="tracker1",
        tracker_names=["tracker1"],
        tracker_serials=[tracker_serial],
        fps=30,
        put_desired_frequency=30,
    )
    
    tracker = TrackerController(tracker_config)

    tracker.start()
    tracker._ready_event.wait()
    
    while not tracker.check_ready():
        pass
    
    
    tracker_posrotvec_lis = [] # list of (6, )
    ur_posrotvec_lis = [] # list of (6, )
    
    def append_posrotvec(ur5e: RTDEInterpolationController, tracker: TrackerController):
        tracker_feedback_state_dict = tracker.get_feedback()
        ur5e_feedback_state_dict = ur5e.get_state()
        tracker_pose = tracker_feedback_state_dict['tracker1_pose'] # (7, ), xyz + quat
        tracker_posrotvec = np.concatenate([
            tracker_pose[:3], # (3, )
            R.from_quat(tracker_pose[3:]).as_rotvec(),  # (3, )
        ]) # (6, )
        tracker_posrotvec_lis.append(tracker_posrotvec) # list of (6, )
        # ActualTCPPose
        ur5e_posrotvec = ur5e_feedback_state_dict['ActualTCPPose'] # (6, )
        ur_posrotvec_lis.append(ur5e_posrotvec) # list of (6, )
    
    # Get initial position
    init_ur5e_pose = ur5e.get_state()['ActualTCPPose']
    
    # Convert rotation vector to rotation matrix
    def rotation_vector_to_matrix(rvec):
        """Convert rotation vector to rotation matrix"""
        angle = np.linalg.norm(rvec)
        if angle == 0:
            return np.eye(3)
        axis = rvec / angle
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        return R
    
    # Get rotation matrix of initial pose
    init_rotation_vec = init_ur5e_pose[3:6]
    rotation_matrix = rotation_vector_to_matrix(init_rotation_vec)
    
    # 6D sinusoidal wave parameters
    pos_amplitude = 0.02  # Position amplitude (meters)
    rot_amplitude = 0.02   # Rotation amplitude (radians)
    frequencies = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # Frequency for each axis (Hz)
    phase_offsets = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])  # Phase offsets
    duration = 10.0  # Total motion duration (seconds)
    
    print(f"Starting 6D synchronized sinusoidal motion, duration: {duration} seconds...")
    start_time = time.monotonic()
    
    while time.monotonic() - start_time < duration:
        t = time.monotonic() - start_time
        
        # Calculate 6D sinusoidal wave offsets
        # Position offsets (x, y, z)
        pos_offsets = np.array([
            pos_amplitude * np.sin(2 * np.pi * frequencies[0] * t + phase_offsets[0]),  # x
            pos_amplitude * np.sin(2 * np.pi * frequencies[1] * t + phase_offsets[1]),  # y
            pos_amplitude * np.sin(2 * np.pi * frequencies[2] * t + phase_offsets[2])   # z
        ])
        
        # Rotation offsets (rx, ry, rz) - in local coordinate system
        rot_offsets = np.array([
            rot_amplitude * np.sin(2 * np.pi * frequencies[3] * t + phase_offsets[3]),  # rx
            rot_amplitude * np.sin(2 * np.pi * frequencies[4] * t + phase_offsets[4]),  # ry
            rot_amplitude * np.sin(2 * np.pi * frequencies[5] * t + phase_offsets[5])   # rz
        ])
        
        # Convert local coordinate system position offsets to global coordinate system
        global_pos_offsets = rotation_matrix @ pos_offsets
        
        # Convert local coordinate system rotation offsets to global coordinate system
        global_rot_offsets = rotation_matrix @ rot_offsets
        
        # Build target pose
        target_pose = init_ur5e_pose.copy()
        
        # Apply position offsets
        target_pose[0:3] = init_ur5e_pose[0:3] + global_pos_offsets
        
        # Apply rotation offsets: first apply initial rotation, then apply additional rotation
        init_rotation = R.from_rotvec(init_ur5e_pose[3:6])
        additional_rotation = R.from_rotvec(global_rot_offsets)
        combined_rotation = additional_rotation * init_rotation
        target_pose[3:6] = combined_rotation.as_rotvec()
        
        # Send command
        ur5e.schedule_waypoint(
            pose=target_pose,
            target_time=time.time() + 1.0 / fps
        )
        
        append_posrotvec(ur5e, tracker)
        
        time.sleep(1.0 / fps)
    
    # Return to initial position
    print("Motion ended, returning to initial position...")
    ur5e.schedule_waypoint(
        pose=init_ur5e_pose,
        target_time=time.time() + 2.0
    )
    time.sleep(2.0)
    
    print("6D synchronized sinusoidal motion completed!")
    
    tracker_posrotvec_np = np.stack(tracker_posrotvec_lis, axis=0) # (T, 6)
    ur_posrotvec_np = np.stack(ur_posrotvec_lis, axis=0) # (T, 6)
    
    os.makedirs("./tmp", exist_ok=True)
    np.savetxt(f"./tmp/tracker_posrotvec_np.txt", tracker_posrotvec_np, fmt="%.6f")
    np.savetxt(f"./tmp/robot_posrotvec_np.txt", ur_posrotvec_np, fmt="%.6f")

    # tracker.stop()
    # ur5e.stop()
    exit()
if __name__ == '__main__':
    main()
