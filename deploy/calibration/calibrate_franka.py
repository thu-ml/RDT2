"""
python tests/test_frank_tracker_combine.py

Test script for Franka robot with tracker combination
"""

import sys
sys.path.append('.')

import os

from deploy.umi.real_world.franka_interpolation_controller import FrankaInterpolationController

from deploy.calibration.tracker import (
    TrackerController, TrackerControllerConfig,
    get_all_tracker_serials
)
from multiprocessing.managers import SharedMemoryManager
from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np
import time

def main():
    fps = 30
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--franka_ip", type=str, default="localhost")
    parser.add_argument("--franka_port", type=int, default=4243)
    args = parser.parse_args()
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    franka = FrankaInterpolationController(
        shm_manager=shm_manager,
        robot_ip=args.franka_ip,
        robot_port=args.franka_port,
        frequency=300,
        verbose=False,
        receive_latency=0.001
    )
    franka.start(wait=False)
    franka.start_wait()

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
        print(f"Tracker not ready")
        pass
    print(f"Tracker ready")
    tracker_posrotvec_lis = [] # list of (6, )
    ur_posrotvec_lis = [] # list of (6, )
    
    def append_posrotvec(franka, tracker: TrackerController):
        tracker_feedback_state_dict = tracker.get_feedback()
        franka_feedback_state_dict = franka.get_state()
        tracker_pose = tracker_feedback_state_dict['tracker1_pose'] # (7, ), xyz + quat
        tracker_posrotvec = np.concatenate([
            tracker_pose[:3], # (3, )
            R.from_quat(tracker_pose[3:]).as_rotvec(),  # (3, )
        ]) # (6, )
        tracker_posrotvec_lis.append(tracker_posrotvec) # list of (6, )
        # Get pose from Franka (pos_rotvec is position + rotation vector)
        franka_posrotvec = franka_feedback_state_dict['ActualTCPPose'] # (6, )
        ur_posrotvec_lis.append(franka_posrotvec) # list of (6, )
    
    # Use the defined initial pose
    init_pose = franka.get_state()['ActualTCPPose']

    start_time = time.monotonic()
    pos_amplitude = 0.02  # Position amplitude (meters)
    rot_amplitude = 0.02  # Rotation amplitude (radians)
    frequencies = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) / 10  # Frequency for each axis (Hz)
    phase_offsets = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])  # Phase offsets
    # Get rotation matrix of initial pose
    init_pose = franka.get_state()['ActualTCPPose']
    print(init_pose)
    init_rotation_vec = init_pose[3:6]
    rotation_matrix = R.from_rotvec(init_rotation_vec).as_matrix()
    
    from tqdm import tqdm
    time.sleep(5.0)
    for i in tqdm(range(1000)):
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
        target_pose = init_pose.copy()
        
        # Apply position offsets
        target_pose[0:3] = init_pose[0:3] + global_pos_offsets
        
        # Apply rotation offsets: first apply initial rotation, then apply additional rotation
        init_rotation = R.from_rotvec(init_pose[3:6])
        additional_rotation = R.from_rotvec(global_rot_offsets)
        combined_rotation = additional_rotation * init_rotation
        target_pose[3:6] = combined_rotation.as_rotvec()

        # Send command to Franka
        append_posrotvec(franka, tracker)
        
        franka.schedule_waypoint(
            pose=target_pose,
            target_time=time.time() + 1.0 / fps
        )

        time.sleep(1.0 / fps)
    
    # Return to initial position
    print("Motion ended, returning to initial position...")
    franka.schedule_waypoint(
        pose=init_pose,
        target_time=time.monotonic() + 2.0
    )
    time.sleep(2.0)
    
    print("6D synchronized sinusoidal motion completed!")
    
    tracker_posrotvec_np = np.stack(tracker_posrotvec_lis, axis=0) # (T, 6)
    franka_posrotvec_np = np.stack(ur_posrotvec_lis, axis=0) # (T, 6)
    
    os.makedirs("./tmp", exist_ok=True)
    np.savetxt(f"./tmp/tracker_posrotvec_np_cup.txt", tracker_posrotvec_np, fmt="%.6f")
    np.savetxt(f"./tmp/robot_posrotvec_np_cup.txt", franka_posrotvec_np, fmt="%.6f")
    exit()
    # tracker.stop()
    # franka.stop()

if __name__ == '__main__':
    main()
