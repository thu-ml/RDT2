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

def interpolate_action_data(action_data, threshold=0.0003):
    new_data = [action_data[0]]
    for i in range(1, len(action_data)):
        prev = new_data[-1]
        curr = action_data[i]

        # Position
        pos_prev, pos_curr = prev[:3], curr[:3]

        # Assume action_data[3:6] is rotation vector (axis-angle)
        rot_prev = R.from_rotvec(prev[3:6])
        rot_curr = R.from_rotvec(curr[3:6])

        # Calculate position and rotation differences
        dist = np.linalg.norm(pos_curr - pos_prev)
        angle = (rot_prev.inv() * rot_curr).magnitude()

        if angle > 1e-6:
            dist = max(dist, angle / 1e-3 * threshold)

        if dist <= threshold:
            new_data.append(curr)
        else:
            num_insert = int(np.ceil(dist / threshold))

            # Prepare Slerp interpolator
            slerp = Slerp([0, 1], R.concatenate([rot_prev, rot_curr]))

            for j in range(1, num_insert):
                alpha = j / num_insert
                # Linear interpolation for position
                pos_interp = (1 - alpha) * pos_prev + alpha * pos_curr
                # Spherical interpolation for rotation
                rot_interp = slerp([alpha])[0]
                interp = np.concatenate([pos_interp, rot_interp.as_rotvec()])
                new_data.append(interp)

            new_data.append(curr)

    return np.array(new_data)

def load_data():
    filepath = "/home/franka/code/FastVLA/tmp/250911_195138_197898.h5"
    import h5py
    with h5py.File(filepath, "r") as f:
        eef_pos = f["robot1_eef_pos"][:]
        eef_rot = f["robot1_eef_rot_axis_angle"][:]
        action_data = np.concatenate([eef_pos, eef_rot], axis=-1)
    # Interpolate action_data
    print(f"Interpolating action_data, original shape: {action_data.shape}")
    raw_action_data = action_data.copy()
    action_data = interpolate_action_data(raw_action_data, threshold=0.0003)
    print(f"Replaying, new shape: {action_data.shape}")
    # Visualize raw_action_data and action_data, save as image
    import matplotlib.pyplot as plt
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Action Data Comparison: Raw vs Interpolated', fontsize=16)
    
    # Position data visualization
    for i in range(3):
        axes[0, i].plot(raw_action_data[:, i], 'b-', label='Raw', alpha=0.7)
        axes[0, i].plot(action_data[:, i], 'r-', label='Interpolated', alpha=0.7)
        axes[0, i].set_title(f'Position {["X", "Y", "Z"][i]}')
        axes[0, i].set_xlabel('Frame')
        axes[0, i].set_ylabel('Position (m)')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
    
    # Rotation data visualization
    for i in range(3):
        axes[1, i].plot(raw_action_data[:, i+3], 'b-', label='Raw', alpha=0.7)
        axes[1, i].plot(action_data[:, i+3], 'r-', label='Interpolated', alpha=0.7)
        axes[1, i].set_title(f'Rotation {["RX", "RY", "RZ"][i]}')
        axes[1, i].set_xlabel('Frame')
        axes[1, i].set_ylabel('Rotation (rad)')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save image
    os.makedirs("./tmp", exist_ok=True)
    plt.savefig('./tmp/action_data_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Action data visualization saved to ./tmp/action_data_comparison.png")
    plt.close()
    
    # Additional distance analysis plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calculate Euclidean distance between adjacent frames
    raw_pos_dists = np.linalg.norm(np.diff(raw_action_data[:, :3], axis=0), axis=1)
    interp_pos_dists = np.linalg.norm(np.diff(action_data[:, :3], axis=0), axis=1)
    
    raw_rot_dists = np.linalg.norm(np.diff(raw_action_data[:, 3:6], axis=0), axis=1)
    interp_rot_dists = np.linalg.norm(np.diff(action_data[:, 3:6], axis=0), axis=1)
    print(f"Raw pos dists: {np.max(raw_pos_dists)}")
    print(f"Interpolated pos dists: {np.max(interp_pos_dists)}")
    print(f"Raw rot dists: {np.max(raw_rot_dists)}")
    print(f"Interpolated rot dists: {np.max(interp_rot_dists)}")
    # Position distance analysis
    ax1.plot(raw_pos_dists, 'b-', label='Raw', alpha=0.7)
    ax1.plot(interp_pos_dists, 'r-', label='Interpolated', alpha=0.7)
    ax1.axhline(y=0.0003, color='g', linestyle='--', label='Threshold (0.0003)')
    ax1.set_title('Position Distance Between Consecutive Frames')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Distance (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Rotation distance analysis
    ax2.plot(raw_rot_dists, 'b-', label='Raw', alpha=0.7)
    ax2.plot(interp_rot_dists, 'r-', label='Interpolated', alpha=0.7)
    ax2.set_title('Rotation Distance Between Consecutive Frames')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Distance (rad)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('./tmp/distance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Distance analysis saved to ./tmp/distance_analysis.png")
    plt.close()
    return raw_action_data

def main():
    action_data = load_data()
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
    for i in tqdm(range(action_data.shape[0])):
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

        target_pose = action_data[i]
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
