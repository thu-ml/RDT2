"""
python tests/test_ur_tracker_combine_cal.py

Use gradient descent to optimize calibration parameters from tracker to UR
"""

import sys
sys.path.append(".")

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import os


def convert_pose_mat_rep(pose_mat, base_pose_mat, pose_rep='relative', backward=False):
    if not backward:
        # training transform
        if pose_rep == 'abs':
            return pose_mat
        elif pose_rep == 'relative':
            out = np.linalg.inv(base_pose_mat) @ pose_mat
            return out
        elif pose_rep == 'delta':
            all_pos = np.concatenate([base_pose_mat[None,:3,3], pose_mat[...,:3,3]], axis=0)
            out_pos = np.diff(all_pos, axis=0)

            all_rot_mat = np.concatenate([base_pose_mat[None,:3,:3], pose_mat[...,:3,:3]], axis=0)
            # TODO: avoid heavy inverse computation
            prev_rot = np.linalg.inv(all_rot_mat[:-1])
            curr_rot = all_rot_mat[1:]
            out_rot = np.matmul(curr_rot, prev_rot)

            out = np.copy(pose_mat)
            out[...,:3,:3] = out_rot
            out[...,:3,3] = out_pos
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")

    else:
        # eval transform
        if pose_rep == 'abs':
            return pose_mat
        elif pose_rep == 'relative':
            out = base_pose_mat @ pose_mat
            return out
        elif pose_rep == 'delta':
            output_pos = np.cumsum(pose_mat[...,:3,3], axis=0) + base_pose_mat[:3,3]

            output_rot_mat = np.zeros_like(pose_mat[...,:3,:3])
            curr_rot = base_pose_mat[:3,:3]
            for i in range(len(pose_mat)):
                curr_rot = pose_mat[i,:3,:3] @ curr_rot
                output_rot_mat[i] = curr_rot

            out = np.copy(pose_mat)
            out[...,:3,:3] = output_rot_mat
            out[...,:3,3] = output_pos
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")

def mat_posrotvec(mat):
    """
    Args:
        mat: (T, 4, 4) numpy array, transformation matrix
    """
    T = mat.shape[0]
    posrotvec = np.zeros((T, 6), dtype=np.float32)
    for t in range(T):
        translation = mat[t, :3, 3]
        rotation_mat = mat[t, :3, :3]
        rotation_vec = R.from_matrix(rotation_mat).as_rotvec()
        posrotvec[t, :3] = translation
        posrotvec[t, 3:] = rotation_vec
    return posrotvec
def convert_tracker_pose_mat(
    pose_mat: np.ndarray, # (N, 4, 4)
    tcp_x_error: float = 0.0,
    tcp_y_error: float = 0.10,
    tcp_z_error: float = 0.24,
    cam_ur_error: float = 0.0,
):
    """
    Apply transformation to tracker pose matrices
    Same as in test_ur_tracker_combine_vis.py
    """
    
    x_theta = np.pi / 2
    x_rot_90_degree_mat = np.array([
        [1, 0, 0, 0],
        [0, np.cos(x_theta), -np.sin(x_theta), 0],
        [0, np.sin(x_theta), np.cos(x_theta), 0],
        [0, 0, 0, 1],
    ])

    pose_mat = pose_mat @ x_rot_90_degree_mat
    
    tx_tracker_to_tcp = np.array([
        [1, 0, 0, tcp_x_error],
        [0, 1, 0, (tcp_y_error + cam_ur_error)],
        [0, 0, 1, tcp_z_error],
        [0, 0, 0, 1],
    ])
    
    # T_1, T_2, T_3, ... in tcp frame
    pose_mat = pose_mat @ tx_tracker_to_tcp

    return pose_mat

def posrotvec_to_mat(posrotvec):
    """
    Convert position and rotation vector to transformation matrix
    Args:
        posrotvec: (T, 6) numpy array, [tx, ty, tz, rx, ry, rz]
    Returns:
        pose_mat: (T, 4, 4) transformation matrices
    """
    T = posrotvec.shape[0]
    pose_mat = np.zeros((T, 4, 4), dtype=np.float32)
    for t in range(T):
        translation = posrotvec[t, :3]
        rotation_vec = posrotvec[t, 3:]
        rotation_mat = R.from_rotvec(rotation_vec).as_matrix()
        pose_mat[t, :3, :3] = rotation_mat
        pose_mat[t, :3, 3] = translation
        pose_mat[t, 3, 3] = 1.0
    return pose_mat

def mat_to_posrotvec(mat):
    """
    Convert transformation matrix to position and rotation vector
    Args:
        mat: (T, 4, 4) numpy array, transformation matrix
    Returns:
        posrotvec: (T, 6) [tx, ty, tz, rx, ry, rz]
    """
    T = mat.shape[0]
    posrotvec = np.zeros((T, 6), dtype=np.float32)
    for t in range(T):
        translation = mat[t, :3, 3]
        rotation_mat = mat[t, :3, :3]
        rotation_vec = R.from_matrix(rotation_mat).as_rotvec()
        posrotvec[t, :3] = translation
        posrotvec[t, 3:] = rotation_vec
    return posrotvec

def calculate_errors(tracker_posrotvec_np, ur_posrotvec_np):
    """Calculate errors in six dimensions - only consider maximum position error"""
    # Position error (xyz) - use maximum absolute value instead of mean
    pos_error = np.mean((tracker_posrotvec_np[:, :3] - ur_posrotvec_np[:, :3]) ** 2)
    
    # Rotation error (for display, but not included in combined error calculation) - use maximum absolute value
    tracker_rotvec = tracker_posrotvec_np[:, 3:]
    ur_rotvec = ur_posrotvec_np[:, 3:]
    
    # Calculate rotation angle error (in radians) - use maximum absolute value
    rot_errors_rad = np.mean((tracker_rotvec - ur_rotvec) ** 2)
    
    return pos_error + rot_errors_rad

def _rmse_objective_factory(tracker_posrotvec_adjusted: np.ndarray, ur_posrotvec_adjusted: np.ndarray):
    """
    Create a smooth RMSE objective over position differences, using precomputed
    relative matrices. This makes gradients well-behaved for L-BFGS-B.
    Returns a function f(params)->rmse.
    Note: Match grid-search pipeline by using conjugation T^{-1} * B * T.
    """
    def f(params: np.ndarray) -> float:
        tcp_y_error, tcp_z_error = params
        
        # Transform tracker data
        tracker_mat = posrotvec_to_mat(tracker_posrotvec_adjusted)
        tracker_mat = convert_tracker_pose_mat(
            tracker_mat, 
            tcp_y_error=tcp_y_error, 
            tcp_z_error=tcp_z_error
        )
        tracker_mat = convert_pose_mat_rep(
            tracker_mat, base_pose_mat=tracker_mat[0],
            pose_rep="relative", backward=False
        )
        
        # Transform UR data
        ur_mat = posrotvec_to_mat(ur_posrotvec_adjusted)
        ur_rel_mat = convert_pose_mat_rep(
            ur_mat, base_pose_mat=ur_mat[0], 
            pose_rep="relative", backward=False)
        
        # Convert to posrotvec
        tracker_posrotvec_np = mat_posrotvec(tracker_mat)
        ur_posrotvec_np = mat_posrotvec(ur_rel_mat)
        
        # Calculate error
        error = calculate_errors(tracker_posrotvec_np, ur_posrotvec_np)

        return float(error)
    return f

def gradient_descent_optimize():
    """
    Use gradient descent (via scipy.optimize.minimize) to find optimal parameters
    """
    # Load data
    tracker_posrotvec = np.loadtxt(f"./tmp/tracker_posrotvec_np.txt")
    ur_posrotvec = np.loadtxt(f"./tmp/robot_posrotvec_np.txt")
    
    print("=== Gradient Descent Optimization ===")
    print(f"Tracker data points: {len(tracker_posrotvec)}")
    print(f"UR data points: {len(ur_posrotvec)}")
    
    # Ensure same length
    min_len = min(len(tracker_posrotvec), len(ur_posrotvec))
    tracker_posrotvec = tracker_posrotvec[:min_len]
    ur_posrotvec = ur_posrotvec[:min_len]
    
    # Initial guess (user can modify externally; default reasonable values)
    initial_params = np.array([0.064,  -0.33])
    
    # Bounds for parameters
    bounds = [(-1.0, 1.0), (-1.0,1.0)]
    
    print(f"\nInitial parameters: tcp_y={initial_params[0]:.3f}, tcp_z={initial_params[1]:.3f}")
    print(f"Parameter bounds: tcp_y=[{bounds[0][0]:.2f}, {bounds[0][1]:.2f}], tcp_z=[{bounds[1][0]:.2f}, {bounds[1][1]:.2f}]")
    

    # Smooth RMSE objective for stable gradients
    rmse_objective = _rmse_objective_factory(tracker_posrotvec, ur_posrotvec)

    # Calculate initial error (RMSE)
    initial_error = rmse_objective(initial_params)
    print(f"Initial error (RMSE): {initial_error:.10f}")
    
    print("\nStarting gradient descent optimization...")
    
    # Use L-BFGS-B method which supports bounds and is efficient for this problem
    result = minimize(
        rmse_objective,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'ftol': 1e-12,
            'gtol': 1e-12,
            'maxiter': 2000,
            'eps': 1e-3,  # finite-difference step for stable gradient estimation
            'disp': True
        }
    )
    
    optimal_params = result.x.copy()
    optimal_tcp_y, optimal_tcp_z = float(optimal_params[0]), float(optimal_params[1])
    
    print("\n=== Optimization Results ===")
    print(f"Optimal parameters:")
    print(f"  tcp_y_error: {optimal_tcp_y:.6f}")
    print(f"  tcp_z_error: {optimal_tcp_z:.6f}")
    print(f"  Optimization success: {result.success}")
    print(f"  Iterations: {result.nit}")
    print(f"  Function evaluations: {result.nfev}")
    print(f"  Final error (RMSE): {rmse_objective(optimal_params):.10f}")
    
    return optimal_tcp_y, optimal_tcp_z

def visualize_results(optimal_tcp_y, optimal_tcp_z):
    """
    Visualize the results with optimal parameters
    """
    # Load data
    tracker_posrotvec = np.loadtxt(f"./tmp/tracker_posrotvec_np.txt")
    ur_posrotvec = np.loadtxt(f"./tmp/robot_posrotvec_np.txt")
    
    # Ensure same length
    min_len = min(len(tracker_posrotvec), len(ur_posrotvec))
    tracker_posrotvec = tracker_posrotvec[:min_len]
    ur_posrotvec = ur_posrotvec[:min_len]
    
    # Apply transformation
    tracker_mat = posrotvec_to_mat(tracker_posrotvec)
    ur_mat = posrotvec_to_mat(ur_posrotvec)
    
    tracker_mat_transformed = convert_tracker_pose_mat(
        tracker_mat,
        tcp_x_error=0.0,
        tcp_y_error=optimal_tcp_y,
        tcp_z_error=optimal_tcp_z,
        cam_ur_error=0.0
    )
    tracker_mat_transformed = convert_pose_mat_rep(
        tracker_mat_transformed, base_pose_mat=tracker_mat_transformed[0],
        pose_rep="relative", backward=False
    )
    
    ur_rel_mat = convert_pose_mat_rep(
        ur_mat, base_pose_mat=ur_mat[0],
        pose_rep="relative", backward=False
    )
    
    # Convert to posrotvec
    tracker_posrotvec_np = mat_to_posrotvec(tracker_mat_transformed)
    ur_posrotvec_np = mat_to_posrotvec(ur_rel_mat)
    
    # Extract position and rotation
    tracker_pos = tracker_posrotvec_np[:, :3]
    tracker_rotvec = tracker_posrotvec_np[:, 3:]
    ur_pos = ur_posrotvec_np[:, :3]
    ur_rotvec = ur_posrotvec_np[:, 3:]
    
    # Convert to Euler angles
    tracker_euler = R.from_rotvec(tracker_rotvec).as_euler('xyz', degrees=True)
    ur_euler = R.from_rotvec(ur_rotvec).as_euler('xyz', degrees=True)
    
    time_steps = np.arange(len(tracker_posrotvec_np))
    
    # Create figure with subplots for positions
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Position Comparison (Gradient Descent: tcp_y={optimal_tcp_y:.4f}, tcp_z={optimal_tcp_z:.4f})', fontsize=14)
    
    # X position
    axes[0].plot(time_steps, tracker_pos[:, 0], label='Tracker X', linewidth=2, alpha=0.8)
    axes[0].plot(time_steps, ur_pos[:, 0], label='UR X', linewidth=2, alpha=0.8)
    axes[0].set_ylabel('X Position (m)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Y position
    axes[1].plot(time_steps, tracker_pos[:, 1], label='Tracker Y', linewidth=2, alpha=0.8)
    axes[1].plot(time_steps, ur_pos[:, 1], label='UR Y', linewidth=2, alpha=0.8)
    axes[1].set_ylabel('Y Position (m)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Z position
    axes[2].plot(time_steps, tracker_pos[:, 2], label='Tracker Z', linewidth=2, alpha=0.8)
    axes[2].plot(time_steps, ur_pos[:, 2], label='UR Z', linewidth=2, alpha=0.8)
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Z Position (m)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./tmp/positions_gradient_descent.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create figure with subplots for rotations
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Rotation Comparison (Gradient Descent: tcp_y={optimal_tcp_y:.4f}, tcp_z={optimal_tcp_z:.4f})', fontsize=14)
    
    # Roll
    axes[0].plot(time_steps, tracker_euler[:, 0], label='Tracker Roll', linewidth=2, alpha=0.8)
    axes[0].plot(time_steps, ur_euler[:, 0], label='UR Roll', linewidth=2, alpha=0.8)
    axes[0].set_ylabel('Roll (degrees)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Pitch
    axes[1].plot(time_steps, tracker_euler[:, 1], label='Tracker Pitch', linewidth=2, alpha=0.8)
    axes[1].plot(time_steps, ur_euler[:, 1], label='UR Pitch', linewidth=2, alpha=0.8)
    axes[1].set_ylabel('Pitch (degrees)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Yaw
    axes[2].plot(time_steps, tracker_euler[:, 2], label='Tracker Yaw', linewidth=2, alpha=0.8)
    axes[2].plot(time_steps, ur_euler[:, 2], label='UR Yaw', linewidth=2, alpha=0.8)
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Yaw (degrees)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./tmp/rotations_gradient_descent.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate and print mean errors for each axis
    pos_errors_mean = np.mean(np.abs(tracker_pos - ur_pos), axis=0)
    rot_errors_mean_rad = np.mean(np.abs(tracker_rotvec - ur_rotvec), axis=0)
    rot_errors_mean_deg = rot_errors_mean_rad * 180 / np.pi
    
    print(f"\nMean error for each axis:")
    print(f"Position error (m):")
    print(f"  X-axis: {pos_errors_mean[0]:.6f}")
    print(f"  Y-axis: {pos_errors_mean[1]:.6f}")
    print(f"  Z-axis: {pos_errors_mean[2]:.6f}")
    print(f"  Average: {np.mean(pos_errors_mean):.6f}")
    
    print(f"Rotation error (degrees):")
    print(f"  Roll (X): {rot_errors_mean_deg[0]:.6f}")
    print(f"  Pitch (Y): {rot_errors_mean_deg[1]:.6f}")
    print(f"  Yaw (Z): {rot_errors_mean_deg[2]:.6f}")
    print(f"  Average: {np.mean(rot_errors_mean_deg):.6f}")
    
    print(f"Rotation error (radians):")
    print(f"  Roll (X): {rot_errors_mean_rad[0]:.6f}")
    print(f"  Pitch (Y): {rot_errors_mean_rad[1]:.6f}")
    print(f"  Yaw (Z): {rot_errors_mean_rad[2]:.6f}")
    print(f"  Average: {np.mean(rot_errors_mean_rad):.6f}")
    
    # Calculate combined error (position + rotation in radians)
    combined_error = np.mean(pos_errors_mean) + np.mean(rot_errors_mean_rad)
    print(f"\nCombined error (position average + rotation average in radians): {combined_error:.6f}")
    
    print("\nVisualization results saved:")
    print("- ./tmp/positions_gradient_descent.png")
    print("- ./tmp/rotations_gradient_descent.png")

def main():
    """
    Main function using gradient descent optimization
    """
    # Run gradient descent optimization
    optimal_tcp_y, optimal_tcp_z = gradient_descent_optimize()
    
    # Visualize results
    # visualize_results(optimal_tcp_y, optimal_tcp_z)
    
    print("\nOptimization complete! All results saved to ./tmp directory")
    
    # Output the calibration matrix in the format used in config files
    print(f"\nCalibration matrix (tx_tracker_to_tcp):")
    print(f"  [")
    print(f"    [ 1.        ,  0.        ,  0.        ,  0.        ],")
    print(f"    [ 0.        ,  1.        ,  0.        ,  {optimal_tcp_y:.6f}  ],")
    print(f"    [ 0.        ,  0.        ,  1.        ,  {optimal_tcp_z:.6f}  ],")
    print(f"    [ 0.        ,  0.        ,  0.        ,  1.        ]")
    print(f"  ]")

if __name__ == "__main__":
    main()



