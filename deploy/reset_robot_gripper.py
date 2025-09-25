import sys
sys.path.append(".")
import os
import numpy as np
import yaml

from deploy.umi.real_world.zhixing_driver import ZhixingDriver
from deploy.umi.real_world.zhixing_controller import get_serial_dev

left_reset_joints_ur_deg = [82.89, -65.00, 140.96, -240.11, -50.19, -9.49]
right_reset_joints_ur_deg = [-102.00, -112.00, -138.89, 54.98, 81.91, -1.68]
left_reset_joints_ur = np.radians(left_reset_joints_ur_deg)
right_reset_joints_ur = np.radians(right_reset_joints_ur_deg)

left_reset_joints_franka = [0.790078, 0.741306, -1.13052, -2.77817, -0.124533, 4.41087, 2.21978]
right_reset_joints_franka = [-0.6994, 0.623364, 0.777071, -2.69294, -0.274674, 4.33062, 1.15367]

reset_joints_ur = [right_reset_joints_ur, left_reset_joints_ur]
reset_joints_franka = [right_reset_joints_franka, left_reset_joints_franka]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_config", type=str, required=True)
    args = parser.parse_args()
    robot_config_path = args.robot_config
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config_path), 'r'))
    robots_config = robot_config_data['robots']
    
    # Reset gripper
    serials = [gripper['serial'] for gripper in robot_config_data['grippers']]
    for serial in serials:
        with ZhixingDriver(serial_dev=get_serial_dev(serial)) as zx:
            zx.move_to(1.0 / 1.2)
    print("Gripper reset done.")
    
    # Reset Robot
    for robot_id, robot in enumerate(robots_config):
        if robot['robot_type'] == 'ur5e':
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
            rtde_c = RTDEControlInterface(hostname=robot['robot_ip'])
            rtde_r = RTDEReceiveInterface(hostname=robot['robot_ip'])
            rtde_c.moveJ(reset_joints_ur[robot_id], 1.05, 1.4)
        elif robot['robot_type'] == 'franka':
            # The Franka robot is automatically reset to the home pose
            pass
        else:
            raise ValueError(f"Robot type {robot['robot_type']} reset is not supported for now.")
    print("Robot reset done.")