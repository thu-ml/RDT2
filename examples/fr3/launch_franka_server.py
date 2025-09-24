import zerorpc
from polymetis import RobotInterface, GripperInterface
import scipy.spatial.transform as st
import numpy as np
import torch
import click
import time

# FIXME: modify the reset joint according to your configuration
franka_left_home_joint = torch.tensor([0.790078, 0.741306, -1.13052, -2.77817, -0.124533, 4.41087, 2.21978])
franka_right_home_joint = torch.tensor([-0.6994, 0.623364, 0.777071, -2.69294, -0.274674, 4.33062, 1.15367])
class FrankaInterface:
    
    def __init__(self, franka_ip="192.168.3.100", poly_serve_port=50051):
        print(f"Connecting to Franka robot at {franka_ip} port {poly_serve_port}...")
        self.robot = RobotInterface(ip_address=franka_ip, port=poly_serve_port)
        if poly_serve_port == 50051: # right arm
            self.robot.set_home_pose(franka_right_home_pose)
        elif poly_serve_port == 50052: # left arm
            self.robot.set_home_pose(franka_left_home_pose)
        self.go_home()

    ###
    # franka
    ###
    def go_home(self):
        start_time = time.monotonic()
        self.robot.go_home()
    
    # joint
    def start_joint_impedance(self):
        start_time = time.monotonic()
        print(f"Starting joint impedance control for Franka robot...")
        self.robot.start_joint_impedance()
        end_time = time.monotonic()
        print(f"Joint impedance control started for Franka robot: {self.robot}, time taken: {end_time - start_time:.4f} seconds")
    
    def get_joint_positions(self):
        return self.robot.get_joint_positions().numpy().tolist()
    
    def get_joint_velocities(self):
        return self.robot.get_joint_velocities().numpy().tolist()
    
    def update_desired_joint_positions(self, positions):
        self.robot.update_desired_joint_positions(
            positions=torch.Tensor(positions)
        )

    # eef
    def start_cartesian_impedance(self, Kx=None, Kxd=None):
        start_time = time.monotonic()
        print(f"Starting Cartesian impedance control for Franka robot at {start_time:.4f} seconds")
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(Kx) if Kx is not None else None,
            Kxd=torch.Tensor(Kxd) if Kxd is not None else None,
        )
        print(f"Started Cartesian impedance control for Franka robot, time taken: {time.monotonic() - start_time:.4f} seconds")
    
    def get_ee_pose(self):
        data = self.robot.get_ee_pose()
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist()

    def update_desired_ee_pose(self, pose):''
        pose = np.asarray(pose)
        self.robot.update_desired_ee_pose(
            position=torch.Tensor(pose[:3]),
            orientation=torch.Tensor(st.Rotation.from_rotvec(pose[3:]).as_quat())
        )
        

@click.command()
@click.option('--franka_ip', default='localhost', help='IP address of the Franka robot')
@click.option('--serve_ip', default='0.0.0.0', help='IP address for the ZeroRPC server')
@click.option('--serve_port', default=4242, help='Port for the ZeroRPC server')
@click.option('--poly_serve_port', default=50051, help='Port for the PolyRobot gRPC server')
def main(
    franka_ip: str,
    serve_ip: str,
    serve_port: int,
    poly_serve_port: int,
):
    s = zerorpc.Server(FrankaRobotZeroRPC(franka_ip=franka_ip, poly_serve_port=poly_serve_port))
    server_address = f"tcp://{serve_ip}:{serve_port}"
    print(f"Starting Franka interface server on {server_address}...")
    s.bind(server_address)
    s.run()

if __name__ == "__main__":
    main()
