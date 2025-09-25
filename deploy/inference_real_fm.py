"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly!

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import os
import sys
sys.path.append('.')
import re
import time
from multiprocessing.managers import SharedMemoryManager

# import av
import json
import click
import cv2
import yaml
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image, ImageTk
import tkinter as tk

from models.normalizer import LinearNormalizer
from data.umi.common.pytorch_util import dict_apply
from deploy.collision_utils import solve_table_collision
from deploy.umi.common.precise_sleep import precise_wait
from deploy.umi.real_world.bimanual_umi_env import BimanualUmiEnv
from deploy.umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from deploy.umi.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_umi_obs_dict,
    get_real_umi_action,
    convert_policy_to_tcp_space
    )
# from deploy.umi.real_world.spacemouse_shared_memory import Spacemouse
from data.umi.pose_util import pose_to_mat, mat_to_pose, pose10d_to_mat, mat_to_pose10d
from models.rdt_inferencer import RDTInferencer

OmegaConf.register_new_resolver("eval", eval, replace=True)

actions_records_save_path = 'actions_records'
all_grippers = []
actions_submit_time = []

def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('[WARN] avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal

                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--pretrained_vision_language_model_name_or_path', '-vlm', required=True, help='Path to VAE checkpoint')
@click.option('--normalizer_path', '-np', required=True, help='Path to normalizer checkpoint')
@click.option('--model_config', '-mc', required=True, help='Path to model_config yaml file')
@click.option('--data_config', '-dc', required=True, help='Path to data_config yaml file')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--steps_per_inference', '-si', default=24, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=30, type=float, help="Control frequency in Hz.")
@click.option('--instruction', type=str, default=None)
@click.option('--binarize_gripper', '-bg', is_flag=True, default=False, help="Binarize gripper action.")
@click.option('--interact', is_flag=True, default=False, help="Interactive mode.")
# Add global cache for models
def main(
    input, output, 
    pretrained_vision_language_model_name_or_path, 
    normalizer_path, model_config,
    data_config, robot_config,
    steps_per_inference, max_duration,
    frequency,
    instruction, binarize_gripper,
    interact
):
    # Create tkinter window for visualization
    root = tk.Tk()
    root.title("Visualization")
    label = tk.Label(root)
    label.pack()
    
    def update_image(img):
        """
        Args:
            img: (H, W, 3) uint8
        """
        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(img)
        # Convert PIL Image to PhotoImage
        photo = ImageTk.PhotoImage(image=pil_img)
        # Update label
        label.configure(image=photo)
        label.image = photo  # Keep a reference
        root.update()  # Update the window
    
    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))

    # load left-right robot relative transform
    tx_left_right = np.array(robot_config_data['tx_left_right'])
    tx_robot1_robot0 = tx_left_right

    cameras_config = robot_config_data['cameras']
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']

    # load checkpoint
    cfg = OmegaConf.load(data_config)
    if len(robots_config) < 2:
        # del dummy keys for unimanual compatibility
        keys_to_del = [
            "robot1_eef_pos",
            "robot1_eef_rot_axis_angle",
            "robot1_gripper_width",
        ]
        for key in keys_to_del:
            del cfg.task.shape_meta.obs[key]

    print("model_path:", input)

    # setup experiment
    dt = 1/frequency

    raw_obs_res, obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    for cam_id, cam_kwargs in enumerate(cameras_config):
        cam_kwargs['input_res'] = raw_obs_res
        cam_kwargs['output_res'] = obs_res

    print("steps_per_inference:", steps_per_inference)
    with SharedMemoryManager() as shm_manager:
        # with Spacemouse(shm_manager=shm_manager) as sm, \
        with KeystrokeCounter() as key_counter, \
            BimanualUmiEnv(
                output_dir=output,
                cameras_config=cameras_config,
                robots_config=robots_config,
                grippers_config=grippers_config,
                frequency=frequency,
                # latency
                camera_obs_latency=0.07,
                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                # action
                max_pos_speed=2.0,
                max_rot_speed=6.0,
                shm_manager=shm_manager) as env:
            # print("[DBG] Env creaated")
            cv2.setNumThreads(1)
            print("Waiting for camera")
            time.sleep(1.0)

            print("Loading model and processor...")
            # Check if models are already cached in memory
            device = torch.device('cuda')
            
            with open(model_config, "r") as f:
                model_config = yaml.safe_load(f)
            
            model = RDTInferencer(
                config=model_config,
                pretrained_path=input,
                normalizer_path=normalizer_path,
                pretrained_vision_language_model_name_or_path=pretrained_vision_language_model_name_or_path,
                device=device,
                dtype=torch.bfloat16,
            )
            # ensure the model is reset
            
            print("Model loaded")

            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)

            print("Warming up policy inference")
            time.sleep(3.0)
            obs = env.get_obs()
            episode_start_pose = list()
            for robot_id in range(len(robots_config)):
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                episode_start_pose.append(pose)
            with torch.no_grad():
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta,
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=tx_robot1_robot0,
                    episode_start_pose=episode_start_pose)
                obs_dict = dict_apply(obs_dict_np,
                    lambda x: torch.from_numpy(x).to(device))
                print(f"Instruction: {instruction}")
                result = model.step(
                    observations={
                        'images': {
                            # 'exterior_rs': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                            'left_stereo': obs_dict['camera1_rgb'][-1].cpu().numpy(),
                            'right_stereo': obs_dict['camera0_rgb'][-1].cpu().numpy(),
                        },
                        'state': np.zeros(model_config["common"]["state_dim"]).astype(np.float32)
                    },
                    instruction=instruction
                )
                action = result.detach().cpu().numpy()
                # support unimanual manipulation
                if len(robots_config) < 2:
                    action = action[..., : 10 * len(robots_config)]
                
                assert action.shape[-1] == 10 * len(robots_config)
                action = convert_policy_to_tcp_space(action)

                action = get_real_umi_action(action, obs, action_pose_repr)
                assert action.shape[-1] == 7 * len(robots_config)
                del result

            print('Ready!')
            waiting_for_instruction = False
            while True:
                control_robot_idx_list = [0,1]

                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    obs_left_img = vis_img = np.concatenate(
                        [obs[f'camera{cam_id}_rgb'][-1] for cam_id in range(len(cameras_config))],
                        axis=1)

                    vis_img = np.concatenate([obs_left_img, vis_img], axis=1)

                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255, 255, 255)
                    )
                    update_image(vis_img)
                    
                    press_events = key_counter.get_press_events()
                    start_policy = True
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            env.end_episode()
                            exit(0)
                            
                    if start_policy:
                        break
                    
                    # solve collision with table
                    for robot_idx in control_robot_idx_list:
                        solve_table_collision(
                            ee_pose=target_pose[robot_idx],
                            gripper_width=gripper_target_pos[robot_idx],
                            height_threshold=robots_config[robot_idx]['height_threshold'],
                            finger_thickness=grippers_config[robot_idx]['finger_thickness']
                        )

                    # solve collison between two robots
                    solve_sphere_collision(
                        ee_poses=target_pose,
                        robots_config=robots_config
                    )

                    action = np.zeros((7 * target_pose.shape[0],))

                    for robot_idx in range(target_pose.shape[0]):
                        action[7 * robot_idx + 0: 7 * robot_idx + 6] = target_pose[robot_idx]
                        action[7 * robot_idx + 6] = gripper_target_pos[robot_idx]
                             
                    # execute teleop command
                    env.exec_actions(
                        actions=[action],
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        compensate_latency=False)
                    precise_wait(t_cycle_end)
                    iter_idx += 1

                # ========== policy control loop ==============
                try:
                    # start episode
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    # get current pose
                    obs = env.get_obs()
                    episode_start_pose = list()
                    for robot_id in range(len(robots_config)):
                        pose = np.concatenate([
                            obs[f'robot{robot_id}_eef_pos'],
                            obs[f'robot{robot_id}_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        episode_start_pose.append(pose)

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta,
                                obs_pose_repr=obs_pose_rep,
                                tx_robot1_robot0=tx_robot1_robot0,
                                episode_start_pose=episode_start_pose)
                            obs_dict = dict_apply(obs_dict_np,
                                lambda x: torch.from_numpy(x).to(device))
                            result = model.step(
                                observations={
                                    'images': {
                                        # 'exterior_rs': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                                        'left_stereo': obs_dict['camera1_rgb'][-1].cpu().numpy(),
                                        'right_stereo': obs_dict['camera0_rgb'][-1].cpu().numpy(),
                                    },
                                    'state': np.zeros(model_config["common"]["state_dim"]).astype(np.float32)
                                },
                                instruction=instruction
                            )
                            # ensure the action is unormalized
                            raw_action = result.detach().cpu().numpy()
                            
                            # support unimanual manipulation
                            if len(robots_config) < 2:
                                raw_action = raw_action[..., : 10 * len(robots_config)]
                            raw_action = convert_policy_to_tcp_space(raw_action)
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)
                            
                            # rescale gripper to real gripper width
                            for robot_idx in range(len(robots_config)):
                                action[:, robot_idx * 7 + 6] = action[:, robot_idx * 7 + 6] / 0.088 * 0.10
                            
                            print('Inference latency:', time.time() - s)
                            t_cycle_end += time.time() - s  # FIXME：do latency matching
                        # NOTE pre-closure
                        for robot_id in range(len(robots_config)):
                            action[:, robot_idx * 7 + 6] = action[:, robot_idx * 7 + 6] / 0.088 * 0.10
                            if action[0, robot_id * 7 + 6] > action[-1, robot_id * 7 + 6]:
                                closure_delta = action[0, robot_id * 7 + 6] - action[-1, robot_id * 7 + 6]
                                action[:, robot_id * 7 + 6] -= np.maximum(closure_delta * 0.2, 0.010)
                        
                        if binarize_gripper:
                            def soft_mono_binarize(open_vec, thredshold, exp):
                                return np.where(open_vec < thredshold, thredshold * (open_vec / thredshold) ** exp, open_vec)
                            for robot_idx in range(len(robots_config)):
                                action[:, robot_idx * 7 + 6] = action[:, robot_idx * 7 + 6] / 0.088 * 0.10
                                action[:, robot_idx * 7 + 6] = soft_mono_binarize(
                                    action[:, robot_idx * 7 + 6], 0.04, 50)
                                
                        # convert policy action to env actions
                        this_target_poses = action
                        assert this_target_poses.shape[1] == len(robots_config) * 7
                        for target_pose in this_target_poses:
                            for robot_idx in range(len(robots_config)):
                                solve_table_collision(
                                    ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                                    gripper_width=target_pose[robot_idx * 7 + 6],
                                    height_threshold=robots_config[robot_idx]['height_threshold'],
                                    finger_thickness=grippers_config[robot_idx]['finger_thickness']
                                )

                            # solve collison between two robots
                            solve_sphere_collision(
                                ee_poses=target_pose.reshape([len(robots_config), -1]),
                                robots_config=robots_config
                            )

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(1, len(action) + 1, dtype=np.float64)
                            ) * dt + obs_timestamps[-1] # FIXME: do latency matching
                        
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]
                        
                        # execute actions
                        actions_submit_time.append(time.time())
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True    # FIXME: do latency matching
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")
                        
                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = obs['camera0_rgb'][-1]
                        if len(cameras_config) > 1:
                            vis_img = np.concatenate([vis_img, obs['camera1_rgb'][-1]], axis=1)
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255, 255, 255)
                        )
                        update_image(vis_img)
                        
                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)

                        reached = [False] * len(robots_config)
                        while True:
                            obs = env.get_obs()
                            for robot_id in range(len(robots_config)):
                                reached[robot_id] = obs[f'robot{robot_id}_gripper_reached'][-1]
                                # print(f"Robot {robot_id} reached: {reached[robot_id]}")
                            if all(reached):
                                break
                            time.sleep(1 / frequency)
                        iter_idx += steps_per_inference
                        
                        press_events = key_counter.get_press_events()

                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s') and interact and not waiting_for_instruction:
                                waiting_for_instruction = True
                                try:
                                    print("Enter new instruction (press Enter to confirm):")
                                    key_counter.get_press_events() 
                                    new_instruction = builtins.input("New instruction: ")
                                    if new_instruction.strip():
                                        instruction = new_instruction.strip()
                                        print(f"Updated instruction to: {instruction}")
                                    else:
                                        print("No instruction provided, keeping current instruction.")
                                except (EOFError, KeyboardInterrupt):
                                    print("Input cancelled, keeping current instruction.")
                                finally:
                                    waiting_for_instruction = False
                                    key_counter.get_press_events()  

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()

                print("Stopped.")
# %%
if __name__ == '__main__':
    main()
