import os
import re
import time
from multiprocessing.managers import SharedMemoryManager
import sys
sys.path.append('.')
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
from vqvae.models.multivqvae import MultiVQVAE
from utils import batch_predict_action

OmegaConf.register_new_resolver("eval", eval, replace=True)

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

# TODO: add support for the third camera from reals
@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--vae_path', '-v', required=True, help='Path to VAE checkpoint')
@click.option('--data_config', '-dc', required=True, help='Path to data_config yaml file')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--steps_per_inference', '-si', default=24, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=30, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--instruction', type=str, default=None)
@click.option('--codec', type=str, default='ffv1')
@click.option('--binarize_gripper', '-bg', is_flag=True, default=False, help="Binarize gripper action.")
# Add global cache for models
def main(
    input, output, vae_path, data_config, robot_config,
    steps_per_inference, max_duration,
    frequency, command_latency,
    instruction, codec, binarize_gripper,
):
    # Initialize cache attributes
    if not hasattr(main, '_cached_processor'):
        main._cached_processor = None
    if not hasattr(main, '_cached_model'):
        main._cached_model = None
        main._cached_model_path = None
    if not hasattr(main, '_cached_vae'):
        main._cached_vae = None
        main._cached_vae_path = None
    if not hasattr(main, '_cached_normalizer'):
        main._cached_normalizer = None
        main._cached_normalizer_path = None
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
    
    filenames = {
        "camera0_rgb": rf"right_umi_[^_]+_mvs_mono_[^_]+_{codec}\.mkv$",
        "camera1_rgb": rf"left_umi_[^_]+_mvs_mono_[^_]+_{codec}\.mkv$"
    }

    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))

    # load left-right robot relative transform
    tx_left_right = np.array(robot_config_data['tx_left_right'])
    tx_robot1_robot0 = tx_left_right
    
    tx_tracker_to_tcp = np.array(robot_config_data['tx_tracker_to_tcp'])

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
            
            # Use a global cache for models to avoid reloading
            # First define the cache directory for transformers models
            os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface/transformers")
            
            # Cache model loading for processor and model
            processor_path = "Qwen/Qwen2.5-VL-7B-Instruct"
            if True or not hasattr(main, '_cached_processor') or main._cached_processor is None:
                print("Loading processor from scratch...")
                main._cached_processor = AutoProcessor.from_pretrained(
                    processor_path, use_fast=False, local_files_only=True)
            else:
                print("Using cached processor")
            processor = main._cached_processor
            
            # Load model with caching
            if True or not hasattr(main, '_cached_model') or main._cached_model is None or main._cached_model_path != input:
                print("Loading model from scratch...")
                main._cached_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    input,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=device
                )
                main._cached_model_path = input
            else:
                print("Using cached model")
            model = main._cached_model

            model.eval()
            
            # Load VAE with caching
            if not hasattr(main, '_cached_vae') or main._cached_vae is None or main._cached_vae_path != vae_path:
                print("Loading VAE from scratch...")
                main._cached_vae = MultiVQVAE.from_pretrained(vae_path)
                main._cached_vae.eval()
                main._cached_vae.to(device, dtype=torch.float32)
                main._cached_vae_path = vae_path
            else:
                print("Using cached VAE")
            vae = main._cached_vae
            
            valid_action_id_length = (
                vae.pos_id_len + vae.rot_id_len + vae.grip_id_len
            )

            normalizer_path = os.path.join(os.path.dirname(input), 'umi_vq_normalizer.pt')
            # Cache normalizer as well
            if not hasattr(main, '_cached_normalizer') or main._cached_normalizer is None or main._cached_normalizer_path != normalizer_path:
                print("Loading normalizer from scratch...")
                main._cached_normalizer = LinearNormalizer.load(normalizer_path)
                main._cached_normalizer_path = normalizer_path
            else:
                print("Using cached normalizer")
            normalizer = main._cached_normalizer

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
                result = batch_predict_action(
                    model, processor, vae, normalizer,
                    examples=[{
                        "obs": obs_dict,    
                        "meta": {
                            "num_camera": len(cameras_config)
                        }
                    }],
                    valid_action_id_length=valid_action_id_length,
                    apply_jpeg_compression=True,
                    instruction=instruction
                )
                action = result['action_pred'][0].detach().to('cpu').numpy()
                # support unimanual manipulation
                if len(robots_config) < 2:
                    action = action[..., : 10 * len(robots_config)]
                
                assert action.shape[-1] == 10 * len(robots_config)
                action = convert_policy_to_tcp_space(
                    action, T_tracker_to_tcp=tx_tracker_to_tcp)

                action = get_real_umi_action(action, obs, action_pose_repr)
                assert action.shape[-1] == 7 * len(robots_config)
                del result

            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                robot_states = env.get_robot_state()
                target_pose = np.stack([rs['TargetTCPPose'] for rs in robot_states])
                print("[DBG] ", target_pose)

                gripper_states = env.get_gripper_state()
                gripper_target_pos = np.asarray([gs['gripper_position'] for gs in gripper_states])
                print("[DBG] ", gripper_target_pos)

                control_robot_idx_list = [0]

                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
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
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            start_policy = True
                            
                    if start_policy:
                        break

                    precise_wait(t_sample)

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
                            result = batch_predict_action(
                                model, processor, vae, normalizer,
                                examples=[{
                                    "obs": obs_dict,
                                    "meta": {
                                        "num_camera": len(cameras_config)
                                    }
                                }],
                                valid_action_id_length=valid_action_id_length,
                                apply_jpeg_compression=True,
                                instruction=instruction
                            )
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            # print("[DBG] ", raw_action)
                            # assert False
                            
                            # support unimanual manipulation
                            if len(robots_config) < 2:
                                raw_action = raw_action[..., : 10 * len(robots_config)]
                            raw_action = convert_policy_to_tcp_space(
                                raw_action, T_tracker_to_tcp=tx_tracker_to_tcp)
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

                        # # deal with timing
                        # # the same step actions are always the target for
                        # action_timestamps = (np.arange(1, len(action) + 1, dtype=np.float64)
                        #     ) * dt + obs_timestamps[-1] # FIXME: do latency matching

                        # following code is for debug without latency matching
                        action_timestamps = (np.arange(1, len(action) + 1, dtype=np.float64)
                            ) * dt + time.time() # FIXME: alternative without latency matching (for debug)

                        # print(action_timestamps)
                        # print(t_cycle_end)

                        # print(dt)
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
                        
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                # Stop episode
                                # Hand control back to human
                                print('Stopped.')
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if stop_episode:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        time.sleep(1)
                        # Wait for Gripper to reach
                        reached = [False] * len(robots_config)
                        while True:
                            obs = env.get_obs()
                            for robot_id in range(len(robots_config)):
                                reached[robot_id] = obs[f'robot{robot_id}_gripper_reached'][-1]
                            if all(reached):
                                break
                            time.sleep(1 / frequency)
                        iter_idx += steps_per_inference
                        
                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()

                print("Stopped.")
# %%
if __name__ == '__main__':
    main()
