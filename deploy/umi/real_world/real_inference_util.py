from typing import Dict, Callable, Tuple, List
import numpy as np
import collections
from data.umi.common.cv2_util import get_image_transform
from data.umi.common.pose_repr_util import convert_pose_mat_rep
from data.umi.pose_util import (
    pose_to_mat, mat_to_pose, 
    mat_to_pose10d, pose10d_to_mat)

def get_real_obs_resolution(
    shape_meta: dict
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Get the input and output resolution of the real-world observation.
    Args:
        shape_meta: The shape meta of the observation.
    Returns:
        in_res: The input resolution of the observation (width, height).
        out_res: The output resolution of the observation (width, height).
    """
    in_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('raw_shape')
        if type == 'rgb':
            ci,hi,wi = shape
            if in_res is None:
                in_res = (wi, hi)
            assert in_res == (wi, hi)
            
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return in_res, out_res


def get_real_umi_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        obs_pose_repr: str='abs',
        tx_robot1_robot0: np.ndarray=None,
        episode_start_pose: List[np.ndarray]=None,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    # process non-pose
    obs_shape_meta = shape_meta['obs']
    robot_prefix_map = collections.defaultdict(list)
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            # if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
            #     tf = get_image_transform(
            #         input_res=(wi,hi), 
            #         output_res=(wo,ho), 
            #         bgr_to_rgb=False)
            #     out_imgs = np.stack([tf(x) for x in this_imgs_in])
            #     if this_imgs_in.dtype == np.uint8:
            #         out_imgs = out_imgs.astype(np.float32) / 255
            obs_dict_np[key] = out_imgs # THWC
        elif type == 'low_dim' and ('eef' not in key):
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
            # handle multi-robots
            ks = key.split('_')
            if ks[0].startswith('robot'):
                robot_prefix_map[ks[0]].append(key)

    # generate relative pose
    # for robot_prefix in robot_prefix_map.keys():
    #     # convert pose to mat
    #     pose_mat = pose_to_mat(np.concatenate([
    #         env_obs[robot_prefix + '_eef_pos'],
    #         env_obs[robot_prefix + '_eef_rot_axis_angle']
    #     ], axis=-1))

    #     # solve reltaive obs
    #     obs_pose_mat = convert_pose_mat_rep(
    #         pose_mat, 
    #         base_pose_mat=pose_mat[-1],
    #         pose_rep=obs_pose_repr,
    #         backward=False)

    #     obs_pose = mat_to_pose10d(obs_pose_mat)
    #     obs_dict_np[robot_prefix + '_eef_pos'] = obs_pose[...,:3]
    #     obs_dict_np[robot_prefix + '_eef_rot_axis_angle'] = obs_pose[...,3:]
    
    # generate pose relative to other robot
    # n_robots = len(robot_prefix_map)
    # for robot_id in range(n_robots):
    #     # convert pose to mat
    #     assert f'robot{robot_id}' in robot_prefix_map
    #     tx_robota_tcpa = pose_to_mat(np.concatenate([
    #         env_obs[f'robot{robot_id}_eef_pos'],
    #         env_obs[f'robot{robot_id}_eef_rot_axis_angle']
    #     ], axis=-1))
    #     for other_robot_id in range(n_robots):
    #         if robot_id == other_robot_id:
    #             continue
    #         tx_robotb_tcpb = pose_to_mat(np.concatenate([
    #             env_obs[f'robot{other_robot_id}_eef_pos'],
    #             env_obs[f'robot{other_robot_id}_eef_rot_axis_angle']
    #         ], axis=-1))
    #         tx_robota_robotb = tx_robot1_robot0
    #         if robot_id == 0:
    #             tx_robota_robotb = np.linalg.inv(tx_robot1_robot0)
    #         tx_robota_tcpb = tx_robota_robotb @ tx_robotb_tcpb

    #         rel_obs_pose_mat = convert_pose_mat_rep(
    #             tx_robota_tcpa,
    #             base_pose_mat=tx_robota_tcpb[-1],
    #             pose_rep='relative',
    #             backward=False)
    #         rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
    #         obs_dict_np[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[:,:3]
    #         obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_obs_pose[:,3:]

    # generate relative pose with respect to episode start
    # if episode_start_pose is not None:
    #     for robot_id in range(n_robots):        
    #         # convert pose to mat
    #         pose_mat = pose_to_mat(np.concatenate([
    #             env_obs[f'robot{robot_id}_eef_pos'],
    #             env_obs[f'robot{robot_id}_eef_rot_axis_angle']
    #         ], axis=-1))
            
    #         # get start pose
    #         start_pose = episode_start_pose[robot_id]
    #         start_pose_mat = pose_to_mat(start_pose)
    #         rel_obs_pose_mat = convert_pose_mat_rep(
    #             pose_mat,
    #             base_pose_mat=start_pose_mat,
    #             pose_rep='relative',
    #             backward=False)
            
    #         rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
    #         # obs_dict_np[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
    #         obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

    return obs_dict_np

def get_real_umi_action(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray], 
        action_pose_repr: str='abs'
    ):

    n_robots = int(action.shape[-1] // 10)
    env_action = list()
    for robot_idx in range(n_robots):
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([
            env_obs[f'robot{robot_idx}_eef_pos'][-1],
            env_obs[f'robot{robot_idx}_eef_rot_axis_angle'][-1]
        ], axis=-1))

        start = robot_idx * 10
        action_pose10d = action[..., start:start+9]
        action_grip = action[..., start+9:start+10]
        action_pose_mat = pose10d_to_mat(action_pose10d)

        # solve relative action
        action_mat = convert_pose_mat_rep(
            action_pose_mat, 
            base_pose_mat=pose_mat,
            pose_rep=action_pose_repr,
            backward=True)

        # convert action to pose
        action_pose = mat_to_pose(action_mat)
        env_action.append(action_pose)
        env_action.append(action_grip)

    env_action = np.concatenate(env_action, axis=-1)
    return env_action


def convert_policy_to_tcp_space(
    action: np.ndarray,
    T_tracker_to_policy=np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0.078],
        [0, 0, 1, -0.026705],
        [0, 0, 0, 1],
    ]),
    T_tracker_to_tcp=np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0.070261],
        [0, 0, 1, 0.272602],
        [0, 0, 0, 1],
    ]),
    backward=False
):
    n_robots = int(action.shape[-1] // 10)
    converted_action = list()
    for robot_idx in range(n_robots):
        start = robot_idx * 10
        action_pose10d = action[..., start:start+9]
        action_grip = action[..., start+9:start+10]
        action_pose_mat = pose10d_to_mat(action_pose10d)
        composed_T = np.linalg.inv(T_tracker_to_tcp) @ T_tracker_to_policy
        if backward:
            composed_T = np.linalg.inv(composed_T)
        
        action_mat = composed_T @ action_pose_mat @ np.linalg.inv(composed_T)
        action_pose = mat_to_pose10d(action_mat)
        
        converted_action.append(action_pose)
        converted_action.append(action_grip)
        
    converted_action = np.concatenate(converted_action, axis=-1)
        
    return converted_action