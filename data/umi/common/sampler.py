"""
Source: https://github.com/real-stanford/universal_manipulation_interface
"""

from typing import Optional

import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st

from data.umi.common.replay_buffer import ReplayBuffer


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask

def resample_actions(
    actions: np.ndarray,
    timestamps: np.ndarray,
    resample_timestamps: np.ndarray,
    num_robot: int
):
    action_lst = []
    for robot_idx in range(num_robot):
        pos_interp = si.interp1d(
            x=timestamps,
            y=actions[..., 7 * robot_idx: 7 * robot_idx + 3],
            axis=0,
            assume_sorted=True
        )
        resampled_pos = pos_interp(resample_timestamps)
        
        rot_preprocess, rot_postprocess = \
            st.Rotation.from_rotvec, st.Rotation.as_rotvec
        rot_slerp = st.Slerp(
            times=timestamps,
            rotations=rot_preprocess(
                actions[..., 7 * robot_idx + 3: 7 * robot_idx + 6]
            )
        )
        resampled_rot = rot_postprocess(rot_slerp(resample_timestamps))
        
        gripper_width_interp = si.interp1d(
            x=timestamps,
            y=actions[..., 7 * robot_idx + 6: 7 * robot_idx + 7],
            axis=0,
            assume_sorted=True
        )
        resampled_gripper_width = gripper_width_interp(resample_timestamps)
        
        action_lst.append(np.concatenate(
            [resampled_pos, resampled_rot, resampled_gripper_width], axis=-1))
        
    return np.concatenate(action_lst, axis=-1)
        
        
class SequenceSampler:
    def __init__(self,
        shape_meta: dict,
        video_paths: list,
        replay_buffer: ReplayBuffer,
        rgb_keys: list,
        lowdim_keys: list,
        key_horizon: dict,
        key_latency_steps: dict,
        key_down_sample_steps: dict,
        episode_mask: Optional[np.ndarray] = None,
        action_padding: bool = False,
        resample_with_timestamp: bool = False,
    ):
        episode_ends = replay_buffer.episode_ends[:]

        # fecth all video_ids
        episode_video_ids = replay_buffer.meta.episode_video_id[:]
        assert len(episode_ends) == len(episode_video_ids), \
            "episode_ends and episode_video_ids should have the same length"
    
        indices = list()
        for i in range(len(episode_ends)):
            video_id = episode_video_ids[i].item()

            if video_paths is not None and (len(video_paths[video_id]) <= 0 or
                (episode_mask is not None and not episode_mask[i])
            ):
                # skip episode
                # print(f"Skip episode {i} due to empty video ({len(video_paths[i]) <= 0}) or mask")
                continue
            
            start_idx = 0 if i == 0 else episode_ends[i-1]
            end_idx = episode_ends[i]
            
            early_end_idx = end_idx
            if not action_padding:
                # ensure all the obs horizon = 1
                early_end_idx = max(start_idx, (end_idx - (key_horizon['action'] * key_down_sample_steps['action'])))
            
            indices.extend(
                [(current_idx, start_idx, end_idx, video_id)
                 for current_idx in range(start_idx, early_end_idx)]
            )

        # load low_dim to memory and keep rgb as compressed zarr array
        self.replay_buffer = dict()
        self.num_robot = 0
        for key in lowdim_keys:
            if key.endswith('eef_pos'):
                self.num_robot += 1
            
            self.replay_buffer[key] = replay_buffer[key][:]
        for key in rgb_keys:
            self.replay_buffer[key] = replay_buffer["rgb"]

        if 'action' in replay_buffer:
            self.replay_buffer['action'] = replay_buffer['action'][:]
        else:
            # construct action (concatenation of [eef_pos, eef_rot, gripper_width])
            actions = list()
            for robot_idx in range(self.num_robot):
                for cat in ['eef_pos', 'eef_rot_axis_angle', 'gripper_width']:
                    key = f'robot{robot_idx}_{cat}'
                    if key in self.replay_buffer:
                        actions.append(self.replay_buffer[key])
            self.replay_buffer['action'] = np.concatenate(actions, axis=-1)
        
        if 'action_token' in replay_buffer:
            self.replay_buffer['action_token'] = replay_buffer['action_token'][:]
            
        if 'gripper_width_mask' in replay_buffer:
            self.replay_buffer['gripper_width_mask'] = replay_buffer['gripper_width_mask'][:]

        if resample_with_timestamp and 'timestamp' in replay_buffer:
            self.replay_buffer['timestamp'] = replay_buffer['timestamp'][:]

        self.action_padding = action_padding
        self.indices = indices
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.resample_with_timestamp = resample_with_timestamp

        self.ignore_rgb_is_applied = False # speed up the interation when getting normalizaer


    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        current_idx, start_idx, end_idx, video_id = self.indices[idx]

        result = dict()
        result['video_id'] = video_id

        obs_keys = self.rgb_keys + self.lowdim_keys
        if self.ignore_rgb_is_applied:
            obs_keys = self.lowdim_keys

        # observation
        for key in obs_keys:
            input_arr = self.replay_buffer[key]
            this_horizon = self.key_horizon[key]
            this_latency_steps = self.key_latency_steps[key]
            this_downsample_steps = self.key_down_sample_steps[key]

            if key in self.rgb_keys:
                assert this_latency_steps == 0
                num_valid = min(this_horizon, (current_idx - start_idx) // this_downsample_steps + 1)
                slice_start = current_idx - (num_valid - 1) * this_downsample_steps

                output = input_arr[slice_start: current_idx + 1: this_downsample_steps]
                assert output.shape[0] == num_valid
                
                # solve padding
                if output.shape[0] < this_horizon:
                    padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                    output = np.concatenate([padding, output], axis=0)                    
                
            else:
                # Latency is not considered for now
                # idx_with_latency = np.array(
                #     [current_idx - idx * this_downsample_steps + this_latency_steps for idx in range(this_horizon)],
                #     dtype=np.float32)
                # idx_with_latency = idx_with_latency[::-1]
                # idx_with_latency = np.clip(idx_with_latency, start_idx, end_idx - 1)
                # interpolation_start = max(int(idx_with_latency[0]) - 5, start_idx)
                # interpolation_end = min(int(idx_with_latency[-1]) + 2 + 5, end_idx)

                # if 'rot' in key:
                #     # rotation
                #     rot_preprocess, rot_postprocess = None, None
                #     if key.endswith('quat'):
                #         rot_preprocess = st.Rotation.from_quat
                #         rot_postprocess = st.Rotation.as_quat
                #     elif key.endswith('axis_angle'):
                #         rot_preprocess = st.Rotation.from_rotvec
                #         rot_postprocess = st.Rotation.as_rotvec
                #     else:
                #         raise NotImplementedError
                #     slerp = st.Slerp(
                #         times=np.arange(interpolation_start, interpolation_end),
                #         rotations=rot_preprocess(input_arr[interpolation_start: interpolation_end]))
                #     output = rot_postprocess(slerp(idx_with_latency))
                # else:
                #     interp = si.interp1d(
                #         x=np.arange(interpolation_start, interpolation_end),
                #         y=input_arr[interpolation_start: interpolation_end],
                #         axis=0, assume_sorted=True)
                #     output = interp(idx_with_latency)
                # TODO(lingxuan): current implementation do not consider the latency
                output = input_arr[current_idx: current_idx + 1]

            result[key] = output

        # aciton
        input_arr = self.replay_buffer['action']
        action_horizon = self.key_horizon['action']
        action_latency_steps = self.key_latency_steps['action']
        assert action_latency_steps == 0
        action_down_sample_steps = self.key_down_sample_steps['action']
        # slice_end = min(end_idx, (current_idx + 1) + (action_horizon - 1) * action_down_sample_steps + 1)
        
        input_arr_slice = slice(
            current_idx + 1,
            min((current_idx + 1) + (action_horizon - 1) * action_down_sample_steps + 1, end_idx),
            action_down_sample_steps
        )
        output = input_arr[input_arr_slice]
        if self.resample_with_timestamp and 'timestamp' in self.replay_buffer:
            input_arr_slice_without_downsample = slice(
                current_idx + 1,
                (current_idx + 1) + (action_horizon - 1) * action_down_sample_steps + 1,
            )
            timestamps = self.replay_buffer['timestamp'][input_arr_slice_without_downsample]
            if (timestamps[-1] - timestamps[0] + 1) > len(timestamps):
                # if this chunk is slowed down with transformation
                output = resample_actions(
                    input_arr[input_arr_slice_without_downsample],
                    timestamps - timestamps[0],
                    np.arange(0, action_horizon) * action_down_sample_steps,
                    self.num_robot
                )
        
        # solve padding
        if not self.action_padding:
            assert output.shape[0] == action_horizon
        elif output.shape[0] < action_horizon:
            padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
            output = np.concatenate([output, padding], axis=0)
        result['action'] = output
        
        # Apply action history
        # history_arr_slice = slice(
        #     max(current_idx - ((action_horizon - 1) * action_down_sample_steps + 1), start_idx),
        #     current_idx,
        #     action_down_sample_steps
        # )
        # if current_idx == start_idx:
        #     history_arr_slice = slice(
        #         start_idx, start_idx + 1, action_down_sample_steps
        #     )
        # output = input_arr[history_arr_slice]
        # if output.shape[0] < action_horizon:
        #     padding = np.repeat(output[:1], action_horizon - output.shape[0], axis=0)
        #     output = np.concatenate([padding, output], axis=0)
        # result['history'] = output
        
        # action_token
        if 'action_token' in self.replay_buffer:
            input_arr = self.replay_buffer['action_token']
            output = input_arr[current_idx]
            result['action_token'] = output

        # gripper valid mask
        result['gripper_valid'] = True
        if 'gripper_width_mask' in self.replay_buffer:
            input_arr = self.replay_buffer['gripper_width_mask']
            result['gripper_valid'] = np.all(
                input_arr[current_idx: current_idx + 1]).item()

        return result

    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply