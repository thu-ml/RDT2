import copy
import os
import json
import pathlib
import random
import sys
import shutil
from datetime import datetime
from typing import Any, Dict, Optional, Literal

import numpy as np
import torch
import zarr
import click
from filelock import FileLock
from moviepy import VideoFileClip
from omegaconf import OmegaConf
from PIL import Image
from threadpoolctl import threadpool_limits

from data.base_dataset import BaseDataset
from data.umi.codecs.imagecodecs_numcodecs import register_codecs
from data.umi.common.pose_repr_util import convert_pose_mat_rep
from data.umi.common.pytorch_util import dict_apply
from data.umi.common.replay_buffer import ReplayBuffer
from data.umi.common.sampler import SequenceSampler, get_val_mask
from data.umi.common.cv2_util import get_image_transform
from data.umi.pose_util import mat_to_pose10d, pose_to_mat
from models.normalizer import LinearNormalizer


register_codecs()


def suppress_stdout(func):
    """Decorator to suppress stdout during the function call."""
    def wrapper(*args, **kwargs):
        # Save the current stdout
        original_stdout = sys.stdout
        # Redirect stdout to os.devnull to suppress output
        sys.stdout = open(os.devnull, 'w')
        try:
            return func(*args, **kwargs)
        finally:
            # Restore the original stdout
            sys.stdout = original_stdout
    return wrapper


@suppress_stdout
def get_frame_from_video(video_path, frame_index):
    # Load the video
    with VideoFileClip(video_path) as video_clip:
    
        # Get the video's FPS (frames per second)
        fps = video_clip.fps
        
        # Calculate the time (in seconds) corresponding to the frame index
        time_in_seconds = frame_index / fps
        
        # Extract the frame at the given time
        frame = video_clip.get_frame(time_in_seconds)
    
    # Return the frame
    return frame


def get_frame_from_folder(video_path, frame_index, exts=["jpg", "png"]):
    # remove ext to get folder path
    folder_path = os.path.splitext(video_path)[0]
    for ext in exts:
        img_path = os.path.join(folder_path, f"{frame_index}.{ext}")
        if os.path.exists(img_path):
            break
    
    try:
        with Image.open(img_path) as img:
            return np.array(img)
    except Exception as e:
        # TODO(lingxuan): ensure all the images are valid for sft data
        print(f"Error loading image from {img_path}: {e}")
        return np.zeros((480, 480, 3), dtype=np.uint8)

class UmiVideoDataset(BaseDataset):

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        video_annotation_path: str,
        normalizer_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        config_path = "configs/bimanual_video_data.yaml",
        val_ratio: Optional[float] = None,
        use_image: bool = False,
        use_instruction: bool = False,
        instruction_path: Optional[str] = None,
        instruction_type: Literal["Mix", "MajorTask", "SubTask"] = "SubTask",
    ):
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        self.config_path = config_path
        config = OmegaConf.load(self.config_path)
        self.config = config
        self.dataset_name = dataset_name
        
        with open(video_annotation_path, "r") as fp:
            self.video_paths = json.load(fp)
        
        self.pose_repr = config.task.pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'relative')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'relative')
        self.use_image = use_image
        self.use_instruction = use_instruction
        self.instruction_type = instruction_type

        if use_instruction:
            with open(instruction_path, "r") as fp:
                # TODO(lingxuan): if the json file is too large,
                # we should cache it to disk
                self.instructions = json.load(fp)

        if cache_dir is None:
            # load into memory store
            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store,
                    store=zarr.MemoryStore()
                )
        else:
            # determine path name
            mod_time = os.path.getmtime(dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(dataset_path).split('.')[0]
            cache_name = '_'.join([stem_name, stamp])
            cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
            lock_path = cache_dir.joinpath(cache_name + '.lock')

            # load cached file
            print('Acquiring lock on cache.')
            with FileLock(lock_path):
                # cache does not exist
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(str(cache_path),
                            writemap=True, metasync=False, sync=False, map_async=True, lock=False
                        ) as lmdb_store:
                            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                                print(f"Copying data to {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=zip_store,
                                    store=lmdb_store
                                )
                        print("Cache written to disk!")
                    except Exception as e:
                        shutil.rmtree(cache_path)
                        raise e

            # open read-only lmdb store
            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(
                group=zarr.group(store)
            )
        
        self.num_robot = 0

        shape_meta = config.task.shape_meta

        rgb_keys = list()
        point_keys = list()
        lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            # solve obs type
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'point':
                point_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
            assert shape_meta['obs'][key]['horizon'] == 1, \
                "observation should have horizon = 1"
        
            if key.endswith('eef_pos'):
                self.num_robot += 1

            # solve obs_horizon
            horizon = shape_meta['obs'][key]['horizon']
            key_horizon[key] = horizon

            # solve latency_steps
            latency_steps = shape_meta['obs'][key]['latency_steps']
            key_latency_steps[key] = latency_steps

            # solve down_sample_steps
            down_sample_steps = shape_meta['obs'][key]['down_sample_steps']
            key_down_sample_steps[key] = down_sample_steps

        # solve action
        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        self.num_camera = len(rgb_keys)
        
        if val_ratio is None:
            val_ratio = self.config.task.dataset.val_ratio
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=config.task.dataset.seed
        )
        train_mask = ~val_mask

        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if not 'wrt' in key:
                self.sampler_lowdim_keys.append(key)

        for key in replay_buffer.keys():
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                self.sampler_lowdim_keys.append(key)
                query_key = key.split('_')[0] + '_eef_pos'
                key_horizon[key] = shape_meta['obs'][query_key]['horizon']
                key_latency_steps[key] = shape_meta['obs'][query_key]['latency_steps']
                key_down_sample_steps[key] = shape_meta['obs'][query_key]['down_sample_steps']

        self.resize_tf_dict = {}
        # define resize_tf
        for key in rgb_keys:
            assert shape_meta['obs'][key]['horizon'] == 1, \
                "RGB observation should have horizon = 1"
            
            raw_rgb_shape = shape_meta['obs'][key]['raw_shape']
            rgb_shape = shape_meta['obs'][key]['shape']
            
            if shape_meta['obs'][key]['camera_type'] == "binocular":
                in_res = (raw_rgb_shape[2], raw_rgb_shape[1])
                out_res = (rgb_shape[2], rgb_shape[1])
                # get resize transformation for each sub-image
                in_res_per_img = (in_res[0] // 2, in_res[1])
                out_res_per_img = (out_res[0] // 2, out_res[1])
                resize_tf = get_image_transform(
                    input_res=in_res_per_img,
                    output_res=out_res_per_img,
                )
                
                def binocular_resize_tf(img):
                    # Split binocular image into 2 images [H, 2 * W] -> 2 * [H, W]
                    sub_imgs = np.split(img, 2, axis=1)
                    # Resize separately
                    sub_imgs = [resize_tf(sub_img) for sub_img in sub_imgs]
                    # Concatenate
                    img = np.concatenate(sub_imgs, axis=1)
                    return img
                
                self.resize_tf_dict[key] = binocular_resize_tf
                
            elif shape_meta['obs'][key]['camera_type'] == "monocular":
                in_res = (raw_rgb_shape[2], raw_rgb_shape[1])
                out_res = (rgb_shape[2], rgb_shape[1])
                self.resize_tf_dict[key] = get_image_transform(
                    input_res=in_res,
                    output_res=out_res,
                )
            else:
                raise ValueError(f"Unknown camera type: {shape_meta['obs'][key]['camera_type']}")
        
        assert len(point_keys) <= 1, \
            "Currently only support less than 1 point cloud input"

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.point_keys = point_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = config.task.dataset.action_padding
        self.resample_with_timestamp = config.task.dataset.get("resample_with_timestamp", False)
        # self.repeat_frame_prob = config.task.dataset.repeat_frame_prob
        self.sampler = SequenceSampler(
            shape_meta=shape_meta,
            video_paths=self.video_paths,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys + point_keys, # use RGB API to sample pointcloud for they share the same function
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=config.task.dataset.action_padding,
            resample_with_timestamp=self.resample_with_timestamp,
        )
        self.temporally_independent_normalization = config.task.dataset.temporally_independent_normalization
        self.threadpool_limits_is_applied = False

        if normalizer_path is None:
            normalizer_path = os.path.join(os.path.dirname(dataset_path), "normalizer.pt")

        if os.path.exists(normalizer_path):
            self.normalizer = self.load_normalizer(normalizer_path)
        else:
            self.normalizer = self.get_normalizer()
            self.normalizer.save(normalizer_path)
        
    def load_normalizer(self, normalizer_path: str) -> LinearNormalizer:
        return LinearNormalizer.load(normalizer_path)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            video_paths=self.video_paths,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys + self.point_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            resample_with_timestamp=self.resample_with_timestamp,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        return self.normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        data = self.sampler.sample_sequence(idx)
        video_id = data["video_id"]
        video_path_dict = self.video_paths[video_id]
        # assert len(video_path_dict) == 2, \
        #     "video_path_dict should have 2 entries: 'left' and 'right'"

        # currently, the videos for all cameras is aligned 
        # and the instruction is label on the videos
        # so the instructions and the rgb observation shares the same indices
        # following code block is erroneous if the assumption above is not hold
        # instr_frame_idx = data[self.rgb_keys[0]].item()
        
        obs_dict = dict()
        for key in self.rgb_keys:
            if not key in data:
                continue
            
            video_path = video_path_dict[key]
            
            # since RGB horizon = 1
            # get the indices and video_path
            frame_idx = data[key].item()
            
            if self.use_image:
                # load from folder
                frame_dir = os.path.join(os.path.dirname(video_path), key)
                img = get_frame_from_folder(frame_dir, frame_idx)
            else:
                img = get_frame_from_video(video_path, frame_idx)
            
            img = self.resize_tf_dict[key](img)
            
            # split binocular image into 2 images [H, 2 * W] -> 2 * [H, W]
            # sub_imgs = np.split(img, 2, axis=1)
            # # resize seperately
            # sub_imgs = [self.resize_tf(sub_img) for sub_img in sub_imgs]
            # # concatenate
            # img = np.concatenate(sub_imgs, axis=1)
            
            obs_dict[key] = img[None]   # T,H,W,C
            del data[key]
        
        for key in self.point_keys:
            point_path = video_path_dict[key]
            frame_idx = data[key].item()
            
            points = np.load(os.path.join(point_path, f"{frame_idx}.npy"))
            obs_dict[key] = points  # N, 3 or N, 6
            del data[key]
            
        for key in self.sampler_lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]

        actions = list()
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            action_mat = pose_to_mat(data['action'][...,7 * robot_id: 7 * robot_id + 6])

            # solve relative obs
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)
            action_pose_mat = convert_pose_mat_rep(
                action_mat,
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)

            obs_pose = mat_to_pose10d(obs_pose_mat)
            action_pose = mat_to_pose10d(action_pose_mat)

            action_gripper = data['action'][..., 7 * robot_id + 6: 7 * robot_id + 7]
            actions.append(np.concatenate([action_pose, action_gripper], axis=-1))

            # generate data
            obs_dict[f'robot{robot_id}_eef_pos'] = obs_pose[:, :3]
            obs_dict[f'robot{robot_id}_eef_rot_axis_angle'] = obs_pose[:, 3:]

        data['action'] = np.concatenate(actions, axis=-1)

        torch_data = {
            'meta': {
                'dataset_name': self.dataset_name,
                'num_camera': self.num_camera,
                'num_robot': self.num_robot,
            },
            'obs': dict_apply(obs_dict, lambda x: torch.from_numpy(x).contiguous()),
            'action': torch.from_numpy(data['action'].astype(np.float32)),
            'action_token': torch.from_numpy(data['action_token']).to(dtype=torch.long),
            'gripper_valid': data['gripper_valid']
        }
        
        if self.use_instruction and "sub_task_instruction_key" in video_path_dict:
            instruction_key = video_path_dict["sub_task_instruction_key"]
            instruction = self.instructions.get(instruction_key, "")
            
            torch_data['meta']['instruction_key'] = instruction_key
            torch_data['instruction'] = instruction
        
        return torch_data
