"""
Source: https://github.com/real-stanford/universal_manipulation_interface
"""
from typing import Union, Dict

import numpy as np
import torch
import torch.nn as nn

from data.umi.common.pytorch_util import dict_apply
from models.normalizer.dict_of_tensor_mixin import DictOfTensorMixin


class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']

    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray], mask_dict=None) -> torch.Tensor:
        return self.normalize(x, mask_dict)

    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str , value: 'SingleFieldLinearNormalizer'):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, mask_dict=None, forward=True):
        if mask_dict is None:
            mask_dict = dict()
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                params = self.params_dict[key]
                normal_res = _normalize(value, params, forward=forward)
                if key in mask_dict:
                    mask = mask_dict[key]
                    normal_res[..., mask] = value[..., mask]
                result[key] = normal_res
            return result
        else:
            if '_default' not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict['_default']
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray], mask_dict=None) -> torch.Tensor:
        return self._normalize_impl(x, mask_dict, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray], mask_dict=None) -> torch.Tensor:
        return self._normalize_impl(x, mask_dict, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and '_default' in self.params_dict:
            return self.params_dict['_default']['input_stats']

        result = dict()
        for key, value in self.params_dict.items():
            if key != '_default':
                result[key] = value['input_stats']
        return result


    def get_output_stats(self, key='_default'):
        input_stats = self.get_input_stats()
        if 'min' in input_stats:
            # no dict
            return dict_apply(input_stats, self.normalize)

        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key:value})[key]
            result[key] = this_dict
        return result

    def save(self, filepath: str):
        torch.save(self.params_dict, filepath)
        print(f"LinearNormalizer saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'LinearNormalizer':
        normalizer = cls()
        normalizer.params_dict = torch.load(filepath, weights_only=False)
        print(f"LinearNormalizer loaded from {filepath}")
        return normalizer


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']

    @classmethod
    def create_manual(cls,
            scale: Union[torch.Tensor, np.ndarray],
            offset: Union[torch.Tensor, np.ndarray],
            input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]]):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            # HACK: we do not flatten here, to ensure the concat is right
            # x = x.flatten()
            return x

        # check
        for x_name, x in [("offset", offset)] + list(zip(list(input_stats_dict.keys()), list(input_stats_dict.values()))):
            # print(f"x_name: {x_name}, x: {x.dtype}, scale: {scale.dtype}")
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype

        params_dict = nn.ParameterDict({
            'scale': to_tensor(scale),
            'offset': to_tensor(offset),
            'input_stats': nn.ParameterDict(
                dict_apply(input_stats_dict, to_tensor))
        })
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            'min': torch.tensor([-1], dtype=dtype),
            'max': torch.tensor([1], dtype=dtype),
            'mean': torch.tensor([0], dtype=dtype),
            'std': torch.tensor([1], dtype=dtype)
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict['input_stats']

    def get_output_stats(self):
        return dict_apply(self.params_dict['input_stats'], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)


def _normalize(x, params, forward=True):
    assert 'scale' in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params['scale']
    offset = params['offset']
    x = x.to(device=scale.device, dtype=scale.dtype)
    # src_shape = x.shape
    # x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    # x = x.reshape(src_shape)
    return x