from typing import Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from vqvae.models.vqvae import VQVAE


class MultiVQVAE(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                input_dim,
                embedding_dim,
                cnn_config, 
                num_embeddings,
                action_horizon,
                n_codebooks=[6, 2, 1],
                codebook_restart_interval=64,
                commitment_cost=0.25,
                codebook_cost=0.,
                local_rank=0):
        super(MultiVQVAE, self).__init__()
        
        self.pos_vqvae = VQVAE(
            input_dim=input_dim["pos"],
            embedding_dim=embedding_dim,
            cnn_config=cnn_config,
            num_embeddings=num_embeddings,
            action_horizon=action_horizon,
            n_codebooks=n_codebooks["pos"],  # 6 * 3 = 18tokens
            codebook_restart_interval=codebook_restart_interval,
            commitment_cost=commitment_cost,
            codebook_cost=codebook_cost,
            local_rank=local_rank
        )
        self.pos_id_len = n_codebooks["pos"] * 3
        
        self.rot_vqvae = VQVAE(
            input_dim=input_dim["rot"],
            embedding_dim=embedding_dim,
            cnn_config=cnn_config,
            num_embeddings=num_embeddings,
            action_horizon=action_horizon,
            n_codebooks=n_codebooks["rot"],  # 3 * 3 = 9tokens
            codebook_restart_interval=codebook_restart_interval,
            commitment_cost=commitment_cost,
            codebook_cost=codebook_cost,
            local_rank=local_rank
        )
        self.rot_id_len = n_codebooks["rot"] * 3
        
        self.grip_vqvae = VQVAE(
            input_dim=input_dim["grip"],
            embedding_dim=embedding_dim,
            cnn_config=cnn_config,
            num_embeddings=num_embeddings,
            action_horizon=action_horizon,
            n_codebooks=n_codebooks["grip"],  # 1 * 3 = 3tokens
            codebook_restart_interval=codebook_restart_interval,
            commitment_cost=commitment_cost,
            codebook_cost=codebook_cost,
            local_rank=local_rank
        )
        self.grip_id_len = n_codebooks["grip"] * 3
        
        self.action_dim = input_dim["pos"] + input_dim["rot"] + input_dim["grip"]
        self.action_horizon = action_horizon
        self.num_embeddings = num_embeddings    # attr for validness check on generated action_ids
    
    def load_checkpoint(self, pos_state_dict, rot_state_dict, grip_state_dict):
        self.pos_vqvae.load_state_dict(torch.load(pos_state_dict)["module"])
        self.rot_vqvae.load_state_dict(torch.load(rot_state_dict)["module"])
        self.grip_vqvae.load_state_dict(torch.load(grip_state_dict)["module"])
    
    def encode(self, x: Union[torch.Tensor, dict[str, torch.Tensor]]):
        """
        Encode the input tensor using the VQVAE model.
        
        Parameters
        ----------
        x : Tensor[B x T x D], a batch of action chunks,
            or a dict with keys `pos`, `rot`, `grip`
        
        Returns
        -------
        Tensor[B x L]
            Quantized token ids of input,
            with an order of `pos`-`rot`-`grip`
        """
        if isinstance(x, dict):
            x_pos = x["pos"]
            x_rot = x["rot"]
            x_grip = x["grip"]
        else:
            x_pos = select_act_dim(x, "pos")
            x_rot = select_act_dim(x, "rot")
            x_grip = select_act_dim(x, "grip")

        pos_ids = self.pos_vqvae.encode(x_pos)
        rot_ids = self.rot_vqvae.encode(x_rot)
        grip_ids = self.grip_vqvae.encode(x_grip)
        
        # Concatenate the token ids from each VQVAE model
        # and return the result
        return torch.cat([pos_ids, rot_ids, grip_ids], dim=-1)
    
    def decode(self, ids: torch.Tensor, return_dict=False):
        """
        Decode the input tensor using the VQVAE model.
        
        Parameters
        ----------
        ids : Tensor[B x L], a batch of action token ids
            with an order of `pos`-`rot`-`grip`
        return_dict : bool, optional
            If True, return a dict with keys `pos`, `rot`, `grip`
            Otherwise, return a Tensor[B x T x D]
            Default: False
        
        Returns
        -------
        Tensor[B x T x D] or dict
            Reconstructed action tensor,
            or a dict with keys `pos`, `rot`, `grip`
        """
        pos_ids = ids[..., :self.pos_id_len]
        rot_ids = ids[..., self.pos_id_len:self.pos_id_len + self.rot_id_len]
        grip_ids = ids[..., self.pos_id_len + self.rot_id_len:]
        
        x_pos_recon = self.pos_vqvae.decode(pos_ids)
        x_rot_recon = self.rot_vqvae.decode(rot_ids)
        x_grip_recon = self.grip_vqvae.decode(grip_ids)
        
        if return_dict:
            return {
                "pos": x_pos_recon,
                "rot": x_rot_recon,
                "grip": x_grip_recon
            }
        x = torch.zeros(
            ids.shape[0], self.action_horizon, self.action_dim,
            device=ids.device
        )
        x = apply_act_dim(x, x_pos_recon, "pos")
        x = apply_act_dim(x, x_rot_recon, "rot")
        x = apply_act_dim(x, x_grip_recon, "grip")
        return x
    
    def calculate_loss(self, x, x_recon):
        x_pos = select_act_dim(x, "pos")
        x_rot = select_act_dim(x, "rot")
        x_grip = select_act_dim(x, "grip")
        x_pos_recon = select_act_dim(x_recon, "pos")
        x_rot_recon = select_act_dim(x_recon, "rot")
        x_grip_recon = select_act_dim(x_recon, "grip")
        loss_pos = self.pos_vqvae.calculate_loss(x_pos, x_pos_recon, act_type="pos")
        loss_rot = self.rot_vqvae.calculate_loss(x_rot, x_rot_recon, act_type="rot")
        loss_grip = self.grip_vqvae.calculate_loss(x_grip, x_grip_recon, act_type="grip")
        
        return {
            "pos": loss_pos,
            "rot": loss_rot,
            "grip": loss_grip
        }


def select_act_dim(x, act_type):
    if act_type == "pos":
        return torch.cat([x[..., :3], x[..., 10:13]], dim=-1)
    elif act_type == "rot":
        return torch.cat([x[..., 3:9], x[..., 13:19]], dim=-1)
    elif act_type == "grip":
        return torch.cat([x[..., 9:10], x[..., 19:20]], dim=-1)
    else:
        raise ValueError(f"Unknown action type: {act_type}")


def apply_act_dim(x, y, act_type):
    if act_type == "pos":
        x[..., :3] = y[..., :3]
        x[..., 10:13] = y[..., 3:]
    elif act_type == "rot":
        x[..., 3:9] = y[..., :6]
        x[..., 13:19] = y[..., 6:]
    elif act_type == "grip":
        x[..., 9:10] = y[..., :1]
        x[..., 19:20] = y[..., 1:]
    else:
        raise ValueError(f"Unknown action type: {act_type}")
    return x
