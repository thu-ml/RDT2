from typing import Union

import numpy as np
import torch
import torch.nn as nn

from vqvae.models.vq import VectorQuantizer

class ResidualVectorQuantizer(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        num_embeddings, embedding_dim,
        latent_dim: int = 1024,
        n_codebooks: int = 4,
        commitment_cost=0.25, codebook_cost=0, 
        codebook_restart_interval=64,
        local_rank=0
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    latent_dim=latent_dim,
                    commitment_cost=commitment_cost,
                    codebook_cost=codebook_cost,
                    codebook_restart_interval=codebook_restart_interval,
                    local_rank=local_rank
                )
                for _ in range(n_codebooks)
            ]
        )
    
    def get_codebook_use_ratio(self):
        """
        Returns the ratio of codebook entries used in each sub-quantizer
        """
        return [q.get_codebook_use_ratio() for q in self.quantizers]

    def forward(self, z):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x C]
        
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z_q" : Tensor[B x C]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        loss_dict = {
            'vq': 0,
            'encoder': 0,
            'codebook': 0,
        }

        codebook_indices = []
        latents = []

        for quantizer in self.quantizers:

            z_q_i, loss_dict_i, indices_i, z_e_i = quantizer(residual)

            z_q = z_q + z_q_i
            residual = residual - z_q_i
            
            # Aggregate losses
            loss_dict['vq'] += loss_dict_i['vq']
            loss_dict['encoder'] += loss_dict_i['encoder']
            loss_dict['codebook'] += loss_dict_i['codebook']

            codebook_indices.append(indices_i)
            latents.append(z_e_i)
        
        codebook_indices = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, loss_dict, codebook_indices, latents
    
    def encode(self, z):
        """
        Encode the input tensor using the VQVAE model.
        
        Parameters
        ----------
        z : Tensor[B x C]
            A batch of action chunks
        
        Returns
        -------
        Tensor[B x N]
            Quantized token ids of input
        """
        residual = z
        token_ids_all = []
        
        for quantizer in self.quantizers:
            token_ids = quantizer.encode(residual)
            residual = residual - quantizer.decode(token_ids)
            
            token_ids_all.append(token_ids)
        
        return torch.stack(token_ids_all, dim=-1)
    
    def decode(self, token_ids: torch.Tensor):
        """
        Decode the input tensor using the VQVAE model.
        
        Parameters
        ----------
        token_ids : Tensor[B x N]
            A batch of action token ids
        
        Returns
        -------
        Tensor[B x C]
            Reconstructed action tensor
        """
        z_q = 0.0
        
        for i, quantizer in enumerate(self.quantizers):
            z_q += quantizer.decode(token_ids[:, i])
        
        return z_q