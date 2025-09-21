from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class VectorQuantizer(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
        3. EMA updates: Use exponential moving average to update codebook
        4. CodeBook restart: Restart codebook every N iterations to avoid dead entries
    """
    def __init__(self, num_embeddings, embedding_dim, latent_dim,
                 commitment_cost=0.25, codebook_cost=0,
                 codebook_restart_interval=64,
                 ema_decay=0.99, local_rank=0):
        """
        Args:
            num_embeddings: Size of the codebook (Z)
            embedding_dim: Dimension of the embeddings (C)
            commitment_cost: Weight for commitment loss
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost
        self.codebook_cost = codebook_cost
        self.local_rank = local_rank
        
        # Initialize embeddings
        self.codebook = nn.Embedding(num_embeddings, embedding_dim, _freeze=True)
        self.weight = self.codebook.weight
        
        # Factorized codes trick
        self.in_proj = nn.Linear(latent_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, latent_dim)
        
        # EMA variables
        self.codebook_init = False
        self.ema_decay = ema_decay
        self.register_buffer('ema_w', self.weight.clone())
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        
        # Track dead entries and restart
        self.restart_period = codebook_restart_interval
        self.restart_cnt = 0
        # TODO: hold these variables in CPU memory if overflows
        self.register_buffer('entry_hits',
            torch.zeros(self.restart_period, num_embeddings, dtype=torch.int32))
        # self.encoder_buf = None
    
    def get_codebook_use_ratio(self):
        """
        Returns the ratio of codebook entries used
        """
        ratio = self.entry_hits.sum(0) > 0
        return ratio.sum().item() / self.num_embeddings
    
    def update_entry_hits(self, cluster_size):
        with torch.no_grad():
            self.entry_hits[self.restart_cnt] = cluster_size
            self.restart_cnt += 1
    
    def init_codebook(self, z_e, indices_init=None):
        """
        Initialize codebook with encoder outputs
        
        Args:
            z_e: Encoder outputs of shape [B, C]
            indices_init: Indices to initialize codebook,
                default is all
        """
        with torch.no_grad():
            if indices_init is None:
                indices_init = torch.arange(self.num_embeddings)
            init_size = indices_init.shape[0]
            
            # Gather z_e across all GPUs
            world_size = dist.get_world_size()
            all_z_e = [torch.zeros_like(z_e) for _ in range(world_size)]
            dist.all_gather(all_z_e, z_e)
            all_z_e = torch.cat(all_z_e, dim=0)
            
            # Use all z_e to initialize codebook
            # [B, Z, C]
            random_indices = torch.randperm(all_z_e.shape[0])
            
            if all_z_e.shape[0] < init_size:
                # Not enough samples to initialize CodeBook
                # We will randomly initialize some of them
                shuffle_indices = torch.randperm(init_size)
                indices_init = indices_init[shuffle_indices]
                init_size = all_z_e.shape[0]
                indices_init = indices_init[:init_size]
            
            random_indices = random_indices[:init_size]
            self.weight.index_copy_(0, 
                indices_init.to(device=self.weight.device), 
                all_z_e[random_indices])
            
            return indices_init
    
    def ema_update(self, z_e, encodings, actual_indices_init=None):
        """
        Update EMA variables and entry hits
        """
        with torch.no_grad():
            # Locally calculate some results
            local_new_cluster_size = torch.sum(encodings, 0)
            local_new_ema_w = torch.matmul(encodings.t(), z_e)
            
            # Gather results across all GPUs
            new_cluster_size = local_new_cluster_size.clone()
            new_ema_w = local_new_ema_w.clone()
            dist.all_reduce(new_cluster_size, op=dist.ReduceOp.SUM)
            dist.all_reduce(new_ema_w, op=dist.ReduceOp.SUM)
            
            def _mask_copy(variable, value, mask=None, mask_value=None):
                if mask is not None:
                    value[mask] = mask_value[mask]
                variable.copy_(value)
            
            # Update cluster size
            # self.ema_cluster_size.copy_(self.ema_decay * self.ema_cluster_size + \
            #                     (1 - self.ema_decay) * new_cluster_size)
            _mask_copy(
                variable=self.ema_cluster_size, 
                value=self.ema_decay * self.ema_cluster_size + 
                    (1 - self.ema_decay) * new_cluster_size,
                mask=actual_indices_init,
                mask_value=new_cluster_size
            )
            
            # Update cluster weights
            # self.ema_w.copy_(self.ema_decay * self.ema_w + (1 - self.ema_decay) * new_ema_w)
            _mask_copy(
                variable=self.ema_w,
                value=self.ema_decay * self.ema_w + 
                    (1 - self.ema_decay) * new_ema_w,
                mask=actual_indices_init,
                mask_value=new_ema_w
            )

            # Normalize cluster weights
            n = torch.sum(self.ema_cluster_size)
            smoothed_cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            # Normalize embedding average with smoothed cluster size
            # self.weight.copy_(self.ema_w / smoothed_cluster_size.unsqueeze(1))
            _mask_copy(
                variable=self.weight,
                value=self.ema_w / smoothed_cluster_size.unsqueeze(1),
                mask=actual_indices_init,
                mask_value=self.weight.clone()
            )
            
            # Update entry hits and encoder buffer
            self.update_entry_hits(new_cluster_size)
    
    def forward(self, z_e):
        """
        Args:
            z_e: Encoder outputs of shape [B, C]
        Returns:
            quantized: Quantized vectors
            loss: VQ loss
            indices: Indices of the selected embeddings
            encodings: One-hot encodings of the indices
        """
        # [B, C]
        z_e = self.in_proj(z_e) # [B, E]
        
        # Initialize CodeBook
        actual_indices_init = None
        if self.training:
            if not self.codebook_init:
                self.init_codebook(z_e)
                self.codebook_init = True
            elif self.restart_cnt >= self.restart_period:
                indices_init = self.entry_hits.sum(0) == 0
                indices_init = torch.where(indices_init)[0]
                actual_indices_init = self.init_codebook(z_e, indices_init)
                self.restart_cnt = 0
        
        # L2 normalize encodings and codebook (ViT-VQGAN)
        z_e_normalized = F.normalize(z_e)
        weight_normalized = F.normalize(self.weight)
        
        # Calculate euclidean distances
        # [B, Z]
        distances = torch.sum(z_e_normalized**2, 1, keepdim=True) + \
                    - 2 * torch.matmul(z_e_normalized, weight_normalized.t()) \
                    + torch.sum(weight_normalized**2, 1)
        
        # Find nearest embedding for each vector
        # [B,]
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Convert indices to one-hot encodings
        # [B, Z]
        encodings = F.one_hot(encoding_indices, self.num_embeddings).to(dtype=z_e.dtype)
        
        # Quantize
        # [B, C]
        # quantized = torch.matmul(encodings, self.weight)
        z_q = self.codebook(encoding_indices)
        
        # EMA update
        if self.training:
            self.ema_update(z_e, encodings, actual_indices_init)
        
        # Calculate loss
        # Commitment loss: encourage encoder to output vectors close to codebook
        # [B, C]
        encoder_loss = F.mse_loss(z_e, z_q.detach())
        
        # Codebook loss: update codebook towards encoder outputs
        # [B, C]
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        
        # Total VQ loss
        loss = codebook_loss * self.codebook_cost + encoder_loss * self.commitment_cost
        
        # Straight-through estimator
        # Pass gradients from quantized to z_e
        z_q = z_e + (z_q - z_e).detach()
        
        z_q = self.out_proj(z_q) # [B, C]
        
        loss_dict = {
            'vq': loss,
            'encoder': encoder_loss,
            'codebook': codebook_loss,
        }
        
        return z_q, loss_dict, encoding_indices, z_e
    
    def encode(self, z_e):
        """
        Encode the input tensor using the codebook
        Args:
            z_e: Encoder outputs of shape [B, C]
        Returns:
            token_ids: Token ids [B,]
        """
        # [B, C]
        z_e = self.in_proj(z_e) # [B, E]
        
        # L2 normalize encodings and codebook (ViT-VQGAN)
        z_e_normalized = F.normalize(z_e)
        weight_normalized = F.normalize(self.weight)
        
        # Calculate euclidean distances
        # [B, Z]
        distances = torch.sum(z_e_normalized**2, 1, keepdim=True) + \
                    - 2 * torch.matmul(z_e_normalized, weight_normalized.t()) \
                    + torch.sum(weight_normalized**2, 1)
        
        # Find nearest embedding for each vector
        # [B,]
        ids = torch.argmin(distances, dim=1)
        return ids
    
    def decode(self, token_ids):
        """
        Decode the input token ids using the codebook
        Args:
            token_ids: Token ids [B,]
        Returns:
            z_q: Quantized vectors [B, C]
        """
        # [B, C]
        z_q = self.codebook(token_ids)
        return self.out_proj(z_q)
