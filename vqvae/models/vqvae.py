import torch
import torch.nn as nn
import torch.nn.functional as F

from vqvae.models.rvq import ResidualVectorQuantizer
from vqvae.models.cnn.model import Encoder, Decoder


class VQVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder (VQ-VAE)
    
    Complete VQ-VAE model, containing Encoder, VectorQuantizer and Decoder
    """
    def __init__(self, 
                 input_dim,
                 embedding_dim,
                 cnn_config, 
                 num_embeddings,
                 action_horizon,
                 n_codebooks=4,
                 codebook_restart_interval=64,
                 commitment_cost=0.25,
                 codebook_cost=0.,
                 local_rank=0):
        """
        Args:
            input_dim: Input dimension (D)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension (C)
            num_embeddings: Codebook size (Z)
            encoder_layers: Number of encoder residual blocks
            decoder_layers: Number of decoder residual blocks
            dropout: Dropout rate
            commitment_cost: VQ commitment loss weight
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.action_horizon = action_horizon
        self.n_codebooks = n_codebooks
        self.latent_dim = cnn_config['output_size']
        
        # Encoder
        self.encoder = Encoder(
            ch=cnn_config['hidden_size'],
            in_channels=input_dim,
            z_channels=cnn_config['output_size'],
            act_horizon=action_horizon,
            dropout=cnn_config['dropout']
        )
        
        # Vector quantization layer
        self.vq = ResidualVectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            latent_dim=self.latent_dim,
            n_codebooks=n_codebooks,
            commitment_cost=commitment_cost,
            codebook_cost=codebook_cost,
            codebook_restart_interval=codebook_restart_interval,
            local_rank=local_rank
        )
        
        # Decoder
        self.decoder = Decoder(
            ch=cnn_config['hidden_size'],
            in_channels=input_dim,
            z_channels=cnn_config['output_size'],
            act_horizon=action_horizon,
            dropout=cnn_config['dropout']
        )
        
        # Print Encoder/Decoder #Params
        # encoder_params = sum(p.numel() for p in self.encoder.parameters())
        # decoder_params = sum(p.numel() for p in self.decoder.parameters())
        # vq_params = self.vq.num_embeddings * self.vq.embedding_dim  # Codebook size

        # total_params = encoder_params + decoder_params + vq_params
        # print(f"Encoder Parameters: {encoder_params:,}")
        # print(f"Decoder Parameters: {decoder_params:,}")
        # print(f"VQ Codebook Parameters: {vq_params:,}")
        # print(f"Total Parameters: {total_params:,}")
        
    def encode(self, x):
        """
        Encoding process
        
        Args:
            x: Input vector [B, T, D]
        Returns:
            ids: token ids [B, T/8*N]
        """
        B = x.shape[0]
        # Encoding
        z = self.encoder(x.permute(0, 2, 1))
            # (B, T, D) -> (B, D, T) -> (B, C, T/8)
        
        z = z.permute(0, 2, 1)\
            .reshape(-1, self.latent_dim)
            # (B, C, T/8) -> (B, T/8, C) -> (B*T/8, C)
        
        # Quantization
        token_ids = self.vq.encode(z)
        
        return token_ids.reshape(B, -1) # (B, T/8*N)
    
    def decode(self, ids):
        """
        Decoding process
        
        Args:
            ids: token ids [B, T/8*N]
        Returns:
            x_recon: Reconstructed output [B, T, D]
        """
        B = ids.shape[0]
        N = self.n_codebooks
        
        # Decode
        ids = ids.reshape(B, -1, N).reshape(-1, N)
            # (B, T/8*N) -> (B, T/8, N) -> (B*T/8, N)
        z = self.vq.decode(ids)
        
        z = z.reshape(B, -1, self.latent_dim)\
            .permute(0, 2, 1)
            # (B*T/8, C) -> (B, T/8, C) -> (B, C, T/8)
        
        x_recon = self.decoder(z).permute(0, 2, 1)
            # (B, C, T/8) -> (B, D, T) -> (B, T, D)
        
        return x_recon
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input vector [B, T, D]
        Returns:
            x_recon: Reconstructed output [B, T, D]
            loss_dict: Dictionary of loss components
            encoding_indices: Indices of quantized vectors
        """
        # Encoding
        B, T, D = x.shape
        
        p = self.encoder(x.permute(0, 2, 1))
            # (B, T, D) -> (B, D, T) -> (B, C, T/8)
        
        p = p.permute(0, 2, 1)\
            .reshape(-1, self.latent_dim)
            # (B, C, T/8) -> (B, T/8, C) -> (B*T/8, C)
        
        # Quantization
        q, loss_dict, encoding_indices, _ = self.vq(p)
        
        # Decoding
        q = q.reshape(B, -1, self.latent_dim)\
            .permute(0, 2, 1)
            # (B*T/8, C) -> (B, T/8, C) -> (B, C, T/8)
        
        x_recon = self.decoder(q).permute(0, 2, 1)
            # (B, C, T/8) -> (B, D, T) -> (B, T, D)
        
        return x_recon, loss_dict, encoding_indices

    @staticmethod
    def geodesic_loss(R_pred, R_target, reduce=True, eps=1e-7, normalize=True):
        """
        Compute geodesic loss (rotation error) between predicted and target rotation matrices.

        Args:
            R_pred: (..., 3, 3) batch of predicted rotation matrices
            R_target: (..., 3, 3) batch of ground truth rotation matrices

        Returns:
            loss: (...,) batch of geodesic loss values
        """
        # Compute relative rotation R_err = R_pred^T @ R_target
        R_err = torch.matmul(
            torch.transpose(R_pred, -2, -1), R_target)

        # Compute trace of R_err
        trace = torch.diagonal(R_err, dim1=-2, dim2=-1).sum(-1)  # (B,)

        # Compute geodesic distance using arccos((trace - 1) / 2)
        # Clamp to avoid numerical issues: valid range for acos is [-1, 1]
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1.0+eps, 1.0-eps)

        loss = torch.acos(cos_theta)  # (B,)
        
        # Normalize
        if normalize:
            loss = loss / torch.pi  # Normalize to [0, 1]

        if reduce:
            loss = loss.mean()
        return loss

    @staticmethod
    def rot6d_to_mat(rot6d):
        """
        Convert 6D rotation representation to rotation matrix.

        Args:
            rot6d: (..., 6) batch of 6D rotation vectors

        Returns:
            R: (..., 3, 3) batch of rotation matrices
        """
        a1, a2 = rot6d[..., :3], rot6d[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        R = torch.stack((b1, b2, b3), dim=-1)
        return R
    
    def calculate_loss(self, x, x_recon, loss_dict=None, act_type='pos'):
        """
        Calculate total loss
        
        Args:
            x: Original input [B, D]
            x_recon: Reconstructed output [B, D]
            vq_loss: VQ loss
            p: Encoder output [B, C]
            q: Quantized vector [B, C]
        Returns:
            total_loss: Total loss
            loss_dict: Dictionary of individual loss components
        """
        # Reconstruction loss (MSE)
        if act_type == 'rot':
            # Convert x and x_recon to rotation matrices
            left_x = self.rot6d_to_mat(x[..., :6])
            right_x = self.rot6d_to_mat(x[..., 6:])
            left_x_recon = self.rot6d_to_mat(x_recon[..., :6])
            right_x_recon = self.rot6d_to_mat(x_recon[..., 6:])
            
            # Compute geodesic loss
            recon_loss = (self.geodesic_loss(left_x, left_x_recon) + \
                self.geodesic_loss(right_x, right_x_recon)) / 2
        else:
            recon_loss = F.mse_loss(x_recon, x)
        
        if loss_dict is None:
            return recon_loss
        
        # Total loss
        total_loss = recon_loss + loss_dict['vq']
        
        # Return total loss and individual loss components
        loss_dict = {
            'total': total_loss,
            'recon': recon_loss,
            **loss_dict
        }
        
        return total_loss, loss_dict

    def get_codebook_use_ratio(self):
        return self.vq.get_codebook_use_ratio()