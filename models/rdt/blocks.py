import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp

from models.rdt.norm import RMSNorm
from models.rdt.attention import Attention, CrossAttention


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.

    Source:
    https://github.com/facebookresearch/DiT/blob/main/models.py
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FeedForward(nn.Module):
    """
    A feed-forward network with SiLU activation.

    Reference:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RDTBlock(nn.Module):
    """
    A RDT block with cross-attention conditioning.

    Reference:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """
    def __init__(self, layer_idx: int, config: dict):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config["hidden_size"]
        self.norm_eps = config["norm_eps"]

        self.attn_norm = RMSNorm(
            self.hidden_size, eps=self.norm_eps)
        self.attn = Attention(config)

        self.cross_norm = RMSNorm(
            self.hidden_size, eps=self.norm_eps)
        self.cond_norm = RMSNorm(
            self.hidden_size, eps=self.norm_eps)
        self.cross_attn = CrossAttention(config)

        self.ffn_norm = RMSNorm(
            self.hidden_size, eps=self.norm_eps)
        self.ffn = FeedForward(
            dim=self.hidden_size,
            hidden_dim=4*self.hidden_size,
            multiple_of=config["multiple_of"],
            ffn_dim_multiplier=config["ffn_dim_multiplier"],
        )

        adaLN_from = self.hidden_size
        adaLN_from += self.hidden_size # proprio state

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaLN_from, 9*self.hidden_size, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        ck: Optional[torch.Tensor] = None,
        cv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            t: (B, D * 2)
        """
        assert t.shape[1] == 2 * self.hidden_size, \
            f"Expected t shape to be (B, {2 * self.hidden_size}), " \

        # Adaptive Layer Normalization
        shift_attn, scale_attn, gate_attn, \
            shift_cross, scale_cross, gate_cross, \
            shift_mlp, scale_mlp, gate_mlp \
            = self.adaLN_modulation(t).chunk(9, dim=1)
        # Self-attention
        h = x + gate_attn.unsqueeze(1) * self.attn(
            modulate(self.attn_norm(x), shift_attn, scale_attn))
        # Cross-attention
        if c is not None:
            h = h + gate_cross.unsqueeze(1) * self.cross_attn(
                modulate(self.cross_norm(h), shift_cross, scale_cross),
                c=self.cond_norm(c), mask=mask)
        else:
            h = h + gate_cross.unsqueeze(1) * self.cross_attn(
                modulate(self.cross_norm(h), shift_cross, scale_cross),
                ck=ck, cv=cv, mask=mask)
        # Feed-forward
        out = h + gate_mlp.unsqueeze(1) * self.ffn(
            modulate(self.ffn_norm(h), shift_mlp, scale_mlp))
        return out


class FinalLayer(nn.Module):
    """
    The final layer of RDT.
    """
    def __init__(self, output_size, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.norm_eps = config["norm_eps"]
        self.output_size = output_size

        self.ffn_norm = RMSNorm(
            self.hidden_size, eps=self.norm_eps)
        self.ffn = Mlp(
            in_features=self.hidden_size,
            hidden_features=self.hidden_size*4,
            out_features=self.output_size,
            act_layer=nn.SiLU, drop=0.0
        )

        adaLN_from = self.hidden_size
        adaLN_from += self.hidden_size # proprio state

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaLN_from, 2*self.hidden_size, bias=True)
        )

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor
        ):
        """
        Args:
            t: (B, D + hidden_size)
        """
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.ffn_norm(x), shift, scale)
        x = self.ffn(x)
        return x
