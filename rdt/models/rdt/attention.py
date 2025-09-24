import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from rdt.models.rdt.norm import RMSNorm


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    A self-attention layer with flash attention.
    
    Paper:
    https://arxiv.org/abs/1706.03762
    
    Reference:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """
    def __init__(self, config: dict):
        super().__init__()
        self.n_heads = config["num_heads"]
        self.n_kv_heads = self.n_heads if config["num_kv_heads"] is None else config["num_kv_heads"]
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError('num_heads should be divisible by num_kv_heads')
        self.n_rep = self.n_heads // self.n_kv_heads
        self.hidden_size = config["hidden_size"]
        if self.hidden_size % self.n_heads != 0:
            raise ValueError('hidden_size should be divisible by num_heads')
        self.head_size = self.hidden_size // self.n_heads

        self.wq = nn.Linear(
            self.hidden_size, 
            self.n_heads * self.head_size, 
            bias=False
        )
        self.wkv = nn.Linear(
            self.hidden_size, 
            self.n_kv_heads * self.head_size * 2, 
            bias=False
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_size, 
            self.hidden_size, 
            bias=False
        )

        self.norm_eps = config["norm_eps"]
        self.norm_q = RMSNorm(self.head_size, eps=self.norm_eps)
        self.norm_k = RMSNorm(self.head_size, eps=self.norm_eps)

        self.use_flash_attn = config["use_flash_attn"]

        self.attn_scale = 1.0 / math.sqrt(self.head_size)

    def forward(
        self,
        x: torch.Tensor
    ):
        bs, seq_len, _ = x.shape   # (bs, seq_len, hidden_size), batch size, sequence length, hidden size

        xq = self.wq(x)
        xq = xq.view(bs, seq_len, self.n_heads, self.head_size)

        xkv = self.wkv(x)
        xkv = xkv.view(bs, seq_len, self.n_kv_heads, self.head_size, 2)
        xk, xv = xkv.unbind(-1)

        xq, xk = self.norm_q(xq), self.norm_k(xk)

        # Repeat k/v heads if n_kv_heads < n_heads
        xk = repeat_kv(
            xk, self.n_rep
        )  # (bs, seq_len, n_heads, head_size)
        xv = repeat_kv(
            xv, self.n_rep
        )  # (bs, seq_len, n_heads, head_size)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seq_len, head_size)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.use_flash_attn:
            output = F.scaled_dot_product_attention(
                query=xq,
                key=xk,
                value=xv,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=self.attn_scale,
            )
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) * self.attn_scale
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)   # (bs, n_heads, seq_len, head_size)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(output)


class CrossAttention(nn.Module):
    """
    A cross-attention layer with flash attention.
    
    Paper:
    https://arxiv.org/abs/1706.03762
    
    Reference:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """
    def __init__(self, config: dict):
        super().__init__()
        self.n_heads = config["num_heads"]
        self.n_kv_heads = self.n_heads if config["num_kv_heads"] is None else config["num_kv_heads"]
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError('num_heads should be divisible by num_kv_heads')
        self.n_rep = self.n_heads // self.n_kv_heads
        self.hidden_size = config["hidden_size"]
        if self.hidden_size % self.n_heads != 0:
            raise ValueError('hidden_size should be divisible by num_heads')
        self.head_size = self.hidden_size // self.n_heads

        self.wq = nn.Linear(
            self.hidden_size, 
            self.n_heads * self.head_size, 
            bias=False
        )
        self.wkv = nn.Linear(
            self.hidden_size, 
            self.n_kv_heads * self.head_size * 2, 
            bias=False
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_size, 
            self.hidden_size, 
            bias=False
        )

        self.norm_eps = config["norm_eps"]
        self.norm_q = RMSNorm(self.head_size, eps=self.norm_eps)
        self.norm_k = RMSNorm(self.head_size, eps=self.norm_eps)

        self.use_flash_attn = config["use_flash_attn"]

        self.attn_scale = 1.0 / math.sqrt(self.head_size)

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        ck: Optional[torch.Tensor] = None,
        cv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        bs, seq_len, _ = x.shape   # (bs, seq_len, hidden_size), batch size, sequence length, hidden size
        
        xq = self.wq(x)
        xq = xq.view(bs, seq_len, self.n_heads, self.head_size)
        xq = self.norm_q(xq)

        if c is not None:
            _, c_len, _ = c.shape     # (bs, c_len, hidden_size), batch size, condition length, hidden size

            ckv = self.wkv(c)
            ckv = ckv.view(bs, c_len, self.n_kv_heads, self.head_size, 2)
            ck, cv = ckv.unbind(-1)

            ck = self.norm_k(ck)

        # Repeat k/v heads if n_kv_heads < n_heads
        ck = repeat_kv(
            ck, self.n_rep
        )  # (bs, c_len, n_heads, head_size)
        cv = repeat_kv(
            cv, self.n_rep
        )  # (bs, c_len, n_heads, head_size)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seq_len, head_size)
        ck = ck.transpose(1, 2)  # (bs, n_heads, c_len, head_size)
        cv = cv.transpose(1, 2)  # (bs, n_heads, c_len, head_size)

        # Prepare attn mask (bs, c_len) to mask the condition
        if mask is not None:
            mask = mask.reshape(bs, 1, 1, -1)
            mask = mask.expand(-1, -1, seq_len, -1)

        if self.use_flash_attn:
            output = F.scaled_dot_product_attention(
                query=xq,
                key=ck,
                value=cv,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.attn_scale,
            )
        else:
            scores = torch.matmul(xq, ck.transpose(2, 3)) * self.attn_scale
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, cv)   # (bs, n_heads, seq_len, head_size)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(output)
