from collections import OrderedDict
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from models.rdt.blocks import FinalLayer, RDTBlock, TimestepEmbedder
from models.rdt.pos_emb import get_multimodal_pos_embed


class RDT(nn.Module):
    """
    Class for Robotics Diffusion Transformers.
    """
    def __init__(
        self,
        horizon: int,
        output_size: int,
        config: dict,
        x_pos_emb_config: List[Tuple],
        lang_pos_emb_config: List[Tuple],
        max_lang_len: int,
        img_pos_emb_config: List[Tuple],
        max_img_len: int,
        dtype=torch.bfloat16
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = config["hidden_size"]
        self.n_heads = config["num_heads"]
        self.dtype = dtype

        self.t_embedder = TimestepEmbedder(self.hidden_size, dtype=dtype)

        # Create each RDT layer
        self.depth = config["depth"]
        self.blocks = nn.ModuleList([
            RDTBlock(layer_idx, config=config)
            for layer_idx in range(self.depth)
        ])
        self.final_layer = FinalLayer(output_size, config=config)

        # Append learnable tokens to the input action sequence
        self.num_register_tokens = config.get(
            "num_register_tokens", 4
        )
        self.register_tokens = nn.Parameter(
            torch.randn(1, self.num_register_tokens, self.hidden_size)
        )

        # Required: positional embeddings for action
        self.x_pos_emb_config = x_pos_emb_config
        
        # Optional: positional embeddings for language
        # self.lang_pos_emb_config = lang_pos_emb_config
        # self.lang_pos_emb = nn.Parameter(torch.zeros(
        #     1, max_lang_len, self.hidden_size)) \
        #         if lang_pos_emb_config is not None else None
        
        # Optional: positional embeddings for image
        self.img_pos_emb_config = img_pos_emb_config
        self.img_pos_emb = nn.Parameter(torch.zeros(
            1, max_img_len, self.hidden_size)) \
                if img_pos_emb_config is not None else None
        
        # Required: positional embeddings for state
        self.state_pos_emb_config = [
            ("state", 1)
        ]
        self.x_pos_emb = nn.Parameter(torch.zeros(
            1, self.horizon + self.num_register_tokens, self.hidden_size))
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_emb by sin-cos embeddings
        x_pos_emb = get_multimodal_pos_embed(
            embed_dim=self.hidden_size,
            mm_lens=OrderedDict(self.x_pos_emb_config)
        )
        self.x_pos_emb.data.copy_(
            torch.from_numpy(x_pos_emb).float().unsqueeze(0))

        # if self.lang_pos_embed is not None:
        #     lang_pos_emb = get_multimodal_pos_embed(
        #         embed_dim=self.hidden_size,
        #         mm_lens=OrderedDict(self.lang_pos_emb_config)
        #     )
        #     self.lang_pos_emb.data.copy_(
        #         torch.from_numpy(lang_pos_emb).float().unsqueeze(0))

        if self.img_pos_emb is not None:
            img_pos_embed = get_multimodal_pos_embed(
                embed_dim=self.hidden_size,
                mm_lens=OrderedDict(self.img_pos_emb_config)
            ) # (L_img, D)
            self.img_pos_emb.data.copy_(
                torch.from_numpy(img_pos_embed).float().unsqueeze(0))

        state_pos_emb = get_multimodal_pos_embed(
            embed_dim=self.hidden_size,
            mm_lens=OrderedDict(self.state_pos_emb_config)
        )
        self.state_pos_emb = nn.Parameter(
            torch.from_numpy(state_pos_emb).float().unsqueeze(0)
        ) # (1, D)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in RDT blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.ffn.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn.fc2.bias, 0)

        # Move all the params to given data type
        self.to(self.dtype)

    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        lang_c: Optional[torch.Tensor] = None, 
        lang_c_kv: Optional[List[Tuple[torch.Tensor]]] = None, 
        img_c: Optional[torch.Tensor] = None, 
        state_c: Optional[torch.Tensor] = None, 
        lang_mask: Optional[torch.Tensor] = None, 
        img_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass of RDT.

        Args:
            x: (B, T + 1, D)
                T means action_chunk_size
            t: (B,) or (1,), diffusion timesteps.
            lang_c: (B, depth, L_lang, D) or None, language condition tokens (variable length),
                dimension D is assumed to be the same as the hidden size.
            lang_c_kv: (B, depth, L_lang, 2, D) or None, language condition key and value tokens (variable length),
                dimension D is assumed to be the same as the hidden size.
            img_c: (B, L_img, D) or None, image condition tokens (fixed length),
                dimension D is assumed to be the same as the hidden size.
            state_c: (B, 1, D) or None, state condition tokens (fixed length),
            lang_mask: (B, L_lang) or None, language condition mask (True for valid).
            img_mask: (B, L_img) or None, image condition mask (True for valid).
        """
        t = self.t_embedder(t) # (B, D) or (1, D)
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1)    # (B, D)

        assert state_c is not None, "State condition must be provided."
        state_c = state_c + self.state_pos_emb # (B, 1, D)

        t = torch.cat([t.unsqueeze(1), state_c], dim=1).reshape(x.shape[0], self.hidden_size * 2) # (B, D * 2)

        r = self.register_tokens.expand(x.shape[0], -1, -1)
        # Pack [x, r]
        x = torch.cat([x, r], dim=1)

        # Add positional embeddings
        x = x + self.x_pos_emb
        # Note that the lang is of variable length
        # Since postion id is added by the Qwen2.5-VL-7B-Instruct
        # we don't need to add it here
        # lang_c = lang_c + self.lang_pos_emb[:, :lang_c.shape[1]]
        if img_c is not None and self.img_pos_emb is not None:
            img_c = img_c + self.img_pos_emb
        
        conds = [lang_c_kv or lang_c]
        masks = [lang_mask]
        if self.img_pos_emb is not None:
            conds.append(img_c)
            masks.append(img_mask)
        
        num_conds = len(conds)
        for i, block in enumerate(self.blocks):
            c, mask = conds[i % num_conds], masks[i % num_conds]
            ck, cv = None, None
            if isinstance(c, List):
                ck, cv = c[i % len(c)]
                c = None
                ck = ck.transpose(1, 2) # (bs, n_kv_heads, seq_len, head_size)
                cv = cv.transpose(1, 2) # (bs, n_kv_heads, seq_len, head_size)
            elif c.dim() == 4:
                # its per layer language condition
                c = c[:, i]
            x = block(x, t, c, ck, cv, mask=mask)
        x = self.final_layer(x, t)

        # Unpack [x, r]
        x = x[:, :-self.num_register_tokens]

        return x
