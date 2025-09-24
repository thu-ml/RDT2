import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LogisticNormal

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt.model import RDT


class RDTRunner(
        nn.Module,
        CompatiblePyTorchModelHubMixin,
        repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"
    ):
    def __init__(
        self,
        action_dim: int,
        pred_horizon: int,
        config: dict,
        act_pos_emb_config: list[tuple],
        lang_token_dim: Optional[int] = None,
        img_token_dim: Optional[int] = None,
        lang_pos_emb_config: Optional[list[tuple]] = None,
        max_lang_len: Optional[int] = None,
        img_pos_emb_config: Optional[list[tuple]] = None,
        max_img_len: Optional[int] = None,
        state_dim: Optional[int] = None,
        dtype=torch.bfloat16
    ):
        super(RDTRunner, self).__init__()
        # Create the diffusion model
        hidden_size = config['rdt']['hidden_size']
        self.model = RDT(
            horizon=pred_horizon,
            output_size=action_dim,
            config=config['rdt'],
            x_pos_emb_config=act_pos_emb_config,
            lang_pos_emb_config=lang_pos_emb_config,
            max_lang_len=max_lang_len,
            img_pos_emb_config=img_pos_emb_config,
            max_img_len=max_img_len,
            dtype=dtype,
        )

        # If you directly use the embeds from VLM
        # then you don't need to adapt the conditional inputs
        self.lang_adaptor = self.img_adaptor = None 
        if config.get('lang_adaptor', None) is not None:
            # Create adpators for various conditional inputs
            self.lang_adaptor = self.build_condition_adapter(
                config['lang_adaptor'],
                in_features=lang_token_dim,
                out_features=hidden_size
            )
        if config.get('img_adaptor', None) is not None:
            self.img_adaptor = self.build_condition_adapter(
                config['img_adaptor'],
                in_features=img_token_dim,
                out_features=hidden_size
            )

        # For RDT as action expert,
        # the action and state adaptor are stull required
        self.act_adaptor = self.build_condition_adapter(
            config['act_adaptor'],
            in_features=action_dim,
            out_features=hidden_size
        )

        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'],
            in_features=state_dim, # no mask actually
            out_features=hidden_size
        )

        # Create the noise scheduler
        noise_scheduler_config = config['noise_scheduler']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        # We use logit-normal as it works better in practice
        self.timestep_sampler = LogisticNormal(0, 1)

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        print(f"RDT params: {sum([p.numel() for p in self.model.parameters()])}")
        print("Diffusion params: %e" % sum(
            [p.numel() for p in self.model.parameters()] +
            [p.numel() for p in self.lang_adaptor.parameters()] \
                if self.lang_adaptor is not None else [] +
            [p.numel() for p in self.img_adaptor.parameters()] \
                if self.img_adaptor is not None else [] +
            [p.numel() for p in self.act_adaptor.parameters()] +
            [p.numel() for p in self.state_adaptor.parameters()]
        ))
    
    def sample_timesteps(self, batch_size, device):
        # We only use y0 since y0 + y1 = 1
        return self.timestep_sampler.sample((batch_size,))[:, 0].to(device)

    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_silu_match = re.match(r'^mlp(\d+)x_silu$', projector_type)
            if mlp_silu_match:
                mlp_depth = int(mlp_silu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.SiLU())
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector

    def adapt_conditions(self, lang_tokens, img_tokens, action_tokens, state_tokens):
        '''
        lang_tokens: (batch_size, depth, lang_len, lang_token_dim) or (batch_size, lang_len, lang_token_dim)
        img_tokens: (batch_size, img_len, img_token_dim)
        action_tokens: (batch_size, horizon, action_token_dim)

        return: adapted (..., hidden_size) for all input tokens
        '''
        adapted_lang = self.lang_adaptor(lang_tokens) \
            if lang_tokens is not None else None
        adapted_img = self.img_adaptor(img_tokens) \
            if img_tokens is not None else None
        
        adapted_action = self.act_adaptor(action_tokens) \
            if action_tokens is not None else None
        adapted_state = self.state_adaptor(state_tokens)

        return adapted_lang, adapted_img, adapted_action, adapted_state

    def _prepare_condition_inputs(
        self,
        lang_cond: Optional[torch.Tensor] = None,
        lang_kv_cache: Optional[torch.Tensor] = None,
        lang_attn_mask: Optional[torch.Tensor] = None,
        img_cond: Optional[torch.Tensor] = None,
        state_cond: Optional[torch.Tensor] = None,
    ):
        condition_inputs = {}
        
        # NOTE: ensure only one of lang_c or lang_c_kv is provided
        if lang_kv_cache is not None:
            condition_inputs["lang_c_kv"] = lang_kv_cache
        else:
            condition_inputs["lang_c"] = lang_cond
        
        condition_inputs["lang_mask"] = lang_attn_mask
        condition_inputs["img_c"] = img_cond
        condition_inputs["state_c"] = state_cond
        
        return condition_inputs

    def conditional_sample(
        self,
        state_cond: Optional[torch.Tensor],
        noisy_action: Optional[torch.Tensor] = None,
        lang_cond: Optional[torch.Tensor] = None,
        lang_kv_cache: Optional[torch.Tensor] = None,
        lang_attn_mask: Optional[torch.Tensor] = None,
        img_cond: Optional[torch.Tensor] = None,
    ):
        '''
        Args:
            lang_cond: language conditional data, (batch_size, lang_len, hidden_size).
            lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
                which should be True-False bool tensor.
            img_cond: image conditional data, (batch_size, img_len, hidden_size).
            state_cond: state conditional data, (batch_size, 1, hidden_size).
            noisy_action: (batch_size, horizon, action_dim), optional,
                if provided, will be used as the initial noisy action for sampling.
                If not provided, a random noise will be generated.

        return: (batch_size, horizon, action_dim)
        '''
        batch_size = state_cond.shape[0]
        device = state_cond.device
        dtype = state_cond.dtype
        if noisy_action is None:
            noisy_action = torch.randn(
                size=(batch_size, self.pred_horizon, self.action_dim),
                dtype=dtype, device=device)

        condition_inputs = self._prepare_condition_inputs(
            lang_cond=lang_cond,
            lang_kv_cache=lang_kv_cache,
            lang_attn_mask=lang_attn_mask,
            img_cond=img_cond,
            state_cond=state_cond,
        )

        # Solve the ODE
        timestep = torch.tensor([0.0], dtype=dtype, device=device)
        step_size = 1.0 / self.num_inference_timesteps
        for _ in range(self.num_inference_timesteps):
            # Prepare action trajectory
            action_traj = self.act_adaptor(noisy_action)

            # Predict the model output
            model_output = self.model(
                x=action_traj,
                t=timestep,
                **condition_inputs
            )

            # 1-order Euler integration
            noisy_action = model_output * step_size + noisy_action
            timestep += step_size

        return noisy_action

    # ========= Train  ============
    def compute_loss(
        self,
        state_tokens: torch.Tensor,
        action_gt : torch.Tensor,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_kv_cache: Optional[torch.Tensor] = None,
        lang_attn_mask: Optional[torch.Tensor] = None,
        img_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        Args:
            lang_tokens: (batch_size, depth, lang_len, lang_token_dim) or (batch_size, lang_len, lang_token_dim)
            lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
                which should be True-False bool tensor.
            img_tokens: (batch_size, img_len, img_token_dim)
            state_tokens: (batch_size, 1, state_dim)
            action_gt: (batch_size, horizon, action_dim), ground-truth actions for supervision

        Returns:
            loss_value, a scalar tensor
        '''
        batch_size = action_gt.shape[0]
        dtype = action_gt.dtype
        device = action_gt.device

        # Sample noise that we'll add to the actions
        noise = torch.randn(
            action_gt.shape, dtype=dtype, device=device
        )
        # Sample random diffusion timesteps
        timesteps = self.sample_timesteps(batch_size, device)
        broadcasted_timesteps = timesteps.view(-1, 1, 1)
        # Add noise to the clean actions
        # (this is the forward diffusion process)
        noisy_action = (action_gt * broadcasted_timesteps
                        + noise * (1 - broadcasted_timesteps))
        noisy_action = noisy_action.to(dtype=dtype)

        # Append the action mask to the input sequence
        # Align the dimension with the hidden size
        lang_cond, img_cond, action_traj, state_cond = self.adapt_conditions(
            lang_tokens, img_tokens, noisy_action, state_tokens
        )
        condition_inputs = self._prepare_condition_inputs(
            lang_cond=lang_cond,
            lang_kv_cache=lang_kv_cache,
            lang_attn_mask=lang_attn_mask,
            img_cond=img_cond,
            state_cond=state_cond,
        )
        # Predict the denoised result
        pred = self.model(
            x=action_traj,
            t=timesteps,
            **condition_inputs
        )

        # Compute the ground-truth velocity
        target = action_gt - noise

        # print(f"pred shape: {pred.shape}, target shape: {target.shape}")

        loss = F.mse_loss(pred, target)
        return loss

    def predict_velocity(
        self,
        lang_tokens: torch.Tensor,
        lang_kv_cache: torch.Tensor,
        lang_attn_mask: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        '''
        Args:
            lang_tokens: (batch_size, depth, lang_len, lang_token_dim) or (batch_size, lang_len, lang_token_dim)
            lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
                which should be True-False bool tensor.
            img_tokens: (batch_size, img_len, img_token_dim)
            state_tokens: (batch_size, state_history_len, hidden_size),
                state conditional data, if available.
            noisy_action: (batch_size, horizon, action_dim), noisy action sequence
            timesteps: (batch_size,), diffusion timesteps for the noisy action

        Returns:
            (batch_size, horizon, action_dim), predicted diffusion velocity
        '''
        lang_cond, img_cond, noisy_action_input, state_cond = self.adapt_conditions(
            lang_tokens, img_tokens, noisy_action, state_tokens
        )
        condition_inputs = self._prepare_condition_inputs(
            lang_cond=lang_cond,
            lang_kv_cache=lang_kv_cache,
            lang_attn_mask=lang_attn_mask,
            img_cond=img_cond,
            state_cond=state_cond,
        )
        return self.model(
            x=noisy_action_input,
            t=timesteps,
            **condition_inputs
        )

    # ========= Inference  ============
    @torch.no_grad()
    def predict_action(
        self,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_kv_cache: Optional[torch.Tensor] = None,
        lang_attn_mask: Optional[torch.Tensor] = None,
        img_tokens: Optional[torch.Tensor] = None,
        state_tokens: Optional[torch.Tensor] = None,
        noisy_action: Optional[torch.Tensor] = None,
    ):
        '''
        Args:
            lang_tokens: (batch_size, depth, lang_len, lang_token_dim)
            lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
                which should be True-False bool tensor.
            img_tokens: (batch_size, img_len, img_token_dim)
            state_tokens: (batch_size, state_history_len, hidden_size),
                state conditional data, if available.
            noisy_action: (batch_size, horizon, action_dim), optional,
                if provided, will be used as the initial noisy action for sampling.
                If not provided, a random noise will be generated.

        Returns:
            (batch_size, horizon, action_dim), predicted action sequence
        '''
        # Prepare the state and conditions
        lang_cond, img_cond, _, state_cond = self.adapt_conditions(
            lang_tokens=lang_tokens,
            img_tokens=img_tokens,
            action_tokens=None,
            state_tokens=state_tokens
        )

        # Run sampling
        action_pred = self.conditional_sample(
            lang_cond=lang_cond,
            lang_kv_cache=lang_kv_cache,
            lang_attn_mask=lang_attn_mask,
            img_cond=img_cond,
            state_cond=state_cond,
            noisy_action=noisy_action,
        )

        return action_pred

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)
