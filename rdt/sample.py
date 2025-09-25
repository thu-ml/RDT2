from collections import defaultdict

import torch
import torch.nn.functional as F

from data.umi.pose_util import geodesic_loss, rot6d_to_mat_torch


def compute_action_errors(
    accelerator,
    pred_action, gt_action,
    num_robot
):
    B, T, _ = pred_action.shape
    action_shape = int(pred_action.shape[-1] / num_robot)
    assert action_shape == 10, "The action shape is not 10"

    pred_action = pred_action.view(B, T, -1, action_shape)
    gt_action = gt_action.view(B, T, -1, action_shape)
    
    # Use geodesic loss for rotation
    pred_rot6d = pred_action[..., 3:9]
    gt_rot6d = gt_action[..., 3:9]
    pred_rot_mat = rot6d_to_mat_torch(pred_rot6d).to(dtype=gt_rot6d.dtype)
    gt_rot_mat = rot6d_to_mat_torch(gt_rot6d)
    rot_error = geodesic_loss(pred_rot_mat, gt_rot_mat, reduce=True, return_degrees=True)

    result = {}
    result['action_mse_error'] = F.mse_loss(pred_action, gt_action)
    result['action_mse_error_pos'] = F.mse_loss(pred_action[..., :3], gt_action[..., :3])
    result['action_geodesic_error_rot'] = rot_error
    result['action_mse_error_width'] = F.mse_loss(pred_action[..., 9], gt_action[..., 9])
    
    gathered_result = {}
    for k, v in result.items():
        gathered_result[k] = accelerator.gather(v).mean().item()
    
    return gathered_result

@torch.no_grad()
def log_sample_res(
    vision_language_model, selected_layers, 
    vision_encoder, rdt, normalizer, args,
    accelerator, weight_dtype, dataloader, logger
):
    logger.info(
        f"Running sampling for {args.num_sample_batches} batches..."
    )

    rdt.eval()

    loss_for_log = defaultdict(float)
    loss_counter = defaultdict(int)
    for step, batch in enumerate(dataloader):
        if step >= args.num_sample_batches:
            break

        actions = batch["actions"].to(dtype=weight_dtype)
        states = batch["states"].to(dtype=weight_dtype) # (B, 1, 14)

        if vision_encoder is not None:
            images = {k: v.to(dtype=weight_dtype) for k, v in batch["images"].items()}
            k = next(iter(images))
            batch_size, _, C, H, W = images[k].shape
            for k in images:
                images[k] = images[k].reshape(-1, C, H, W)
            image_embeds = vision_encoder(images).detach()
            image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.embed_dim))
        else:
            image_embeds = None

        vlang_attn_mask = batch["vision_language_model_inputs"]["attention_mask"].to(dtype=torch.bool)
        outputs = vision_language_model(
            **batch["vision_language_model_inputs"],
            use_cache=True,
        )
        
        if isinstance(selected_layers, list):
            vlang_kv_cache = [
                outputs.past_key_values[i]
                for i in selected_layers
            ]
        else:
            vlang_kv_cache = [
                outputs.past_key_values[selected_layers]]

        pred_nsamples = rdt.predict_action(
            lang_kv_cache=vlang_kv_cache,
            lang_attn_mask=vlang_attn_mask,
            img_tokens=image_embeds,
            state_tokens=states,
        )
        pred_actions = normalizer["action"].unnormalize(pred_nsamples).to(rdt.device)
        
        action_errors = compute_action_errors(
            accelerator, pred_actions, actions, num_robot=2)
        for k, v in action_errors.items():
            loss_for_log[k] += v
            loss_counter[k] += 1

    for name in loss_for_log:
        # NOTE: do not round to 1e-4 for gripper width error is smaller than 1e-4
        loss_for_log[name] = loss_for_log[name] / loss_counter[name]

    rdt.train()
    torch.cuda.empty_cache()

    return dict(loss_for_log)