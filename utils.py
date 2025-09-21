import io
import random

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from data.umi.pose_util import geodesic_loss, rot6d_to_mat_torch


def unbatchfy(batch):
    """
    Unbatchify the batch to examples
    """
    batch_size = batch["action"].shape[0]
    examples = []
    
    for i in range(batch_size):
        meta = {
            "dataset_name": batch["meta"]["dataset_name"][i],
            "num_camera": min(batch["meta"]["num_camera"][i].item(), 2),    # less than 2 cameras
            "num_robot": batch["meta"]["num_robot"][i].item(),
        }
        example = {
            "meta": meta,
            "obs": {
                f"camera{j}_rgb": batch["obs"][f"camera{j}_rgb"][i]
                for j in range(meta["num_camera"])
            },
            "action": batch["action"][i],
            "action_token": batch["action_token"][i],
            "gripper_valid": batch["gripper_valid"][i].item(),
            "instruction": batch["instruction"][i],
        }
        examples.append(example)
        
    return examples


def compute_action_metrics(
    model, processor, vae, normalizer,
    examples, valid_action_id_length,
    num_robot=2, instruction=None,
    apply_jpeg_compression=False
):
    device = model.device

    if isinstance(examples, dict):
        # If the examples is a dict, it means the examples is batchified 
        # into a batch by the default collate function in PyTorch DataLoader
        # Unbatchify the batch to list of examples
        examples = unbatchfy(examples)
        
    gripper_valid_mask_lst = []
    gt_action_lst = []
    for example in examples:
        gripper_valid_mask_lst.append(example["gripper_valid"])
        gt_action_lst.append(example["action"])
    gripper_valid_mask = torch.tensor(
        gripper_valid_mask_lst).to(device, dtype=model.dtype)
    gt_action = torch.stack(gt_action_lst).to(device)

    result = batch_predict_action(
        model, processor, vae, normalizer,
        examples, valid_action_id_length,
        instruction=instruction,
        apply_jpeg_compression=apply_jpeg_compression
    )

    mask = (torch.sum(result["action_ids"], dim=1) != 0).to(device)
    if mask.sum() == 0:
        return {
            "action_valid_rate": torch.tensor(0.0).to(device),
            "action_mse_error": torch.tensor(0.0).to(device),
            "action_mse_error_pos": torch.tensor(0.0).to(device),
            "action_geodesic_error_rot": torch.tensor(0.0).to(device),
            "action_mse_error_width": torch.tensor(0.0).to(device),
        }

    action_valid_rate = mask.sum() / gt_action.shape[0]

    pred_action = result["action_pred"].to(device) # (B, horizon, 20)
    result = compute_action_errors(
        pred_action[mask], gt_action[mask], gripper_valid_mask[mask], num_robot)
    result["action_valid_rate"] = action_valid_rate

    return result


def compute_action_errors(
    pred_action, gt_action,
    gripper_valid_mask,
    num_robot
):
    B, T, _ = pred_action.shape
    action_shape = int(pred_action.shape[-1] / num_robot)
    assert action_shape == 10, "The action shape is not 10"

    pred_action = pred_action.view(B, T, -1, action_shape)
    gt_action = gt_action.view(B, T, -1, action_shape)
    gripper_valid_mask = gripper_valid_mask.view(B, 1, 1)
    gripper_valid_mask = gripper_valid_mask.expand(pred_action.shape[:-1])

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
    result['action_mse_error_width'] = (
        (F.mse_loss(pred_action[..., 9], gt_action[..., 9], reduction="none") * gripper_valid_mask).sum()
        / ((gripper_valid_mask).sum() + 1e-9)   # add eps to denominator to avoid nan
    )

    return result


def batch_predict_action(
    model, processor, vae, normalizer,
    examples, valid_action_id_length,
    instruction=None, 
    apply_jpeg_compression=False
):
    texts = []
    images = []
    for example in examples:
        example = preprocess_data_from_umi(
            example,
            instruction=instruction
        )

        images_per_example = example["images"]  # [H, W, C] x N
        image = torch.cat(images_per_example, dim=1)
        if apply_jpeg_compression:
            image = cv2.imencode('.jpg', image.detach().cpu().numpy()[..., ::-1])[1].tobytes()
            image = Image.open(io.BytesIO(image))
        else:
            image = Image.fromarray(image.detach().cpu().numpy()).convert("RGB")
        
        instruction = example["instruction"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        # insert guidance tokens that is 'assistant: <|quad_start|>' after text
        # to ensure the model can generate the action sequence
        text += "<|im_start|>assistant\n<|quad_start|>"
        texts.append(text)
        images.append([image])

    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
    # NOTE: +5 is a necessary hack to make sure the model can generate the action sequence
    # toggle do_sample=True for accelaration and determinstic sampling
    batch_generated_ids = model.generate(**inputs, max_new_tokens=(valid_action_id_length + 2))

    # fetch the generated_ids
    assert torch.all(inputs["input_ids"] == batch_generated_ids[:, :inputs["input_ids"].shape[1]]), \
        "The input_ids is not the same as the generated_ids"
    batch_generated_ids = batch_generated_ids[:, inputs["input_ids"].shape[1]:]

    batch_action_ids = []
    
    # Get token IDs for markers
    quad_end_id = processor.tokenizer.convert_tokens_to_ids("<|quad_end|>")
    
    action_ids = torch.zeros(valid_action_id_length, dtype=torch.long, device=batch_generated_ids.device)
    for generated_ids in batch_generated_ids:
        # Look for quad start/end markers
        quad_end_idx = (generated_ids == quad_end_id).nonzero(as_tuple=True)[0]
        
        if len(quad_end_idx) > 0 :
            # Use first set of markers found
            start_idx = 0
            end_idx = quad_end_idx[0]
            
            # Extract tokens between markers
            if end_idx - start_idx == valid_action_id_length:
                action_ids = generated_ids[start_idx: end_idx]    

        batch_action_ids.append(action_ids)

    batch_action_ids = torch.stack(batch_action_ids, dim=0)
    # from vocab_size - 1 -> (vocab_size - vae.num_embeddings) map to 0 -> vae.num_embeddings
    action_tokens = processor.tokenizer.vocab_size - (batch_action_ids + 1)
    # ensure action_tokens is within [0, vae.num_embeddings)
    action_tokens = torch.clamp(action_tokens, min=0, max=vae.num_embeddings - 1)

    # replace with vae (float32)
    nsample = vae.decode(action_tokens)
    action_pred = normalizer['action'].unnormalize(nsample).to(model.device)
    
    result = {
        'action': action_pred,
        'action_pred': action_pred,
        "action_ids": batch_action_ids,
    }

    return result


@torch.no_grad()
def preprocess_data_from_umi(
    data_dict, instruction=None
):
    # data_dict = flatten_dict(data_dict)
    result = {}

    images = []
    num_camera = data_dict["meta"]["num_camera"]
    for camera_id in range(num_camera):
        image = data_dict["obs"][f'camera{camera_id}_rgb'][-1]  # use the last image
        images.append(image)
        
    # NOTE: rearrange the images to left-right order 
    # to match the spatial common sense
    images = images[::-1]

    result["images"] = images
    
    result["instruction"] = data_dict.get("instruction", "")
    if instruction is not None:
        result["instruction"] = instruction
    assert result["instruction"] is not None, \
        "The instruction should not be None"

    if 'action' in data_dict:
        result.update({
            'action': data_dict['action'],
            'gripper_valid': data_dict['gripper_valid'],
            'dataset_name': data_dict["meta"]["dataset_name"],
        })
    if 'action_token' in data_dict:
        result['action_token'] = data_dict['action_token']

    return result