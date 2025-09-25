"""VLLM Utilities for Robot Action Prediction

This module provides utilities for using vLLM to accelerate inference
for vision-language models in robotic control applications. It includes
functions for predicting robot actions from visual observations using
a quantized LLM with optimized inference.

Inference latency on RTX5090: ~0.237s

Example usage:
    ```python
    from vllm import LLM, SamplingParams
    
    # Initialize components
    model = LLM(
        model=model_path, 
        dtype=torch.bfloat16,
        skip_tokenizer_init=True,
        tensor_parallel_size=1,  # Increase if you have multiple GPUs
        enforce_eager=False,     # Set to False to use CUDA graphs
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.90,
        max_model_len=2048, 
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True
    )
    # Set sampling parameters
    sampling_params = SamplingParams(
        max_tokens=32,
        temperature=0.0,    # Greedy decoding
        # temperature=0.1,  # Sampling temperature
        # top_k=1,          # Top-k sampling
        # top_p=0.001,      # Top-p (nucleus) sampling,
        # repetition_penalty=1.05,
        detokenize=False,
    )
    
    # load processor, vae, and normalizer
    ...
    
    # Predict action
    result = predict_action_vllm(
        model, sampling_params, processor, vae, normalizer,
        examples=[{"obs": obs_dict}],
        valid_action_id_length=valid_action_id_length,
        apply_jpeg_compression=False
    )
    
    # Extract predicted action
    action = result['action_pred'][0].detach().cpu().numpy()
    ```
"""
import io
import cv2
import torch
from PIL import Image

from utils import preprocess_data_from_umi
import time

def predict_action_vllm(
    vllm_model, sampling_params, 
    processor, vae, normalizer,
    examples, valid_action_id_length,
    instruction=None, 
    apply_jpeg_compression=False,
    device="cuda"
):
    # NOTE: device is passed as arugment for speed
    # which reduce the time to get device from model
    
    assert len(examples) == 1, \
        "predict_action_vllm currently only supports batch size of 1."
        
    # for example in examples:
    example = preprocess_data_from_umi(
        examples[0], instruction=instruction
    )

    images_per_example = example["images"]  # [H, W, C] x N
    image = torch.cat(images_per_example, dim=1)
    print(image.shape)
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
    text += "<|im_start|>assistant\n<|quad_start|>"
    image_lst = [image]

    start_inference_time = time.time()
    output = vllm_model.generate(
        {
            "prompt": text,
            "multi_modal_data": {
                "image": image_lst
            }
        },
        use_tqdm=False,
        sampling_params=sampling_params)
    generated_ids = output[0].outputs[0].token_ids
    print(f"Generated id time: {time.time() - start_inference_time}")
    # WARNING: for speed, we do not do valid inspection here
    action_ids = generated_ids[: valid_action_id_length]   # list
    batch_action_ids = torch.LongTensor(
        action_ids).unsqueeze_(0).to(device)    # (1, valid_action_id_length)
    # WARNING: if the output is illegal

    # from vocab_size - 1 -> (vocab_size - vae.num_embeddings) map to 0 -> vae.num_embeddings
    action_tokens = processor.tokenizer.vocab_size - (batch_action_ids + 1)
    # ensure action_tokens is within [0, vae.num_embeddings)
    action_tokens = torch.clamp(action_tokens, min=0, max=vae.num_embeddings - 1)
    
    nsample = vae.decode(action_tokens) # vae dtype: float32
    action_pred = normalizer['action'].unnormalize(nsample)
    print(f"Inference time: {time.time() - start_inference_time}")
    
    result = {
        'action': action_pred,
        'action_pred': action_pred,
        "action_ids": batch_action_ids,
    }

    return result

