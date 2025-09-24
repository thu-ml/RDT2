import re
import os
import io
from typing import Optional
# from collections import deque

import numpy as np
import torch
from PIL import Image
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from models.normalizer import LinearNormalizer
from rdt.models.rdt_runner import RDTRunner
from rdt.models.rdt_distill_runner import RDTDistillRunner


class RDTInferencer:
    """A wrapper for the RDT model for inference, which handles
            1. Model initialization
            2. Encodings of instructions
            3. Model inference
    """
    def __init__(
        self, config,
        pretrained_path,
        normalizer_path,
        pretrained_vision_encoder_name_or_path: Optional[str] = None,
        pretrained_vision_language_model_name_or_path=None,
        pre_compute_txt_embeddings_dir=None,
        device='cuda',
        dtype=torch.bfloat16,
    ):
        print(f"Initializing RDTInferencer with config: {config}.")

        self.config = config
        self.dtype = dtype
        self.device = device
        self.use_jpeg = config["dataset"].get("use_jpeg", False)
        self.his_len = config["common"]["img_history_size"]
        self.camera_names = config["dataset"]["camera_names"]
        self.action_chunk_size = self.config["common"]["action_chunk_size"]

        # if pretrained_vision_encoder_name_or_path is not None:
        #     self.image_transform, self.vision_encoder = \
        #         self.get_vision_encoder(pretrained_vision_encoder_name_or_path)
        # else:
        self.image_transform = None
        self.vision_encoder = None
        
        if pretrained_vision_language_model_name_or_path is None and pre_compute_txt_embeddings_dir is None:
            raise ValueError("Either pretrained_text_encoder_name_or_path or pre_compute_txt_embeddings_dir must be provided.")
        if pretrained_vision_language_model_name_or_path is not None:
            self.processor, self.vision_language_model = self.get_vision_language_model(pretrained_vision_language_model_name_or_path)
        
        self.precomputed_lang_embeds = None
        # If GPU memory is limited, we will load the precomputed text embeddings
        if pre_compute_txt_embeddings_dir is not None:
            if not os.path.exists(pre_compute_txt_embeddings_dir):
                raise ValueError(f"Pre-compute text embeddings directory {pre_compute_txt_embeddings_dir} does not exist.")
            lang_embed_pt_paths = [os.path.join(pre_compute_txt_embeddings_dir, f) for
                f in os.listdir(pre_compute_txt_embeddings_dir) if re.fullmatch(r'lang_embed_\d+\.pt', f)]
            self.precomputed_lang_embeds = [torch.load(pt_path) for pt_path in lang_embed_pt_paths]

        self.policy = self.get_policy(pretrained_path)
        self.normalizer = LinearNormalizer.load(normalizer_path)
        # self.observation_window = None
        self.lang_embeds_cache = {}

        self.reset()

    def get_policy(self, pretrained):
        """Initialize the model."""
        RDT_CFG = {
            "state_dim": self.config["common"]["state_dim"],
            "action_dim": self.config["common"]["action_dim"],
            "pred_horizon": self.config["common"]["action_chunk_size"],
            "config": self.config["model"],
            "act_pos_emb_config": [
                ('action', self.config["common"]["action_chunk_size"]),
                # Learnable register tokens
                ('register', self.config["model"]["rdt"]["num_register_tokens"]),
            ],
            "dtype": self.dtype,
        }
        
        if self.config["model"].get("lang_adaptor", None) is not None:
            RDT_CFG.update({
                "lang_token_dim": self.config["model"]["lang_token_dim"],
            })
        
        if self.vision_encoder is not None:
            img_cond_len = (self.config["common"]["img_history_size"]
                * self.config["common"]["num_cameras"]
                * self.vision_encoder.num_patches)
            RDT_CFG.update({
                "img_pos_emb_config": [
                    ("image", (self.config["common"]["img_history_size"],
                        self.config["common"]["num_cameras"],
                        -self.vision_encoder.num_patches)),
                ],
                "max_img_len": img_cond_len,
            })

        # Initialize model with arguments
        print(f"create from pretrained, pretrained: {pretrained}")
        try:
            _model = RDTRunner.from_pretrained(pretrained)
        except TypeError:
            print("Failed to load RDTRunner, trying RDTDistillRunner")
            _model = RDTDistillRunner.from_pretrained(pretrained, rdt_config=RDT_CFG)

        return _model

    def get_vision_language_model(self, pretrained_vision_language_model_name_or_path):
        vision_language_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_vision_language_model_name_or_path,
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
            device_map=self.device,
        )
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", padding_side="left", use_fast=True)
        return processor, vision_language_model

    # def get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
    #     vision_encoder = DinoSigLIPViTBackbone(
    #         vision_backbone_id=pretrained_vision_encoder_name_or_path,
    #         image_resize_strategy="letterbox"
    #             if self.config["dataset"]["image_aspect_ratio"] == "pad"
    #             else "resize-naive",
    #         default_image_size=384
    #     )
    #     image_transform = vision_encoder.get_image_transform()
    #     return image_transform, vision_encoder

    def reset(self):
        """
        Set model to evaluation mode.
        And reset every state.

        CALL this function every time you start a new task.
        """
        device = self.device
        weight_dtype = self.dtype
        self.policy.eval()
        self.vision_language_model.eval()
        
        self.policy = self.policy.to(device, dtype=weight_dtype)
        
        if self.vision_encoder is not None:
            self.vision_encoder.eval()
            self.vision_encoder = self.vision_encoder.to(device, dtype=weight_dtype)

        self.lang_embeds_cache = {}
        # self.observation_window = None

    def clear_lang_cache_if_exceed(self):
        MAX_CACHE_SIZE = 1024
        if len(self.lang_embeds_cache) > MAX_CACHE_SIZE:
            self.lang_embeds_cache = {}
            print("Cleared language instruction cache due to exceeding size limit.")


    def prepreare_data_from_vision_language_model(self, images, instruction):
        """
        Prepare data from vision language model.

        Args:
            images (List[np.ndarray]): a list of images
            instruction (str): a string of instruction

        Returns:
            intpus (Dict): inputs for vision language model
        """
        all_texts = []
        all_images = []
        
        # for example in examples:
        # ensure the images are left -> right
        image = np.concatenate(images, axis=1)  # (384, 768, 3)
        if self.use_jpeg:
            image = cv2.imencode('.jpg', image[..., ::-1])[1].tobytes()
            image = Image.open(io.BytesIO(image))
        else:
            image = Image.fromarray(image).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        # insert guidance tokens that is 'assistant: <|quad_start|>' after text
        # to ensure the model can generate the action sequence
        text += "<|im_start|>assistant\n<|quad_start|>"
        all_texts.append(text)
        all_images.append([image])
        
        inputs = self.processor(
            text=all_texts, images=all_images, padding=True, return_tensors="pt").to(self.device)

        return inputs
    
    def encode_image_and_instruction(self, images, instruction):
        """Encode string instruction to latent embeddings.

        Args:
            images: a list of images (from left to right)
            instruction: a string of instruction
            device: a string of device

        Returns:
            vlang_embdeds: a tensor of latent embeddings of shape 
                (1, depth, seq_len, hidden_dim) or (1, seq_len, hidden_dim)
            vlang_attn_mask: a tensor of attention mask of shape (1, seq_len)
        """
        with torch.no_grad():
            inputs = self.prepreare_data_from_vision_language_model(images, instruction)
            vlang_attn_mask = inputs["attention_mask"]
            vlang_attn_mask = vlang_attn_mask.to(self.device, dtype=torch.bool)
            
            outputs = self.vision_language_model(
                **inputs,
                use_cache=True,
            )
            
            if isinstance(self.config["model"]["selected_layers"], list):
                vlang_kv_cache = [
                    outputs.past_key_values[i]
                    for i in self.config["model"]["selected_layers"]
                ]
            else:
                vlang_kv_cache = [
                    outputs.past_key_values[self.config["model"]["selected_layers"]]]
            
        return vlang_kv_cache, vlang_attn_mask
    
    @torch.inference_mode()
    def step(self, observations, instruction):
        """
        Predict the next action chunk given the observation (
        proprioceptive states and images) and language instruction.

        Args:
            observations: a dict containing:
                images (RGB images):
                    <cam_1> (array; H, W, 3)
                    <cam_2> (array; H, W, 3)
                    ...
                state (array; state_dim,)
            instruction: a string of language instruction

        Returns:
            action: predicted action, (horizon, action_dim)
        """
        # self.update_observation_window()

        device = self.device
        dtype = self.dtype

        assert len(observations['images'].keys()) == len(self.camera_names), \
            f"Expected {len(self.camera_names)} images, but got {len(observations['images'].keys())}."
        images = [observations['images'][cam_name] for cam_name in self.camera_names]

        proprio = observations['state']
        proprio = torch.from_numpy(proprio).reshape(1, 1, -1).to(device, dtype=dtype)
            # (1, 1, 26) batch_size=1, horizon=1
        
        # Preprocess the images by order and encode them
        if self.vision_encoder is not None:
            # JPEG transformation
            # Align with training
            def jpeg_mapping(img):
                if img is None:
                    return None
                img = cv2.imencode('.jpg', img)[1].tobytes()
                img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                return img
            
            all_pixel_values = []
            for image in images:
                if self.use_jpeg:
                    image = jpeg_mapping(image)
                pimage = Image.fromarray(image)
                pixel_values = self.image_transform(pimage) # dict: dino, siglip, etc.
                all_pixel_values.append(pixel_values)
            
            pv_example = all_pixel_values[0]
            merged_pixel_values = {
                k: torch.stack(
                    [pv[k] for pv in all_pixel_values]
                )
                for k in pv_example
            } # dict[str, Tensor]: {dino: [N, C, H, W], siglip: [N, C, H, W], ...}
            merged_pixel_values = {
                k: v.to(device, dtype=dtype)
                for k, v in merged_pixel_values.items()
            }

            image_embeds = self.vision_encoder(merged_pixel_values).detach()
            image_embeds = image_embeds.reshape((1, -1, self.vision_encoder.embed_dim))

            image_embeds = image_embeds.to(device, dtype=dtype)
        else:
            image_embeds = None
          
        vlang_kv_cache, vlang_attn_mask = self.encode_image_and_instruction(images, instruction)
        # text_embeds = text_embeds.to(device, dtype=dtype)
        # text_attn_mask = text_attn_mask.to(device, dtype=torch.bool)
        
        # Predict the next action chunk given the inputs
        nsample = self.policy.predict_action(
            lang_kv_cache=vlang_kv_cache,
            lang_attn_mask=vlang_attn_mask,
            img_tokens=image_embeds,
            state_tokens=proprio,
        ) # torch.Tensor: (1, horizon, action_dim)
        # convert to cpu and float32 to match the device and dtype of the normalizer
        nsample = nsample.to(dtype=torch.float32).cpu()
        
        trajectory = self.normalizer['action'].unnormalize(nsample)     # torch.Tensor: (1, horizon, action_dim)
        
        self.clear_lang_cache_if_exceed()

        return trajectory.squeeze(0)    # torch.Tensor: (horizon, action_dim)


# Test
if __name__ == "__main__":
    import yaml
    CONFIG_PATH = "configs/demo.yaml"
    config = yaml.safe_load(open(CONFIG_PATH, 'r'))
    TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
    VISION_ENCODER_NAME="dinosiglip-vit-so-384px"
    CHECKPOINT="checkpoints/rdt2-demo/20250803_234020/checkpoint-50000"

    model = RDTInferencer(
        config=config,
        pretrained_path=CHECKPOINT,
        pretrained_vision_encoder_name_or_path=VISION_ENCODER_NAME,
        pretrained_text_encoder_name_or_path=TEXT_ENCODER_NAME,
        pre_compute_txt_embeddings_dir=None,  # Set to None for now
        device='cuda',
    )

    observations = {
        'images': {
            'exterior_rs': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'left_stereo': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'right_stereo': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        },
        'state': np.random.rand(14).astype(np.float32)  # Example state
    }
    instruction = "fold the clothes"
    action_chunk = model.step(observations, instruction).cpu().numpy()
        # (24, 14)
    print(action_chunk.shape)
