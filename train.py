import socket
from functools import partial

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoProcessor, BitsAndBytesConfig,
                          Qwen2_5_VLForConditionalGeneration,
                          TrainingArguments)

from data.image_corrupt import image_corrupt
from data.utils import (get_eval_datasets,
                        get_instructions_and_blended_train_dataset)
from models.normalizer import LinearNormalizer
from utils import compute_action_metrics
from vla_trainer import VLATrainer
from vqvae.models.multivqvae import MultiVQVAE


def train(args):
    # model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(args.tokenizer_name, use_fast=True)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.use_qlora or args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "down_proj",
                "o_proj",
                "k_proj",
                "q_proj",
                "gate_proj",
                "up_proj",
                "v_proj",
            ],
            use_dora=False if args.use_lora else True,
            init_lora_weights="gaussian",
        )
        lora_config.inference_mode = False
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=weight_dtype,
            )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.pretrained_model_name_or_path,
            quantization_config=bnb_config if args.use_qlora else None,
            attn_implementation="flash_attention_2",
            torch_dtype=weight_dtype,
            device_map=args.local_rank,
        )
        model.add_adapter(lora_config)
        model.enable_adapters()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=weight_dtype,
            attn_implementation="flash_attention_2",
            device_map=args.local_rank,
        )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", trainable_params)

    model.config.use_cache = False

    vae = MultiVQVAE.from_pretrained(args.vae_name)
    vae.eval()
    vae.to(model.device, dtype=torch.float32)

    valid_action_id_length = vae.pos_id_len + vae.rot_id_len + vae.grip_id_len

    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<action>"]},
        replace_additional_special_tokens=False,
    )

    with open(args.dataset, "r") as f:
        hostname = socket.gethostname()
        dataset_config_str = f.read().format(hostname=hostname)
        dataset_config = yaml.safe_load(dataset_config_str)

    instructions, train_ds = get_instructions_and_blended_train_dataset(dataset_config)
    
    if args.eval_strategy != "no":
        eval_ds = get_eval_datasets(dataset_config)
    else:
        eval_ds = None
    
    normalizer = LinearNormalizer.load(dataset_config["kwargs"]["normalizer_path"])
    num_robot = 2
    
    def collate_fn(examples):
        texts = []
        images = []
        actions = []

        for example in examples:
            image = example["image"]
            example["action_token"] = torch.from_numpy(example["action_token"]).to(
                dtype=torch.long
            )

            if args.image_corruption:
                image = image_corrupt(image)

            instruction = instructions.get(example["meta"]["sub_task_instruction_key"], "")
            
            action_tokens = example["action_token"]  # range: [0, num_embeddings)
            action_input_ids = processor.tokenizer.vocab_size - (action_tokens + 1)

            # NOTE: replace invalid gripper action_ids to <pad>
            # NOTE: utilize the <quad_start> and <quad_end> as <action_start> and <action_end>
            action_input_ids[action_tokens < 0] = processor.tokenizer.pad_token_id
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<|quad_start|>{'<action>' * len(action_input_ids)}<|quad_end|>"}
                    ],
                },
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])
            actions.append(action_input_ids)

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        for i, (input_ids, action) in enumerate(zip(batch["input_ids"], actions)):
            action_idx = (
                input_ids
                == processor.tokenizer.additional_special_tokens_ids[
                    processor.tokenizer.additional_special_tokens.index("<action>")
                ]
            )
            input_ids[action_idx] = action
            batch["input_ids"][i] = input_ids

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # labels[labels == image_token_id] = -100
        assistant_marker = "assistant"
        assistant_marker_id = processor.tokenizer.convert_tokens_to_ids(
            assistant_marker
        )

        for i in range(labels.shape[0]):
            input_ids = batch["input_ids"][i].tolist()
            try:
                start_index = input_ids.index(assistant_marker_id)
            except ValueError:
                # TODO(bangguo): inspect the occurrence of this error
                start_index = len(input_ids)
            labels[i, : start_index - 1] = -100
        batch["labels"] = labels

        return batch

    # https://github.com/QwenLM/Qwen2.5-VL/tree/35ba6e18636510de4bf8d4a7caaca3f4f5163a84?tab=readme-ov-file#training
    training_args = TrainingArguments(
        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_train_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.lr_warmup_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.checkpointing_steps,
        save_total_limit=args.checkpoints_total_limit,
        optim="adamw_torch",  # for 8-bit, keep this, else adamw_hf
        bf16=(args.mixed_precision == "bf16"),  # underlying precision for 8bit
        fp16=(args.mixed_precision == "fp16"),  # underlying precision for 8bit
        bf16_full_eval=(args.mixed_precision == "bf16"),
        fp16_full_eval=(args.mixed_precision == "fp16"),
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        hub_model_id=args.hub_model_id,
        report_to=args.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        # dataloader_persistent_workers=True, # NOTE: DO NOT TOGGLE this on, which may result threads leakage
        gradient_checkpointing=args.gradient_checkpointing,
        log_level=args.log_level,
        ignore_data_skip=isinstance(train_ds, torch.utils.data.IterableDataset),    # Do not skip the data when use IterableDataset, otherwise the resume will be EXTREMELY SLOW
        accelerator_config={
            "dispatch_batches": (
                False
                if isinstance(train_ds, torch.utils.data.IterableDataset)
                else None
            ),
        },
    )

    eval_processor = AutoProcessor.from_pretrained(
        args.tokenizer_name, padding_side="left", use_fast=False
    )
    compute_metrics = partial(
        compute_action_metrics,
        processor=eval_processor,
        vae=vae,
        normalizer=normalizer,  # model.normalizer
        valid_action_id_length=valid_action_id_length,
        num_robot=num_robot,
        # only use the first instruction for evaluation for instructed SFT
        instruction=None,
        # since validation image is dumped to jpeg, we don't need to apply jpeg compression
        apply_jpeg_compression=True,   
    )
    trainer = VLATrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        num_eval_batches=args.num_eval_batches,
        use_default_collate_fn_for_eval=args.use_default_collate_fn_for_eval,
    )

    if args.resume_from_checkpoint == "latest":
        args.resume_from_checkpoint = True

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if args.push_to_hub:
        trainer.push_to_hub(token=args.hub_token)
