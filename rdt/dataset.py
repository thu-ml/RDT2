import glob
import os
import json
import yaml
import random

import torch

import webdataset as wds

from data.image_corrupt import image_corrupt


def no_split(src):
    yield from src

def get_train_dataset(shards_dir):
    shards = sorted(glob.glob(os.path.join(shards_dir, "shard-*.tar")))
    random.shuffle(shards)
    
    num_workers = wds.utils.pytorch_worker_info()[-1]
    workersplitter = wds.split_by_worker if len(shards) > num_workers else no_split
    
    assert shards, f"No shards under {shards_dir}"
    dataset = (
        wds.WebDataset(
            shards,
            shardshuffle=False,
            nodesplitter=no_split,
            workersplitter=workersplitter,
            resampled=True,
        )
        .repeat()
        .shuffle(8192, initial=8192)
        .decode("pil")
        .map(
            lambda sample: {
                "image": sample["image.jpg"],
                "action": sample["action.npy"],
                "meta": sample["meta.json"],
            }
        )
        .with_epoch(nsamples=(2048 * 30 * 60 * 60))
    )
    
    return dataset


def get_instructions_and_blended_train_dataset(config):
    instruction_path = config["kwargs"]["instruction_path"]
    with open(instruction_path, "r") as f:
        instructions = json.load(f)
    
    if "addtional_instructions" in config["kwargs"]:
        instructions.update(config["kwargs"]["addtional_instructions"])
    
    if config["type"] == "single":
        print(f"Using single dataset for training: {config['name']}")
        return instructions, get_train_dataset(config["shards_dir"])
    
    # resolve the corresponding shards_dir
    datasets_recipe = config["train"]
    for dataset_name, dataset_config in datasets_recipe.items():
        with open(dataset_config["config_path"], "r") as f:
            dataset_config = yaml.safe_load(f)
        datasets_recipe[dataset_name]["shards_dir"] = dataset_config["shards_dir"]
    
    print(f"Using datasets for training: {list(datasets_recipe.keys())}")
    # print([dataset_config["shards_dir"] for dataset_config in datasets_recipe.values()])
    subsets = [
        get_train_dataset(dataset_config["shards_dir"])
        for dataset_config in datasets_recipe.values()
    ]
    weights = [
        dataset_config["weight"]
        for dataset_config in datasets_recipe.values()
    ]
    
    if len(subsets) == 1:
        return instructions, subsets[0]
    
    blended_dataset = wds.RandomMix(
        subsets, weights,
        longest=False,
    )
    
    return instructions, blended_dataset

def collate_fn(
    examples,
    processor,
    instructions,
    image_corruption,
    state_dim,
):
    texts = []
    images = []
    actions = []

    for example in examples:
        image = example["image"]
        action = torch.from_numpy(example["action"])    # [T, 20]

        if image_corruption:
            image = image_corrupt(image)

        instruction = instructions.get(example["meta"]["sub_task_instruction_key"], "")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        # insert guidance tokens that is 'assistant: <|quad_start|>' after text
        # to ensure the model can generate the action sequence
        text += "<|im_start|>assistant\n<|quad_start|>"
        
        texts.append(text)
        images.append([image])
        actions.append(action)

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    actions = torch.stack(actions, dim=0)  # (B, T, 20)
    
    batch = {
        "vision_language_model_inputs": inputs,
        "states": torch.zeros((actions.shape[0], 1, state_dim)),
        "actions": actions,
    }
    
    return batch