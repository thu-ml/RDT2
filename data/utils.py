import os
import glob
import json
import random

import yaml
import webdataset as wds

from data.umi_video_dataset import UmiVideoDataset


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
            # workersplit#ter=wds.split_by_worker,
            workersplitter=workersplitter,
            resampled=True,
        )
        .repeat()
        .shuffle(8192, initial=8192)
        .decode("pil")
        .map(
            lambda sample: {
                "image": sample["image.jpg"],
                "action_token": sample["action_token.npy"],
                "meta": sample["meta.json"],
            }
        )
        .with_epoch(nsamples=(2048 * 30 * 60 * 60))    # 2048 hours
    )
    
    return dataset


def get_instructions_and_blended_train_dataset(config):
    instruction_path = config["kwargs"]["instruction_path"]
    with open(instruction_path, "r") as f:
        instructions = json.load(f)
    
    if "addtional_instructions" in config["kwargs"]:
        instructions.update(config["kwargs"]["addtional_instructions"])
    
    if config["type"] == "single":
        return instructions, get_train_dataset(config["shards_dir"])
    
    # resolve the corresponding shards_dir
    datasets_recipe = config["train"]
    for dataset_name, dataset_config in datasets_recipe.items():
        with open(dataset_config["config_path"], "r") as f:
            dataset_config = yaml.safe_load(f)
        datasets_recipe[dataset_name]["shards_dir"] = dataset_config["shards_dir"]
    
    print(f"Using datasets for training: {list(datasets_recipe.keys())}")
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

def get_eval_datasets(config):
    if config["type"] == "single":
        dataset = UmiVideoDataset(
            dataset_name=config["dataset_name"],
            val_ratio=None,
            **config["kwargs"],
        )
        dataset = dataset.get_validation_dataset()
        return dataset
    
    if "addtional_instructions" in config["kwargs"]:
        addtional_instructions = config["kwargs"]["addtional_instructions"]
    
    datasets = {}
    for dataset_name, dataset_config in config["eval"].items():
        val_ratio = dataset_config.get("val_ratio", None)
        with open(dataset_config["config_path"], "r") as f:
            dataset_config = yaml.safe_load(f)
        
        dataset = UmiVideoDataset(
            dataset_name=dataset_name,
            val_ratio=val_ratio,
            **dataset_config["kwargs"],
        )
        dataset = dataset.get_validation_dataset()
        dataset.instructions.update(addtional_instructions)
        datasets[dataset_name] = dataset
    
    # if len(datasets) == 1:
    #     return datasets[0]
    
    return datasets
