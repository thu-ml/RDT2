
<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <h1 style="font-size: 4rem; font-weight: bold; color: #667eea; margin: 20px 0; display: flex; align-items: center; justify-content: center; gap: 20px;">
    <!-- <img src="assets/tsail_rdt.png" alt="TSAIL RDT" style="height: 8rem; width: auto;" /> -->
    RDT2: Enabling Zero-Shot Cross-Embodiment Generalization by Scaling Up UMI Data
  </h1>
</div>
<!-- <hr> -->
<div align="center" style="line-height: 1;">
  <a href="https://rdt-robotics.github.io/rdt2/"><img alt="Homepage"
    src="https://img.shields.io/badge/RDT%202-Homepage-4287f5?logo=probot&logoColor=#009BD5"/></a>
  <a href="https://huggingface.co/robotics-diffusion-transformer"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TSAIL%20RDT-ffc107?color=ffc107&logoColor=white"/></a>
  <br>
  <a href="https://discord.gg/vsZS3zmf9A"><img alt="Discord"
    src="https://img.shields.io/badge/Discord-RDT-7289da?logo=discord&logoColor=white&color=7289da"/></a>
<a href="https://rdt-robotics.github.io/rdt2/feishu.html"><img alt="Feishu"
    src="https://img.shields.io/badge/Feishu-RDT-blue?logo=lark&logoColor=white"/></a>
  <!-- <a href="https://twitter.com/deepseek_ai"><img alt="Twitter Follow"
    src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white"/></a> -->
  <!-- <br> -->
  <a href="LICENSE"><img alt="License"
    src="https://img.shields.io/badge/License-Apache--2.0-f5de53?logo=apache&color=f5de53"/></a>
  <!-- <a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL"><img alt="Model License"
    src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53"/></a> -->
  <br>
  <!-- <a href="https://arxiv.org/pdf/2412.19437"><b>Blog Link</b>👁️</a>  -->
  <!-- <a href="https://arxiv.org/pdf/2412.19437"><b>Paper Link</b>📄</a> -->
</div>

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Updates](#updates)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Checkpoints](#model-checkpoints)
- [Running Inference for a Pre-Trained Model](#running-inference-for-a-pre-trained-model)
  <!-- - [1. \[IMPORTANT\] Hard-ware Set up and Calibration](#1-important-hard-ware-set-up-and-calibration)
  - [2. Run Inference](#2-run-inference) -->
- [Fine-Tuning Models on Your Own Data](#fine-tuning-models-on-your-own-data)
  <!-- - [1. Convert your data to WebDataset shards](#1-convert-your-data-to-webdataset-shards)
  - [2. Defining training configs and running training](#2-defining-training-configs-and-running-training)
  - [3. Run training](#3-run-training)
    - [RDT2-VQ](#rdt2-vq)
  - [Precision Settings](#precision-settings) -->
- [Troubleshooting](#troubleshooting)

## Overview

RDT2, the sequel to [RDT-1B](https://rdt-robotics.github.io/rdt-robotics/), is the first foundation model that can achieve **zero-shot deployment** on **unseen embodiments** for **simple open-vocabulary** tasks like picking, placing, shaking, wiping, etc. This milestone was made possible by multifaceted efforts:

- We redesigned the [UMI hardware](https://umi-gripper.github.io) by applying higher-strength materials and more precise tracking methods, ensuring its reliability for large-scale data collection.
- We collected **10,000+ hours** of human manipulation videos in **100+ different indoor scenes**, covering the majority of household tasks that a gripper can do.

Currently, this repo contains models:
- the [RDT2-VQ](https://huggingface.co/robotics-diffusion-transformer/RDT2-VQ), an auto vision-language-action model (VLA) which employs [Residual VQ](https://arxiv.org/abs/2107.03312) as the action tokenizer, is adapted from [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) with our UMI dataset, enabling superior zero-shot instruction-following capability.
- the [RDT2-FM](https://huggingface.co/robotics-diffusion-transformer/RDT2-FM), an improved RDT model as action expert with flow-matching objective, running with much lower inference latency.

For all models, we provide checkpoints and examples for using them out of the box or fine-tuning them to your own datasets. Currently, we have verified the efficay of our models on platforms including [Bimanual UR5e](https://www.universal-robots.com/products/ur5e/) and [Bimanual Franka Research 3](https://franka.de/franka-research-3), and we are optimistic will able to deploy them successfully on more platforms in the future by following our [guidelines](#running-inference-for-a-pre-trained-model).


## Updates

- [Sept 2025] We released [RDT2-VQ](https://huggingface.co/robotics-diffusion-transformer/RDT2-VQ) \& [RDT2-FM](https://huggingface.co/robotics-diffusion-transformer/RDT2-FM), the sequel of RDT-1B with better open-world generalization and zero-shot deployment on unseen embodiments.

## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism or offload into CPU to reduce per-GPU memory. Since RDT2 is based on Qwen2.5-VL-7B, you basiclly need to follow the hard-ware requirements for Qwen2.5-VL-7B:

| Mode               | RAM Required | VRAM Required | Example GPU        |
| ------------------ | --------------- | --------------- | ------------------ |
| Inference          | > 32 GB      | ~ 16 GB | RTX 4090           |
| Fine-Tuning RDT2-FM (RDT Expert) |   -     | ~ 16 GB | RTX 4090           |
| Fine-Tuning RDT2-VQ (LoRA) |   -     | > 32 GB | A100 (40GB)           |
| Fine-Tuning RDT2-VQ (Full) |   -    |  > 80 GB  | A100 (80GB) / H100 / B200|

As for zero-shot deployment, you need to purchase the designated _end effector_ and _camera_, and 3D print the corresponding _camera stand_ and _flange_ according to [Harware Set up and Calibration](#1-important-hard-ware-set-up-and-calibration).

The repo has been tested with Ubuntu 24.04, we do not currently support other operating systems.

## Installation

Clone this repo and create a conda environment:

```bash
# clone the repo
git clone https://github.com/thu-ml/RDT2.git
cd RDT2

# create a conda environment
conda create -n rdt2 python=3.10 -y
conda activate rdt2

# install torch (cuda12.8)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# install flash 
pip install flash-attn --no-build-isolation

# install other dependencies
pip install -r requirements.txt

# upgrade nvidia-nccl-cu12
pip install --upgrade --force-reinstall nvidia-nccl-cu12==2.27.5

# Double check that you have the right transformers 4.51.3 installed
pip list | grep transformers

# to deploy on UR5e
pip install -r requirements/ur5e.txt

# to deploy on Franka Research 3
pip install -r requirements/franka_research_3.txt
```

## Model Checkpoints

<!-- ###  Models -->
We provide multiple VLA model checkpoints with capabilities to deploy on various robot platforms and simple vocabulary tasks. If you want to deploy on your own robot platform with other end effectors and cameras, you can fine-tune from the base model.


| Model        | Use Case    | Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| normalizer      | Inference & Fine-Tuning (Freeze) | Normalizer for action normalization   | [umi_normalizer_wo_downsample_indentity_rot.pt](http://ml.cs.tsinghua.edu.cn/~lingxuan/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt)    |
| Residual VQ  | Inference & Fine-Tuning (Freeze) |  Residual VQ (RVQ) as the action tokenizer   | [`robotics-diffusion-transformer/RVQActionTokenizer`](https://huggingface.co/robotics-diffusion-transformer/RVQActionTokenizer)    |
| RDT2-VQ      | Inference & Fine-Tuning | Auto-regressive VLA with Residual VQ as the action tokenizer   | [`robotics-diffusion-transformer/RDT2-VQ`](https://huggingface.co/robotics-diffusion-transformer/RDT2-VQ)    |
| RDT2-FM      | Inference & Fine-Tuning | Auto-regressive VLA (RDT2-VQ) with Flow-Matching Action Expert   | [`robotics-diffusion-transformer/RDT2-FM`](https://huggingface.co/robotics-diffusion-transformer/RDT2-FM)    |

<!-- | $\pi_0$-FAST | Fine-Tuning | Base autoregressive [π₀-FAST model](https://www.physicalintelligence.company/research/fast) for fine-tuning | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$    | Fine-Tuning | Base [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05) for fine-tuning    | `gs://openpi-assets/checkpoints/pi05_base`      | -->

<!-- ### Fine-Tuned Models -->


<!-- | Model                    | Use Case    | Description                                                                                                                                                                                              | Checkpoint Path                                       |
| ------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| $\pi_0$-FAST-DROID       | Inference   | $\pi_0$-FAST model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): can perform a wide range of simple table-top manipulation tasks 0-shot in new scenes on the DROID robot platform | `gs://openpi-assets/checkpoints/pi0_fast_droid`       |
| $\pi_0$-DROID            | Fine-Tuning | $\pi_0$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): faster inference than $\pi_0$-FAST-DROID, but may not follow language commands as well                                | `gs://openpi-assets/checkpoints/pi0_droid`            |
| $\pi_0$-ALOHA-towel      | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can fold diverse towels 0-shot on ALOHA robot platforms                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_towel`      |
| $\pi_0$-ALOHA-tupperware | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can unpack food from a tupperware container                                                                                                             | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap  | Inference   | $\pi_0$ model fine-tuned on public [ALOHA](https://dit-policy.github.io/) data: can uncap a pen                                                                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |
| $\pi_{0.5}$-LIBERO      | Inference   | $\pi_{0.5}$ model fine-tuned for the [LIBERO](https://libero-project.github.io/datasets) benchmark: gets state-of-the-art performance (see [LIBERO README](examples/libero/README.md)) | `gs://openpi-assets/checkpoints/pi05_libero`      |
| $\pi_{0.5}$-DROID      | Inference / Fine-Tuning | $\pi_{0.5}$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/) with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation): fast inference and good language-following | `gs://openpi-assets/checkpoints/pi05_droid`      | -->

## Running Inference for a Pre-Trained Model

### 1. [IMPORTANT] Hard-ware Set up and Calibration

1. Acquire deployment hardware according to our [Hardware Guide](https://docs.google.com/document/d/1HUeM4Wlt4PyINoEwci-hxm8U9wAxiPMgR3sHyaOAsck/edit?tab=t.0#heading=h.sbdalb8w1kk1).

2. Set up Robots

- 2.1 Set up UR5e  
   - Obtain IP address and update [configs/robots/eval_bimanual_ur5e_config.yaml](configs/robots/eval_bimanual_ur5e_config.yaml)/robots/robot_ip.  
  - In Installation > Payload  
    - Set mass to 0.82 kg  
    - Set Inertia Matrix to  
      ```python
      [0.001106, 0, 0,
       0, 0.001106, 0,
       0, 0, 0.001106]
      ```
    - Set speed to 30%(recommened)
  
- 2.2 Set up Franka FR3  
  - Obtain IP address and update [configs/robots/eval_bimanual_fr3_config.yaml](configs/robots/eval_bimanual_fr3_config.yaml)/robots/robot_ip.  
  - On the Franka interface website  
    - Set gripper mass to 1.9 kg  
    - Set Inertia Tensor to  
      ```python
      [0.001, 0, 0,
       0, 0.001, 0,
       0, 0, 0.001]
      ```

3. Set up camera
   * Download SDK from [HikRobot website](https://www.hikrobotics.com/cn/machinevision/service/download/?module=0) and install all the `.deb` files.
   * Run `cd /opt/MVS/bin && ./MVS.sh`. Select your camera, and set Acquisition Control -> Exposure Time to 20000.
  
4. Calibrate your robot to tracker's tcp space
 * Follow Setup Instructions For Calibration in [Hardware Guide](https://docs.google.com/document/d/1HUeM4Wlt4PyINoEwci-hxm8U9wAxiPMgR3sHyaOAsck/edit?tab=t.0#heading=h.sbdalb8w1kk1).
 * Set up Vive Tracker according to this [tutorial](https://docs.google.com/document/d/1ANxSA_PctkqFf3xqAkyktgBgDWEbrFK7b1OnJe54ltw/edit?tab=t.0#heading=h.yxlxo67jgfyx) ->Software Setup Tutorial ->VIVE tracker setup
 * Run the following code to calibrate robot tcp space to tracker's space.
 * IMPORTANT: This script makes the robot perform small-amplitude sinusoidal motions; before running the script, please ensure the robot is in a safe position and the workspace is free of obstacles.
    ```
    python deploy/calibration/calibrate_franka.py --franka_ip <your_franka_server_ip> --franka_port <your_franka_server_port> # if using Franka Research 3
    # or
    python deploy/calibration/calibrate_ur5e.py --ur5e_ip <your_ur5e_ip> # if using UR5e
    ```
  * After calibration, run the following script to obtain the calibration matrix:
    ```
    python deploy/calibration/compute_calibration_matrix.py
    ```
    Then paste the calibration matrix to eval_bimanual_ur5e_config.yaml/tx_tracker_to_tcp (or eval_bimanual_fr3_config.yaml/tx_tracker_to_tcp if using FR3).

### 2. Run Inference

Our pre-trained model checkpoints can be run with a few lines of code (here our [RDT2-VQ model](https://huggingface.co/robotics-diffusion-transformer/RDT2-VQ)):
```python
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from vqvae import MultiVQVAE
from models.normalizer import LinearNormalizer
from utils import batch_predict_action

# assuming using gpu 0
device = "cuda:0"


processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "robotics-diffusion-transformer/RDT2-VQ"
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device
).eval()
vae = MultiVQVAE.from_pretrained("robotics-diffusion-transformer/RVQActionTokenizer").eval()
vae = vae.to(device=device, dtype=torch.float32)

valid_action_id_length = (
    vae.pos_id_len + vae.rot_id_len + vae.grip_id_len
)
# TODO: modify to your own downloaded normalizer path
# download from http://ml.cs.tsinghua.edu.cn/~lingxuan/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt
normalizer = LinearNormalizer.from_pretrained("umi_normalizer_wo_downsample_indentity_rot.pt")  # 

result = batch_predict_action(
    model,
    processor,
    vae,
    normalizer,
    examples=[
        {
            "obs": {
                # NOTE: following the setting of UMI, camera0_rgb for right arm, camera1_rgb for left arm
                "camera0_rgb": ..., # right arm RGB image in np.ndarray of shape (1, 384, 384, 3) with dtype=np.uint8
                "camera1_rgb": ..., # left arm RGB image in np.ndarray of shape (1, 384, 384, 3) with dtype=np.uint8
            },
            "meta": {
                "num_camera": 2
            }
        },
        ...,    # we support batch inference, so you can pass a list of examples
    ]，
    valid_action_id_length=valid_action_id_length,
    apply_jpeg_compression=True,
    # Since model is trained with mostly jpeg images, we suggest toggle this on for better formance
    instruction="Pick up the apple."
    # We suggest using Instruction in format "verb + object" with Capitalized First Letter and trailing period 
)

# get the predict action from example 0
action_chunk = result["action_pred"][0] # torch.FloatTensor of shape (24, 20) with dtype=torch.float32
# action_chunk (T, D) with T=24, D=20
#   T=24: our action_chunk predicts the future 0.8s in fps=30, i.e. 24 frames
#   D=20: following the setting of UMI, we predict the action for both arms from right to left
#   - [0-2]: RIGHT ARM end effector position in x, y, z (unit: m)
#   - [3-8]: RIGHT ARM end effector rotation in 6D rotation representation
#   - [9]: RIGHT ARM gripper width (unit: m)
#   - [10-12]: LEFT ARM end effector position in x, y, z (unit: m)
#   - [13-18]: LEFT ARM end effector rotation in 6D rotation representation
#   - [19]: LEFT ARM gripper width (unit: m)

# rescale gripper width from [0, 0.088] to [0, 0.1]
for robot_idx in range(2):
    action_chunk[:, robot_idx * 10 + 9] = action_chunk[:, robot_idx * 10 + 9] / 0.088 * 0.1
```

And you can also test the [RDT2-FM](https://huggingface.co/robotics-diffusion-transformer/RDT2-FM) with the following code:
```python
# Run under root directory of our repo
import yaml

from models.rdt_inferencer import RDTInferencer


with open("configs/rdt/post_train.yaml", "r") as f:
  model_config = yaml.safe_load(f)

model = RDTInferencer(
  config=model_config,
  pretrained_path="robotics-diffusion-transformer/RDT2-FM",
  # TODO: modify `normalizer_path` to your own downloaded normalizer path
  # download from http://ml.cs.tsinghua.edu.cn/~lingxuan/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt
  normalizer_path="umi_normalizer_wo_downsample_indentity_rot.pt",  
  pretrained_vision_language_model_name_or_path="robotics-diffusion-transformer/RDT2-VQ", # use RDT2-VQ as the VLM backbone
  device="cuda:0",
  dtype=torch.bfloat16,
)

result = model.step(
    observations={
        'images': {
            'left_stereo': ..., # left arm RGB image in np.ndarray of shape (384, 384, 3) with dtype=np.uint8
            'right_stereo': ..., # right arm RGB image in np.ndarray of shape (384, 384, 3) with dtype=np.uint8
        },
        # use zero input current state for currently
        # preserve input interface for future fine-tuning
        'state': np.zeros(model_config["common"]["state_dim"]).astype(np.float32)
    },
    instruction="Pick up the apple." # Language instruction
    # We suggest using Instruction in format "verb + object" with Capitalized First Letter and trailing period 
)


# relative action chunk in np.ndarray of shape (24, 20) with dtype=np.float32
# with the same format as RDT2-VQ
action_chunk = result.detach().cpu().numpy()

# rescale gripper width from [0, 0.088] to [0, 0.1]
for robot_idx in range(2):
    action_chunk[:, robot_idx * 10 + 9] = action_chunk[:, robot_idx * 10 + 9] / 0.088 * 0.1
```

<!-- You can also test this out in the [example notebook](examples/inference.ipynb). -->

We provide detailed step-by-step examples for running inference of our pre-trained checkpoints on [Bimanual UR5e](examples/ur5e/README.md) and [Bimanual Franka Research 3](examples/fr3/README.md) robots.

IMPORTANT: If the inference success rate is still low after checking all the settings, configurations and calibrations, you can refer to the [deployment tips](./examples/DEPLOYMENT_TIPS.md) for help.

<!-- **Remote Inference**: We provide [examples and code](docs/remote_inference.md) for running inference of our models **remotely**: the model can run on a different server and stream actions to the robot via a websocket connection. This makes it easy to use more powerful GPUs off-robot and keep robot and policy environments separate. -->

<!-- **Test inference without a robot**: We provide a [script](examples/simple_client/README.md) for testing inference without a robot. This script will generate a random observation and run inference with the model. See [here](examples/simple_client/README.md) for more details. -->


## Fine-Tuning Models on Your Own Data

We will fine-tune the RDT2 models on the [example dataset from Bimanual UR5e](https://huggingface.co/datasets/robotics-diffusion-transformer/BimanualUR5eExample) as a running example for how to fine-tune a base model on your own data. We will explain three steps:
1. Convert your data to a [webdataset](https://github.com/webdataset/webdataset) shards (which we use for training for high-efficent IO)
2. Define training configs
3. Run training

### 1. Convert your data to WebDataset shards

We provide example scripts for converting assumed data sturcture to a webdataset dataset in [`data/preprocess/robot`](data/preprocess/robot) with detailed [guidelines](data/preprocess/robot/README.md). You can easily modify it to convert your own data! 

Concretely, the sturcture of processed webdatset shardsis like this:
```bash 
shard-000000.tar
├── 0.image.jpg   # Binocular (left wrist camera + right wrist camera) RGB image in np.ndarray of shape (384, 768, 3) with dtype=np.uint8
├── 0.action.npy  # Relative action chunk in np.ndarray of shape (24, 20) with dtype=np.float32
├── 0.action_token.npy # Corresponding action token in np.ndarray of shape (27,) ranging from 0 to 1024 with dtype=np.int16
├── 0.meta.json # Meta data including key `sub_task_instruction_key` to index the corresponding instruction from `instructions.json`
├── 1.image.jpg
├── 1.action.npy
├── 1.action_token.npy
├── 1.meta.json
├── ...
shard-000001.tar
shard-000002.tar
...
```

Moreover, we provde processed [example data](https://huggingface.co/datasets/robotics-diffusion-transformer/BimanualUR5eExample) collected with Bimanual UR5e on huggingface. You can download it and use it directly.

### 2. Defining training configs and running training

Define your dataset config following format in [`configs/datasets/example.yaml`](configs/datasets/example.yaml)
```yaml
# Define your dataset name here
name: <your_dataset_name> # e.g. bimanual/ur_example
type: single
shards_dir: <your_shards_dir> # e.g. /ssd/rdt2/bimanual_fold_cloth/shards 
kwargs:
  instruction_path: <your_instruction_path> # e.g. /ssd/rdt2/ur_example/instruction.json
  normalizer_path: <your_normalizer_path> # e.g. /ssd/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt
```

For the provided example data, its corresponding config is in [`configs/datasets/example.yaml`](configs/datasets/example.yaml). Remember to replace the `<root_dir>` and `<path_to_normalizer>` with your own path for downloading.

### 3. Run training

#### RDT2-VQ

Currently, we support the following fine-tuning methods:

- DeepSpeed training
- LoRA (low-rank adaptation) training

Since RDT2-VQ is based on Qwen2.5-VL, you are free to apply using other techniques including (e.g., fsdp, quantization) by following Qwen2.5-VL's fine-tunig practices.
We provide example fine-tuning scripts for [full-parameter](scripts/finetune_full_param.sh) and [LoRA](scripts/finetune_lora.sh) fine-tuning, which you can directly use to kick off your own training. 

To provide a better understanding, we elaborate the line-by-line explanation of the full-parameter fine-tuning script ([`scripts/finetune_full_param.sh`](scripts/finetune_full_param.sh)) with our example data:

```bash
# Define your env settings here 
# e.g., nccl, network, proxy, etc.

TASK="bimanual-ur5e-example"  # Define your task name here
DATASET_CONFIG_PATH="configs/datasets/example.yaml"  # Define your dataset config path here

export TOKENIZER_ID="Qwen/Qwen2.5-VL-7B-Instruct"
export VAE_ID="robotics-diffusion-transformer/RVQActionTokenizer" 
export MODEL_ID="robotics-diffusion-transformer/RDT2-VQ"
export OUTPUT_DIR="outputs/vqvla-sft-${TASK}" # Define your output directory here

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

accelerate launch main.py \
    --deepspeed="scripts/zero1.json" \  # Deepspeed config file, you can modify it to your own using other sharding strategies
    --tokenizer_name=$TOKENIZER_ID \
    --vae_name=$VAE_ID \
    --pretrained_model_name_or_path=$MODEL_ID \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=64 \
    --eval_batch_size=32 \
    --max_train_steps=10000 \ # We suggest training less than 5 epochs to avoid overfitting, 
                              # you should estimate the number of steps for your data and set it accordingly
    --eval_strategy="no" \
    --logging_steps=25 \
    --checkpoints_total_limit=20 \
    --checkpointing_step=1000 \
    --lr_scheduler="cosine" \
    --learning_rate=1e-5 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=16 \
    --gradient_checkpointing \
    --log_level="info" \
    --report_to="wandb" \
    --lr_warmup_steps=500 \
    --dataset=$DATASET_CONFIG_PATH \
    --image_corruption \ # We suggest toggle this on for better vision robustness
    --use_default_collate_fn_for_eval
```

Although our RVQ demonstrates high generalization among both hand-held gripper data and real robot data. If you want to fine-tune on your own data with our Residual VQ as action tokenzer, 
we sincerely suggest you firstly to check the statistics of your data are within the bound of our Residual VQ, and then test the reconstruction error of your data.

<!-- **Note:** We provide a [script]() for compute normalization statistics fo action normalization for bound violation check. This can be beneficial if you are fine-tuning to a new task on a robot.  -->

#### RDT2-FM

Currently, we support fine-tuning RDT2-FM's Action Expert with DeepSpeed: We provide example fine-tuning scripts for [full-parameter action expert](scripts/finetune_rdt.sh) fine-tuning. After specifying your own [dataset config path](scripts/finetune_rdt.sh#L20) and replacing the `<repository-path>` in [full-parameter action expert](scripts/finetune_rdt.sh#L42) with your own repository path, you can directly run this script to kick off training. 

### Precision Settings

Different models have their specific precision settings:

**Action Tokenizer (Residual VQ):**

Since the size of Residual VQ is very small, we use `float32` for both training and inference.

**RDT VLM ([RDT2-VQ](https://huggingface.co/robotics-diffusion-transformer/RDT2-VQ)):**

Uses full `bfloat16` (default) following Qwen2.5-VL. You can follow the practice for [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) to adjust the precision by applying techniques like mixed precision or quantization.

<!-- **RDT Action Expert ([RDT2-FM](robotics-diffusion-transformer/RDT2-FM) \& [RDT2-FM-UltraFast](robotics-diffusion-transformer/RDT2-FM-UltraFast)):** -->
**RDT Action Expert ([RDT2-FM](robotics-diffusion-transformer/RDT2-FM)):**

Uses full `bfloat16` for both training and inference. 

## Troubleshooting

We will collect common issues and their solutions here. If you encounter an issue, please check here first. If you can't find a solution, please file an issue on the repo (see [here](CONTRIBUTING.md) for guidelines).

| Issue                                     | Resolution                                                                                                                                                                                   |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
| 🚧 In progress 🚧 | 🚧 In progress 🚧 |
