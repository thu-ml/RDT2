# Run Inference on UR5e Robot

Make sure you've set up the hardware accroding to our [Hardware Guide](https://docs.google.com/document/d/1HUeM4Wlt4PyINoEwci-hxm8U9wAxiPMgR3sHyaOAsck/edit?tab=t.0#heading=h.sbdalb8w1kk1).

Install the required dependencies:

```bash
pip install -r requirements/ur5e.txt
```

## 1. Set up configurations.

- UR5e Robot: obtain Robot IP and modify configs/robots/eval_bimanual_ur5e_config.yaml/robot/robot_ip
- HikRobot Camera: run the following code to get camera serials:
    ```
    python deploy/get_camera_serials.py
    ```
    and modify configs/robots/eval_bimanual_ur5e_config.yaml/cameras/serials
- Zhixing Gripper: run the following code to get gripper serial:
    ```
    python deploy/get_gripper_serials.py
    ```
    and modify configs/robots/eval_bimanual_ur5e_config.yaml/grippers/serial.

## 2. Run Inference

Make sure the calibration matrix is properly set in configs/robots/eval_bimanual_ur5e_config.yaml/tx_tracker_to_tcp.

Run the following code to reset robot and gripper:

IMPORTANT: This script makes the robot reset to an initial pose; before running the script, please ensure the robot is in a safe position and the workspace is free of obstacles.

```bash
python deploy/reset_robot_gripper.py \
    --robot_config=configs/robots/eval_bimanual_ur5e_config.yaml
```

### Run Inference with RDT-VQ

Run the following code to start inference:

```bash
python deploy/inference_real_vq.py \
    --data_config=configs/bimanual_video_data.yaml \
    --robot_config=configs/robots/eval_bimanual_ur5e_config.yaml\
    -v <your_vqvae_checkpoint>  \     # e.g., "robotics-diffusion-transformer/RVQActionTokenizer"
    -i <your_rdt2_vq_checkpoint> \    # e.g., "robotics-diffusion-transformer/RDT2-VQ"
    -o <your_output_directory> \
    --instruction <your_instruction> \
```

You can also run the following code to start a multi-instruction inference process. Press "S" to stop the current inference, input a new instruction, then press enter to continue inference with the new instruction

```bash
python deploy/inference_real_vq.py \
    --data_config=configs/bimanual_video_data.yaml \
    --robot_config=configs/robots/eval_bimanual_ur5e_config.yaml\
    -v <your_vqvae_checkpoint>  \     # e.g., "robotics-diffusion-transformer/RVQActionTokenizer"
    -i <your_rdt2_vq_checkpoint> \    # e.g, "robotics-diffusion-transformer/RDT2-VQ"
    -o <your_output_directory> \
    --instruction <your_instruction> \
    --interact
```

You can use `--use_vllm` to accelerate inference with vllm.

### Run Inference with RDT-FM

Run the following code to start inference:

```bash
python deploy/inference_real_fm.py \
    --input <your_rdt_action_expert_checkpoint> \    # e.g., "robotics-diffusion-transformer/RDT2-FM"
    --output <your_output_directory> \
    --pretrained_vision_language_model_name_or_path <your_rdt2_vq_checkpoint> \    # e.g., "robotics-diffusion-transformer/RDT2-VQ"
    --normalizer_path <your_normalizer_checkpoint> \ 
    --model_config=configs/rdt/post_train.yaml \
    --data_config=configs/bimanual_video_data.yaml \
    --robot_config=configs/robots/eval_bimanual_ur5e_config.yaml \
    --instruction "Pick up the pink snack bag with the right hand." \
```

You can also add "--interact" to start a multi-instruction inference process.
