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
    and modify configs/robots/eval_bimanual_fr3_config.yaml/cameras/serials
- Zhixing Gripper: run the following code to get gripper serial:
    ```
    python deploy/get_gripper_serial.py
    ```
    and modify configs/robots/eval_bimanual_fr3_config.yaml/grippers/serial.

## 2. Run Inference

Make sure the calibration matrix is properly set in configs/robots/eval_bimanual_ur5e_config.yaml/tx_tracker_to_tcp.

Run the following code to reset robot and gripper:

```bash
python deploy/reset_robot_gripper.py \
    --robot_config=configs/robots/eval_bimanual_ur5e_config.yaml
```

Run the following code to run inference:

```bash
python deploy/inference_real_vq.py \
    --data_config=configs/bimanual_video_data.yaml \
    --robot_config=configs/robots/eval_bimanual_ur5e_config.yaml\
    -v <your_vqvae_checkpoint>  \ 
    -i <your_vqvla_checkpoint> \
    -o <your_output_directory> \
    --instruction <your_instruction> \
```

Press "C" to start inference, press "Q" to exit.

You can also run the following code to start a multi-instruction inference process. Press "S" to stop the current inference, input a new instruction, then press "C" to start inference.

```bash
python deploy/inference_real_vq.py \
    --data_config=configs/bimanual_video_data.yaml \
    --robot_config=configs/robots/eval_bimanual_ur5e_config.yaml\
    -v <your_vqvae_checkpoint>  \ 
    -i <your_vqvla_checkpoint> \
    -o <your_output_directory> \
    --instruction <your_instruction> \
    --interact
```
