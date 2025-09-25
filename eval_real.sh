python deploy/inference_real_vq.py \
    --data_config=configs/bimanual_video_data.yaml \
    --robot_config=configs/robots/eval_bimanual_ur5e_config.yaml \
    -v "/home/user/code/FastVLA/outputs/vqvae_hf" \
    -i "/home/user/code/FastVLA/outputs/vqvla-bimanual-umi-pretrain/checkpoint-126000" \
    -o "/home/user/code/FastVLA/outputs/vqvla-bimanual-umi-pretrain/checkpoint-126000/real" \
    --instruction "Pick up the blue doll on the table with the left hand." \