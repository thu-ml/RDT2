# Define your env settings here 
# e.g., nccl, network, proxy, etc.

TASK="bimanual-posttrain-5task"
DATASET_CONFIG_PATH="configs/datasets/posttrain/posttrain_5task.yaml"

export TOKENIZER_ID="Qwen/Qwen2.5-VL-7B-Instruct"
export VAE_ID="outputs/vqvae_hf"    # TODO: modify to huggingface link
export MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"   # TODO: modify to RDT2-VQ
export OUTPUT_DIR="outputs/vqvla-sft-${TASK}"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

accelerate launch main.py \
    --deepspeed="scripts/zero1.json" \
    --tokenizer_name=$TOKENIZER_ID \
    --vae_name=$VAE_ID \
    --pretrained_model_name_or_path=$MODEL_ID \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=64 \
    --eval_batch_size=32 \
    --max_train_steps=10000 \
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
    --image_corruption \
    --use_default_collate_fn_for_eval