# Define your env settings here 
# e.g., nccl, network, proxy, etc.

# set your own CFLAGS and LDFLAGS here
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"


CUR_TIME=$(date +%Y%m%d_%H%M%S)

ENV_TYPE="post_train"
WANDB_PROJECT="rdt2-action-expert"
OUTPUT_DIR="outputs/rdt/${WANDB_PROJECT}/"


mkdir -p "./logs/${WANDB_PROJECT}"
LOGGING_DIR="./logs/${WANDB_PROJECT}"
LOGGING_FILE="${LOGGING_DIR}/${CUR_TIME}.log"

WDS_CONFIG_FILE="configs/datasets/example.yaml"
VISION_LANGUAGE_MODEL_NAME_OR_PATH="robotics-diffusion-transformer/RDT2-VQ"

TRAIN_BATCH_SIZE=64
SAMPLE_BATCH_SIZE=32

# delete if exists
if [ -f "$LOGGING_FILE" ]; then
    rm "$LOGGING_FILE"
    echo "Log file '$LOGGING_FILE' deleted"
else
    echo "Log file '$LOGGING_FILE' does not exist"
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# e.g. PYTHONPATH="/ssd/RDT2"
PYTHONPATH="<repository-path>" accelerate launch rdt/main.py \
    --deepspeed="scripts/zero1.json" \
    --config_path="./configs/rdt/${ENV_TYPE}.yaml" \
    --pretrained_vision_language_model_name_or_path=$VISION_LANGUAGE_MODEL_NAME_OR_PATH \
    --output_dir=$OUTPUT_DIR \
    --webdataset_config=$WDS_CONFIG_FILE \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --sample_batch_size=$SAMPLE_BATCH_SIZE \
    --max_train_steps=1000000 \
    --checkpointing_period=5000 \
    --sample_period=1000 \
    --checkpoints_total_limit=40 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --report_to=wandb 2>&1 | tee -a "$LOGGING_FILE"
