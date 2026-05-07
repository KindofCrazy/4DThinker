#!/bin/bash
# 4DThinker RL Training (GRPO, 8-GPU)
#
# Usage:
#   cd <project_root>
#   bash 4drl/run_scripts/train_4dthinker.sh

# ============== Experiment Config ==============
EXP_NAME="4dthinker_rl_grpo"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL settings for single-node multi-GPU
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=300
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTHONUNBUFFERED=1

# ============== Paths (relative to project root) ==============
# Resolve to absolute paths before cd
ROOT_DIR="$(pwd)"
MODEL_PATH="${ROOT_DIR}/model/dift/checkpoints"
DATA_PATH="${ROOT_DIR}/data/4drl_data_filtered.jsonl"
OUTPUT_DIR="${ROOT_DIR}/model/4drl"

# ============== DeepSpeed Multi-GPU Config ==============
NUM_GPUS=8
DS_CONFIG="${ROOT_DIR}/4drl/configs/zero2.json"

# ============== Training Hyperparams ==============
LR=1e-6
EPOCHS=1
BATCH_SIZE=8
GRAD_ACCUM=2
NUM_GENERATIONS=8
MAX_COMPLETION_LENGTH=8192
BETA=0.01
MAX_PIXELS=200704    # 256*28*28
MIN_PIXELS=784       # 28*28

# ============== Logging & Saving ==============
LOGGING_STEPS=1
SAVE_STEPS=100

# ============== Debug ==============
export DEBUG_MODE="true"
mkdir -p "${OUTPUT_DIR}/${EXP_NAME}/logs"
export LOG_PATH="${OUTPUT_DIR}/${EXP_NAME}/logs/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

# ============== Pre-extract video frames ==============
FRAME_CACHE_DIR="${ROOT_DIR}/cache/frame_cache"
mkdir -p "${FRAME_CACHE_DIR}"

cd "${ROOT_DIR}/4drl/src/open-r1-multimodal"

# ============== Launch ==============
echo ">>> Pre-extracting video frames (single process) ..."
python -c "
import sys, os
sys.path.insert(0, 'src')
from open_r1.grpo_4dthinker import load_4dthinker_dataset
ds = load_4dthinker_dataset('${DATA_PATH}', '${FRAME_CACHE_DIR}')
print(f'Pre-extraction done. Dataset size: {len(ds)}', flush=True)
"
echo ">>> Frame extraction complete. Starting distributed training ..."

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12349 \
    src/open_r1/grpo_4dthinker.py \
        --output_dir "${OUTPUT_DIR}/${EXP_NAME}" \
        --model_name_or_path "${MODEL_PATH}" \
        --data_file_paths "${DATA_PATH}" \
        --reward_funcs accuracy format \
        --reward_weights 1.0 0.2 \
        --max_pixels ${MAX_PIXELS} \
        --min_pixels ${MIN_PIXELS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --gradient_checkpointing true \
        --num_train_epochs ${EPOCHS} \
        --bf16 \
        --learning_rate ${LR} \
        --beta ${BETA} \
        --num_generations ${NUM_GENERATIONS} \
        --max_completion_length ${MAX_COMPLETION_LENGTH} \
        --logging_steps ${LOGGING_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --run_name "${EXP_NAME}" \
        --data_seed 42 \
        --report_to tensorboard \
        --deepspeed "${DS_CONFIG}" \
        --freeze_vision_modules false \
        --dataset-name this_is_not_used

# ============== View TensorBoard ==============
# tensorboard --logdir ${OUTPUT_DIR}/${EXP_NAME}/tensorboard --port 6006
