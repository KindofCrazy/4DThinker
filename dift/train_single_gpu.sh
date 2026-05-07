#!/bin/bash
# 4DThinker SFT Training (Single GPU, no DeepSpeed)
#
# Usage:
#   cd <project_root>
#   CUDA_VISIBLE_DEVICES=0 bash dift/train_single_gpu.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============== Experiment Config ==============
EXP_NAME="4dthinker_stage1_7B_single_gpu"
MODEL_PATH="./models/Qwen2.5-VL-7B-Instruct"       # Path to base model
DATA_PATH="./data/dift_data.jsonl"                   # Path to training data
OUTPUT_DIR="./model/dift"

# ============== Training Hyperparams ==============
STAGE="stage1"
TASK="4dthinker"
FPS=1
EPOCHS=1
LR=1e-5
WARMUP=10
WEIGHT_DECAY=0.01
BATCH_SIZE=1
GRAD_ACCUM=8
LATENT_SIZE=4
CE_LOSS_WEIGHT=0.1
SIM_LOSS_WEIGHT=1.0

# ============== Logging & Saving ==============
LOGGING_STEPS=1
SAVE_STEPS=1000
SAVE_TOTAL_LIMIT=1

# ============== Run single-GPU (no DeepSpeed) ==============
python dift/src/main.py \
    --exp_name ${EXP_NAME} \
    --model ${MODEL_PATH} \
    --stage ${STAGE} \
    --task ${TASK} \
    --fps ${FPS} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --warmup_steps ${WARMUP} \
    --weight_decay ${WEIGHT_DECAY} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --latent_size ${LATENT_SIZE} \
    --ce_loss_weight ${CE_LOSS_WEIGHT} \
    --sim_loss_weight ${SIM_LOSS_WEIGHT} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT}

# ============== View TensorBoard ==============
# tensorboard --logdir ${OUTPUT_DIR}/${EXP_NAME}/tensorboard --port 6006
