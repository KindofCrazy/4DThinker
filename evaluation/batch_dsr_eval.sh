#!/bin/bash
# ============================================================================
# Batch DSR Evaluation: Run parallel evaluations on multiple checkpoints
#
# Usage:
#   cd <project_root>
#   bash evaluation/batch_dsr_eval.sh
#
# Format for EXPERIMENTS array:
#   "GPU_ID experiment_name"           — DIFT checkpoint
#   "GPU_ID experiment_name:ckpt_step" — 4DRL checkpoint
#
# latent_size is auto-extracted from experiment name (e.g. latent4 -> 4)
# ============================================================================
set -uo pipefail

# ============== Experiment List (modify here) ==============
EXPERIMENTS=(
    "0 4dthinker_dift"
    "1 4dthinker_4drl:100"
)

# ============== Common Parameters ==============
OUTPUTS_ROOT="./model"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCHMARK_PATH="./raw_data/DSR_Suite-Data/benchmark.parquet"
VIDEO_ROOT="./raw_data/DSR-data/bmk_video"
CACHE_DIR="./cache"
NUM_WORKERS=4
SEED=42
TEMPERATURE=0.7
TOP_P=0.9
MAX_NEW_TOKENS=8192
RESULTS_DIR="${SCRIPT_DIR}/results"
TS="$(date +%Y%m%d_%H%M%S)"

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

echo "=============================================="
echo " Batch DSR Evaluation - ${#EXPERIMENTS[@]} experiments"
echo " Time: ${TS}"
echo "=============================================="

PIDS=()

for i in "${!EXPERIMENTS[@]}"; do
    ENTRY="${EXPERIMENTS[$i]}"

    [[ -z "${ENTRY}" || "${ENTRY}" == PLACEHOLDER_* ]] && continue

    GPU_ID="${ENTRY%% *}"
    EXP_PART="${ENTRY#* }"

    if [[ "${EXP_PART}" == *:* ]]; then
        EXP_NAME="${EXP_PART%%:*}"
        RL_CKPT="${EXP_PART##*:}"
        MODEL_PATH="${OUTPUTS_ROOT}/${EXP_NAME}/checkpoints/checkpoint-${RL_CKPT}"
        RESULTS_NAME="${EXP_NAME}-${RL_CKPT}"
    else
        EXP_NAME="${EXP_PART}"
        RL_CKPT=""
        MODEL_PATH="${OUTPUTS_ROOT}/${EXP_NAME}/checkpoints"
        RESULTS_NAME="${EXP_NAME}"
    fi

    # Extract latent_size from experiment name
    if [[ "${EXP_NAME}" =~ latent([0-9]+) ]]; then
        LATENT_SIZE="${BASH_REMATCH[1]}"
    else
        LATENT_SIZE="4"
    fi

    RESULTS_RUN_DIR="${RESULTS_DIR}/${RESULTS_NAME}"
    mkdir -p "${RESULTS_RUN_DIR}"
    OUTPUT_JSONL="${RESULTS_RUN_DIR}/dsr_eval_${TS}.jsonl"
    LOG_FILE="${RESULTS_RUN_DIR}/dsr_eval_${TS}.log"

    echo ""
    echo "[GPU ${GPU_ID}] ${RESULTS_NAME}"
    echo "  MODEL_PATH:  ${MODEL_PATH}"
    echo "  LATENT_SIZE: ${LATENT_SIZE}"
    echo "  LOG_FILE:    ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python "${SCRIPT_DIR}/dsr_eval.py" \
        --model_path "${MODEL_PATH}" \
        --benchmark_path "${BENCHMARK_PATH}" \
        --video_root "${VIDEO_ROOT}" \
        --cache_dir "${CACHE_DIR}" \
        --output_file "${OUTPUT_JSONL}" \
        --num_workers "${NUM_WORKERS}" \
        --seed "${SEED}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --latent_size "${LATENT_SIZE}" \
        > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}"
done

echo ""
echo "=============================================="
echo " Started ${#PIDS[@]} evaluation tasks"
echo "=============================================="
