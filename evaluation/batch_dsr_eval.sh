#!/usr/bin/env bash
# Sharded DSR evaluation. One shard runs on one GPU; shard jsonl files are
# merged into a single jsonl after all workers finish.

set -u

# =============================================================================
# Resolve Script Locations
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =============================================================================
# User Configuration
# =============================================================================
# ---------------- Run Identity ----------------
# Override RUN_NAME to distinguish different checkpoints/settings in results.
RUN_NAME="${RUN_NAME:-dift_flash_batch}"

# ---------------- Model Args ----------------
MODEL_PATH="${MODEL_PATH:-model/dift/checkpoints}"
LATENT_SIZE="${LATENT_SIZE:-4}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

# ---------------- Data Args ----------------
BENCHMARK_PATH="${BENCHMARK_PATH:-./raw_data/DSR_Suite-Data/benchmark.parquet}"
VIDEO_ROOT="${VIDEO_ROOT:-./raw_data/DSR-data/bmk_video}"
CACHE_DIR="${CACHE_DIR:-./cache}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/results}"

# ---------------- Eval / Generation Args ----------------
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

# ---------------- Runtime Args ----------------
# GPUS_CSV also controls shard count: one eval process per listed GPU.
GPUS_CSV="${GPUS_CSV:-0,1,2,3,4,5}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CONDA_ENV="${CONDA_ENV:-4dthinker}"
CONDA_SH="${CONDA_SH:-/home/yk/miniconda3/etc/profile.d/conda.sh}"
CUDA_HOME_OVERRIDE="${CUDA_HOME_OVERRIDE:-/home/yk/miniconda3/envs/4dthinker/lib/python3.10/site-packages/nvidia/cu13}"

# ---------------- Derived Paths ----------------
# The run directory starts with the batch timestamp by convention.
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULTS_ROOT}/${TS}_${RUN_NAME}"
SHARD_DIR="${RUN_DIR}/shards"
JSONL_DIR="${RUN_DIR}/shard_jsonl"
LOG_DIR="${RUN_DIR}/logs"
MERGED_JSONL="${RUN_DIR}/dsr_eval_merged.jsonl"
SUMMARY_JSON="${RUN_DIR}/dsr_eval_merged_summary.json"
RUN_INFO="${RUN_DIR}/run_info.txt"

# =============================================================================
# Prepare Workspace
# =============================================================================
cd "${PROJECT_ROOT}"
mkdir -p "${SHARD_DIR}" "${JSONL_DIR}" "${LOG_DIR}"

# =============================================================================
# Prepare Runtime Environment
# =============================================================================
if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME="${CUDA_HOME_OVERRIDE}"
export PATH="${CUDA_HOME_OVERRIDE}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME_OVERRIDE}/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME_OVERRIDE}/lib:${LIBRARY_PATH:-}"

# =============================================================================
# Resolve Shard Layout
# =============================================================================
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
NUM_SHARDS="${#GPUS[@]}"

# =============================================================================
# Record Run Metadata
# =============================================================================
{
    echo "timestamp=${TS}"
    echo "run_name=${RUN_NAME}"
    echo "model_path=${MODEL_PATH}"
    echo "benchmark_path=${BENCHMARK_PATH}"
    echo "video_root=${VIDEO_ROOT}"
    echo "gpus=${GPUS_CSV}"
    echo "num_shards=${NUM_SHARDS}"
    echo "max_new_tokens=${MAX_NEW_TOKENS}"
    echo "latent_size=${LATENT_SIZE}"
    echo "attn_implementation=${ATTN_IMPLEMENTATION}"
} | tee "${RUN_INFO}"

# =============================================================================
# Split Benchmark
# =============================================================================
python "${SCRIPT_DIR}/batch_helpers/shard_and_merge.py" shard \
    --benchmark-path "${BENCHMARK_PATH}" \
    --out-dir "${SHARD_DIR}" \
    --num-shards "${NUM_SHARDS}"

# =============================================================================
# Launch One Eval Worker Per Shard
# =============================================================================
PIDS=()
for SHARD in $(seq 0 "$((NUM_SHARDS - 1))"); do
    GPU="${GPUS[$SHARD]}"
    SHARD_BENCHMARK="${SHARD_DIR}/benchmark_shard_${SHARD}_of_${NUM_SHARDS}.parquet"
    SHARD_JSONL="${JSONL_DIR}/dsr_eval_shard_${SHARD}_of_${NUM_SHARDS}.jsonl"
    SHARD_LOG="${LOG_DIR}/dsr_eval_shard_${SHARD}_of_${NUM_SHARDS}.log"

    CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT_DIR}/dsr_eval.py" \
        --model_path "${MODEL_PATH}" \
        --benchmark_path "${SHARD_BENCHMARK}" \
        --video_root "${VIDEO_ROOT}" \
        --cache_dir "${CACHE_DIR}" \
        --frame_root "/tmp/dsr_eval_frames_${TS}_shard_${SHARD}" \
        --output_file "${SHARD_JSONL}" \
        --num_workers "${NUM_WORKERS}" \
        --seed "${SEED}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --latent_size "${LATENT_SIZE}" \
        --attn_implementation "${ATTN_IMPLEMENTATION}" \
        > "${SHARD_LOG}" 2>&1 &

    PIDS+=("$!")
    echo "started shard=${SHARD} gpu=${GPU} pid=${PIDS[-1]} log=${SHARD_LOG}"
done

# =============================================================================
# Wait For Workers
# =============================================================================
FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        FAILED=1
    fi
done

# =============================================================================
# Merge Shard Outputs
# =============================================================================
python "${SCRIPT_DIR}/batch_helpers/shard_and_merge.py" merge \
    --jsonl-dir "${JSONL_DIR}" \
    --out-jsonl "${MERGED_JSONL}" \
    --summary-json "${SUMMARY_JSON}"

# =============================================================================
# Final Status
# =============================================================================
if [[ "${FAILED}" -ne 0 ]]; then
    echo "one_or_more_shards_failed"
    exit 1
fi

echo "batch_eval_done"
