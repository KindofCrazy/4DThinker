#!/bin/bash
# Loop running process_minibatch.py until all videos are processed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/process_minibatch.py"
LOG_FILE="${SCRIPT_DIR}/run.log"

# Counter
batch_count=0

echo "========================================" | tee -a "$LOG_FILE"
echo "Starting batch loop - $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

while true; do
    batch_count=$((batch_count + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "Batch #${batch_count} started - $(date)" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    
    # Run Python script
    python "$PYTHON_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}
    
    echo "Batch #${batch_count} done, exit code: ${exit_code}" | tee -a "$LOG_FILE"
    
    if [ $exit_code -eq 0 ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        echo "All videos processed! - $(date)" | tee -a "$LOG_FILE"
        echo "Total batches run: ${batch_count}" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        break
    elif [ $exit_code -eq 1 ]; then
        echo "Remaining videos, continuing next batch..." | tee -a "$LOG_FILE"
        # Brief pause to avoid being too frequent
        sleep 2
    else
        echo "Error (exit code: ${exit_code}), retrying in 5s..." | tee -a "$LOG_FILE"
        sleep 5
    fi
done

echo "Script execution complete" | tee -a "$LOG_FILE"

