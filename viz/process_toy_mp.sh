#!/bin/bash
# Multi-process version of process_toy.sh
# Uses multiple workers to parallelize episode processing across GPUs

# Camera side
CAMERA="left"  # or "right"
DATA_ROOT="/data3/tonyw/toy_cube_benchmark/cube_gold"
RESULTS_ROOT="/data3/tonyw/toy_cube_benchmark/pi05_vis/cube_gold/${CAMERA}"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - MODIFY THESE VALUES
# ═══════════════════════════════════════════════════════════════════════════════

# Number of workers (default: auto-detect, 1 per available GPU)
# CRITICAL: Each worker needs ~16-18GB GPU memory!
# RTX A6000 (48GB) → Recommend max 2 workers per GPU
NUM_WORKERS=2

# Which GPUs to use (default: auto-detect GPUs with >25GB free)
# Uncomment to manually specify:
# GPUS="0"      # Use only GPU 0
# GPUS="1"      # Use only GPU 1
# GPUS="0,1"    # Use both GPUs



# Set bash to exit immediately if any command fails 
set -e

# ═══════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

echo "=========================================="
echo "Multi-Process Attention Pipeline"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  DATA_ROOT:    ${DATA_ROOT}"
echo "  RESULTS_ROOT: ${RESULTS_ROOT}"
echo "  NUM_WORKERS:  ${NUM_WORKERS}"
echo "  CAMERA:       ${CAMERA}"
echo ""

echo "Checking GPU status..."
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | while IFS=, read -r idx name used free total util; do
    echo "  GPU $idx [$name]:"
    echo "    Memory: ${used}MB used, ${free}MB free (${total}MB total)"
    echo "    GPU Util: ${util}%"
done || echo "  (nvidia-smi not available or failed)"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD COMMAND
# ═══════════════════════════════════════════════════════════════════════════════

CMD="uv run python viz/pipeline_mp.py $DATA_ROOT $RESULTS_ROOT --no-counterfactual --camera $CAMERA"

if [ ! -z "$NUM_WORKERS" ]; then
    CMD="$CMD --workers $NUM_WORKERS"
fi

if [ ! -z "$GPUS" ]; then
    CMD="$CMD --gpus $GPUS"
fi

echo "Command: $CMD"
echo ""
echo "⚠️  WARNINGS:"
echo "  - Each worker uses ~16-18GB GPU memory"
echo "  - ${NUM_WORKERS} workers × 16GB = ~$((NUM_WORKERS * 16))GB total needed"
echo "  - If you get OOM errors, reduce NUM_WORKERS in this script"
echo "  - Recommended: Kill other GPU processes first (streamlit, servers, etc.)"
echo ""
read -p "Press Enter to start (Ctrl+C to cancel)..."

# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

$CMD
