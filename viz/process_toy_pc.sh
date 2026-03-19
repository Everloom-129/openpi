#!/bin/bash
set -e

CAMERA="right"
DATA_ROOT="/data3/tonyw/toy_cube_benchmark/cube_gold"
RESULTS_ROOT="/data3/tonyw/toy_cube_benchmark/pi05_vis/cube_gold/${CAMERA}"

# Producer-consumer knobs
IO_WORKERS=2
MAX_INFLIGHT_LOADS=32

echo "=========================================="
echo "Producer-Consumer Attention Pipeline"
echo "=========================================="
echo "DATA_ROOT:    ${DATA_ROOT}"
echo "RESULTS_ROOT: ${RESULTS_ROOT}"
echo "IO_WORKERS:   ${IO_WORKERS}"
echo "INFLIGHT:     ${MAX_INFLIGHT_LOADS}"
echo "CAMERA:       ${CAMERA}"
echo ""

CMD="uv run python viz/pipeline_pc.py $DATA_ROOT $RESULTS_ROOT \
  --io-workers $IO_WORKERS \
  --max-inflight-loads $MAX_INFLIGHT_LOADS \
  --camera $CAMERA"

echo "Command: $CMD"
echo ""
read -p "Press Enter to start..."

eval "$CMD"
