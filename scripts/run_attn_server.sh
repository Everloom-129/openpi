#!/bin/bash
# Launch pi05_droid_torch server with real-time attention visualization.
#
# Usage:
#   bash scripts/run_attn_server.sh [checkpoint_dir] [device]
#
# Defaults:
#   checkpoint_dir = checkpoints/viz/pi05_droid_pytorch
#   device         = cuda:0

CKPT_DIR="${1:-checkpoints/viz/pi05_droid_pytorch}"
DEVICE="${2:-cuda:0}"

echo "=== Pi0.5 Attention Server ==="
echo "  Checkpoint: $CKPT_DIR"
echo "  Device:     $DEVICE"
echo "  Policy WS:  ws://localhost:8000" 
echo "  Attention:  http://localhost:8001"
echo "=============================="

uv run python scripts/serve_policy_with_attn.py \
    --config pi05_droid \
    --checkpoint-dir "$CKPT_DIR" \
    --device "$DEVICE" \
    --port 8000 \
    --viz-port 8001 \
    --viz-layer 7 \
    --viz-head max
