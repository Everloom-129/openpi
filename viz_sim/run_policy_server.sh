#!/usr/bin/env bash
# Launch the openpi pi0.5-DROID policy as a websocket server on port 8000.
# Run this in the OPENPI .venv (NOT the robocasa_sim conda env).
#
# Prerequisite: the converted pytorch checkpoint must already exist. If not,
# create it with:
#     bash scripts/get_pi05_droid_torch.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${CKPT:-${REPO_ROOT}/checkpoints/viz/pi05_droid_pytorch}"
PORT="${PORT:-8000}"

if [ ! -d "${CKPT}" ]; then
    echo "Checkpoint not found at ${CKPT}"
    echo "Run: bash scripts/get_pi05_droid_torch.sh"
    exit 1
fi

cd "${REPO_ROOT}"
# Use the attention-capturing wrapper so the websocket response includes
# text→image attention alongside actions. Falls back gracefully if the
# buffer is empty.
exec uv run viz_sim/serve_policy_attn.py \
    --port="${PORT}" \
    --default_prompt="pick up the cube" \
    --config=pi05_droid \
    --dir="${CKPT}"
