#!/usr/bin/env bash
# Launch an openpi pi0/pi0.5 policy as a websocket server on port 8000.
# Run this in the OPENPI .venv (NOT the robocasa_sim conda env).
#
# Defaults to pi0.5-DROID. Override with env vars to serve another checkpoint:
#     CONFIG=pi05_libero \
#     CKPT=$PWD/checkpoints/viz/pi05_libero_pytorch \
#     PROMPT="pick up the alphabet soup and place it in the basket" \
#     bash viz_sim/run_pi0_policy_server.sh
#
# Prerequisite: the converted pytorch checkpoint must already exist. If not,
# create it with one of:
#     bash scripts/get_pi05_droid_torch.sh
#     bash scripts/get_pi05_libero_torch.sh
#     bash scripts/get_pi0_droid_torch.sh
#     bash scripts/get_pi0_aloha_towel_torch.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="pi05_libero"  # pi05_libero, pi0_droid, pi0_aloha_towel, etc.
CKPT="${CKPT:-${REPO_ROOT}/checkpoints/viz/${CONFIG}_pytorch}"
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-pick up the cube}"

echo -e "\033[1;33mServing ${CONFIG} from ${CKPT} on port ${PORT} with prompt: ${PROMPT}\033[0m"
read -p "Press Enter to continue"

if [ ! -d "${CKPT}" ]; then
    echo "Checkpoint not found at ${CKPT}"
    echo "Run the matching scripts/get_*_torch.sh script for config=${CONFIG}."
    exit 1
fi

cd "${REPO_ROOT}"
# Use the attention-capturing wrapper so the websocket response includes
# text→image attention alongside actions. Falls back gracefully if the
# buffer is empty.
exec uv run viz_sim/serve_policy_attn.py \
    --port="${PORT}" \
    --default_prompt="${PROMPT}" \
    --config="${CONFIG}" \
    --dir="${CKPT}"
