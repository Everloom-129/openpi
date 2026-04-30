#!/usr/bin/env bash
# Launch the NVIDIA GR00T N1.7 inference server (ZMQ REP on PORT, default 5555).
#
# Run this in the Isaac-GR00T uv venv (Python 3.10, CUDA 12.8 for dGPU) — NOT
# the robocasa_sim conda env or the openpi .venv. First-time setup:
#     cd third_party/Isaac-GR00T && uv sync --all-extras
#
# Defaults to the GR00T-N1.7-DROID finetuned checkpoint. The embodiment tag
# `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` selects the OXE DROID head — note
# that despite "RELATIVE_JOINT" in the name, `action/joint_position` from
# the server is in ABSOLUTE joint angles (radians); verified against the
# model's bundled statistics.json (action mean ≈ Panda home pose, range =
# Panda joint limits). The client (`run_policy_sim_gr00t.py`) sends these
# straight through to robosuite's JOINT_POSITION controller.
#
# GPU pinning: set CUDA_VISIBLE_DEVICES to a single GPU so this doesn't
# fight the openpi or sim processes for memory. Example:
#     CUDA_VISIBLE_DEVICES=0 bash viz_sim/run_gr00t_server.sh
#
# Override knobs (env vars):
#     MODEL  HF id or local path           (default: nvidia/GR00T-N1.7-DROID)
#     TAG    embodiment tag                (default: OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT)
#     PORT   ZMQ REP port                  (default: 5555)
#     HOST   bind address                  (default: 0.0.0.0)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_DIR="${REPO_ROOT}/third_party/Isaac-GR00T"
MODEL="${MODEL:-nvidia/GR00T-N1.7-DROID}"
TAG="${TAG:-OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT}"
PORT="${PORT:-5555}"
HOST="${HOST:-0.0.0.0}"

if [ ! -d "${GR00T_DIR}" ]; then
    echo "Isaac-GR00T checkout not found at ${GR00T_DIR}" >&2
    echo "Clone it: git submodule update --init third_party/Isaac-GR00T" >&2
    exit 1
fi

echo "[gr00t-server] MODEL=${MODEL}"
echo "[gr00t-server] TAG=${TAG}"
echo "[gr00t-server] HOST=${HOST}  PORT=${PORT}"
echo "[gr00t-server] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset, all GPUs>}"

cd "${GR00T_DIR}"
exec uv run python gr00t/eval/run_gr00t_server.py \
    --host "${HOST}" \
    --port "${PORT}" \
    --embodiment-tag "${TAG}" \
    --model-path "${MODEL}"
