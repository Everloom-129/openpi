#!/usr/bin/env bash
# Launch the NVIDIA GR00T N1.7 inference server (ZMQ REP on PORT, default 5555).
#
# Run this in the Isaac-GR00T uv venv (Python 3.10, CUDA 12.8 for dGPU) — NOT
# the robocasa_sim conda env or the openpi .venv. First-time setup:
#     cd third_party/Isaac-GR00T && uv sync --all-extras
#
# Defaults to the base N1.7 checkpoint with the DROID embodiment tag (zero-shot).
# Set MODEL=nvidia/GR00T-N1.7-DROID for the finetuned checkpoint.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_DIR="${REPO_ROOT}/third_party/Isaac-GR00T"
MODEL="${MODEL:-nvidia/GR00T-N1.7-DROID}"
TAG="${TAG:-OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT}"
PORT="${PORT:-5555}"

cd "${GR00T_DIR}"
exec uv run python gr00t/eval/run_gr00t_server.py \
    --port "${PORT}" \
    --embodiment-tag "${TAG}" \
    --model-path "${MODEL}" \
