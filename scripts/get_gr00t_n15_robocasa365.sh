#!/bin/bash
# Download the robocasa365 GR00T N1.5 multitask checkpoint (120k steps) from HuggingFace.
# No conversion — GR00T ckpts are HuggingFace-format safetensors that load directly
# into the Isaac-GR00T server via `--model-path`.
#
# Source: https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/multitask_learning/checkpoint-120000
#
# Skips optimizer/rng/scheduler state (training-only) to save bandwidth.
set -e

REPO_ID="robocasa/robocasa365_checkpoints"
SUBPATH="gr00t_n1-5/multitask_learning/checkpoint-120000"
OUT="./checkpoints/viz/gr00t_n15_robocasa365"

if [ -d "${OUT}" ] && [ -f "${OUT}/model.safetensors.index.json" ]; then
    echo "Found existing checkpoint at ${OUT}"
    exit 0
fi

echo "Downloading ${REPO_ID}:${SUBPATH} -> ${OUT}"
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${REPO_ID}',
    local_dir='${OUT}',
    allow_patterns=[
        '${SUBPATH}/config.json',
        '${SUBPATH}/experiment_cfg/**',
        '${SUBPATH}/model-*.safetensors',
        '${SUBPATH}/model.safetensors.index.json',
        '${SUBPATH}/trainer_state.json',
    ],
)
"

# Flatten so OUT/ is the model dir (drop the long subpath prefix).
if [ -d "${OUT}/${SUBPATH%/*}" ]; then
    mv "${OUT}/${SUBPATH}"/* "${OUT}/"
    rm -rf "${OUT}/${SUBPATH%%/*}"
fi

echo "Done: ${OUT}"
echo "Run with: MODEL=${OUT} bash viz_sim/run_gr00t_server.sh"
