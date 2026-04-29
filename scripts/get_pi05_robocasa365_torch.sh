#!/bin/bash
# Download the robocasa365 pi0.5 multitask checkpoint (75k steps) from HuggingFace
# and convert it to PyTorch format under ./checkpoints/viz/pi05_robocasa365_pytorch.
#
# Source: https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/pi05_pretrain_human300/multitask_learning/75000
#
# The ckpt is a pi0.5 model (action_dim=32, same arch as pi05_droid) finetuned on
# the robocasa365 multi-task suite. We use `pi05_droid` only to drive the
# weight-conversion math — the runtime obs/action transforms at serving time
# come from whatever robocasa-shaped policy wrapper you use. The robocasa
# norm_stats.json is copied into the converted dir's `assets/`.
set -e

REPO_ID="robocasa/robocasa365_checkpoints"
SUBPATH="pi05_pretrain_human300/multitask_learning/75000"
HF_CACHE_BASE="${HOME}/.cache/openpi/hf/${REPO_ID//\//__}"
LOCAL_CKPT="${HF_CACHE_BASE}/${SUBPATH}"
OUT="./checkpoints/viz/pi05_robocasa365_pytorch"

if [ ! -d "${LOCAL_CKPT}/params" ]; then
    echo "Downloading ${REPO_ID}:${SUBPATH} -> ${HF_CACHE_BASE}"
    uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${REPO_ID}',
    local_dir='${HF_CACHE_BASE}',
    allow_patterns=['${SUBPATH}/_CHECKPOINT_METADATA',
                    '${SUBPATH}/params/**',
                    '${SUBPATH}/assets/**'],
)
"
else
    echo "Found existing checkpoint at ${LOCAL_CKPT}"
fi

echo "Converting JAX -> PyTorch (using pi05_droid config for arch)..."
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir "${LOCAL_CKPT}" \
    --config_name pi05_droid \
    --output_path "${OUT}"

# Copy the robocasa-specific norm_stats so the converted ckpt is self-contained.
mkdir -p "${OUT}/assets"
cp -r "${LOCAL_CKPT}/assets/." "${OUT}/assets/"

echo "Done: ${OUT}"
