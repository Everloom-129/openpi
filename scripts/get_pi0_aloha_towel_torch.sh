#!/bin/bash
# pi0_aloha_towel: pi0 model fine-tuned on ALOHA data for towel folding.
# Note: There is no generic "pi0_aloha" or "pi05_aloha" public checkpoint.
# The available ALOHA checkpoints are:
#   pi0_aloha_towel      -> gs://openpi-assets/checkpoints/pi0_aloha_towel
#   pi0_aloha_tupperware -> gs://openpi-assets/checkpoints/pi0_aloha_tupperware
#   pi0_aloha_pen_uncap  -> gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap
# This script handles pi0_aloha_towel; copy and adjust CKPT_NAME for the others.
set -e
if [ -z "$1" ]; then
    echo "Usage: $0 <CKPT_NAME>"
    echo "Available pi0_aloha checkpoints:"
    echo "  towel"
    echo "  tupperware"
    echo "  pen_uncap"
    exit 1
fi
CKPT_NAME="pi0_aloha_${1}"
echo "CKPT_NAME: ${CKPT_NAME}"
if [ ! -d ~/.cache/openpi/openpi-assets/checkpoints/${CKPT_NAME} ]; then
    uv run python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/${CKPT_NAME}')"
    echo "Downloaded ckpt to ~/.cache/openpi/openpi-assets/checkpoints/${CKPT_NAME}"
fi

# Convert JAX checkpoint to PyTorch
echo "Converting JAX model to PyTorch..."
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/${CKPT_NAME} \
    --config_name ${CKPT_NAME} \
    --output_path ./checkpoints/viz/${CKPT_NAME}_pytorch
echo "Done: ./checkpoints/viz/${CKPT_NAME}_pytorch"
