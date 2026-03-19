#!/bin/bash
set -e
if [ ! -d ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid ]; then
    bash scripts/download_ckpt.sh
    echo "Downloaded ckpt to ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid"
fi
# Install and patch transformers
uv pip install transformers==4.53.2
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
echo "transformers patched"

# Convert JAX checkpoint to PyTorch
echo "Converting JAX model to PyTorch..."
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
    --config_name pi05_droid \
    --output_path ./checkpoints/viz/pi05_droid_pytorch
echo "Done: ./checkpoints/viz/pi05_droid_pytorch"