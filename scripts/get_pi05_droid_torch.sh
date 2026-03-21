#!/bin/bash
set -e
if [ ! -d ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid ]; then
    uv run python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_droid')"
    echo "Downloaded ckpt to ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid"
fi

# Convert JAX checkpoint to PyTorch
echo "Converting JAX model to PyTorch..."
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
    --config_name pi05_droid \
    --output_path ./checkpoints/viz/pi05_droid_pytorch
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets \
    ./checkpoints/viz/pi05_droid_pytorch/assets
echo "Done: ./checkpoints/viz/pi05_droid_pytorch"