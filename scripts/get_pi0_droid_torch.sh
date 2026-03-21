#!/bin/bash
set -e
if [ ! -d ~/.cache/openpi/openpi-assets/checkpoints/pi0_droid ]; then
    uv run python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi0_droid')"
    echo "Downloaded ckpt to ~/.cache/openpi/openpi-assets/checkpoints/pi0_droid"
fi
# # Install and patch transformers (only for the first time)
# uv pip install transformers==4.53.2
# cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
# echo "transformers patched"

# Convert JAX checkpoint to PyTorch
echo "Converting JAX model to PyTorch..."
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi0_droid \
    --config_name pi0_droid \
    --output_path ./checkpoints/viz/pi0_droid_pytorch
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi0_droid/assets \
    ./checkpoints/viz/pi0_droid_pytorch/assets
echo "Done: ./checkpoints/viz/pi0_droid_pytorch"
