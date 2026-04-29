#!/bin/bash
set -e
if [ ! -d ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero ]; then
    uv run python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')"
    echo "Downloaded ckpt to ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero"
fi

# Convert JAX checkpoint to PyTorch
echo "Converting JAX model to PyTorch..."
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --config_name pi05_libero \
    --output_path ./checkpoints/viz/pi05_libero_pytorch
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets \
    ./checkpoints/viz/pi05_libero_pytorch/assets
echo "Done: ./checkpoints/viz/pi05_libero_pytorch"
