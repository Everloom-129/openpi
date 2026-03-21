#!/bin/bash
# NOTE: pi0_fast_droid uses the Pi0-FAST architecture (autoregressive + FAST tokenizer).
# There is no PyTorch implementation for Pi0-FAST in this repo, so this script only
# downloads the JAX checkpoint. Use it for JAX inference only.
set -e
if [ ! -d ~/.cache/openpi/openpi-assets/checkpoints/pi0_fast_droid ]; then
    uv run python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi0_fast_droid')"
    echo "Downloaded ckpt to ~/.cache/openpi/openpi-assets/checkpoints/pi0_fast_droid"
else
    echo "Already downloaded: ~/.cache/openpi/openpi-assets/checkpoints/pi0_fast_droid"
fi
echo "Done. JAX checkpoint is at: ~/.cache/openpi/openpi-assets/checkpoints/pi0_fast_droid"
echo "No PyTorch conversion available for Pi0-FAST."
