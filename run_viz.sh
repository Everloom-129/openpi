# Run pi05 at PAL EXX 

# Here is server computing side, 
# you also need to run the pi05 on franka laptop

# Memory optimization for JAX
# Controls the fraction of GPU memory JAX will use (default is 0.75)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
# Alternative: Prevent JAX from preallocating memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# conda activate openpi
# Restrict to use only GPU 1 to prevent memory duplication across GPUs
export CUDA_VISIBLE_DEVICES=1

# Additional JAX optimizations
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-12.3/lib64"

#  Link torch to with correct cuda version for UV
export LD_LIBRARY_PATH=/home/exx/Projects/openpi-main/.venv/lib/python3.11/site-packages/torch/lib/../../nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# uv run, notice the program will not halt
# uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
uv run viz/attn_map.py


